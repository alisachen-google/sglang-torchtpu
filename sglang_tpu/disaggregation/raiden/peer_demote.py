# SPDX-License-Identifier: Apache-2.0
"""Pressure-driven demotion: the prefill's cold host tier moves into the
decode engine's host tier.

This is vllm-torchtpu's evict sweep (``KVCacheStore::SweepOnce`` in tpu-sync)
with two substitutions. HiCache's radix tree on each engine takes the place of
the connector's ``KVCacheStore`` and the store node, and the decode the room
binds takes the place of the global registry's placement targets. Everything
else keeps the sweep's shape:

- Trigger: free pages of the prefill's host pool under a low watermark start
  a pressure episode; the episode ends when they recover to a high watermark.
  A failed host allocation (``evict_host`` on the tree) nudges the sweep,
  otherwise it checks on a period.
- Victims: host-only, unreferenced leaves, oldest first, at most one batch of
  pages per step. The batch is pinned (host lock) for the life of the offer.
- Offer: ``DemoteOffer`` names the victims by token path and source host
  pages. The decode answers ``DemoteAck`` after deduplicating against its own
  tree and allocating landing pages, or refuses when its pool is full. It
  never evicts its own tier to make room.
- Move: the prefill pushes DRAM to DRAM with the host pool's own Raiden
  manager (``push_registered_plan``: the block transport's explicit-
  destination push, no D2H, no H2D), off the scheduler thread.
- Free after ack: when every rank's push returned, the prefill sends
  ``DemoteDone``, drops the victims from its tree and frees their host pages;
  the decode inserts host-only nodes, which makes its tree the directory the
  Plan C reverse pull reads.
- Local drops: no peer, a refusal, a timeout or a failed push leave the pages
  where they are; the tree's ordinary ``evict_host`` drops them later, the
  pre-sweep behaviour. Nothing is retried, nothing cascades.

Lockstep: the attention-TP ranks hold identical trees, so each decision that
changes a tree is taken from values reduced across ranks (prefill) or gathered
across ranks (decode), the same way ``peer_prefix`` does it. The listener
thread only enqueues; every tree operation runs on the scheduler thread.
"""

from __future__ import annotations

import logging
import threading
import time
from array import array
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang_tpu.disaggregation.raiden.common import _tcp_endpoint
from sglang_tpu.disaggregation.raiden.serde import (
    DemoteAck,
    DemoteDone,
    DemoteNode,
    DemoteOffer,
)

logger = logging.getLogger("sglang_tpu.disaggregation.raiden.peer_demote")

# Raiden ids of a demotion push. P/D transfers use uuid == room and the reverse
# pull uses 2^40 + room on the engines; the host pool managers are separate
# transports, but the range stays disjoint for the logs.
DEMOTE_UUID_BASE = 1 << 41

STAGE_OFFERED = 0  # DemoteOffer sent, waiting for the ack
STAGE_ACKED = 1  # landing pages received, push not started
STAGE_PUSHING = 2  # push running off the scheduler thread
STAGE_PUSHED = 3  # push returned ok
STAGE_NOTHING = 4  # the ack left nothing to push (all existing or refused)
STAGE_FAILED = 9


def page_floor(n: int, page_size: int) -> int:
    return (n // page_size) * page_size


@dataclass
class DemoteVictim:
    """One cold host leaf of the prefill tree, as offered."""

    node: Any
    node_id: Any
    token_ids: List[int]
    pages: List[int]
    host_value: torch.Tensor
    extra_key: Optional[str] = None
    cache_salt: Optional[str] = None
    lock_params: Any = None


@dataclass
class DemoteBatch:
    """Prefill-side state of one offer."""

    seq: int
    endpoint: str
    victims: List[DemoteVictim]
    sent_at: float
    stage: int = STAGE_OFFERED
    accepted: List[int] = field(default_factory=list)
    existing: List[int] = field(default_factory=list)
    landing: List[List[int]] = field(default_factory=list)
    address: str = ""
    error: str = ""
    push_thread: Optional[threading.Thread] = None
    push_done: bool = False
    push_error: str = ""
    pushed_pages: int = 0


@dataclass
class DemoteLanding:
    """Decode-side state of one accepted offer, until DemoteDone."""

    offer: DemoteOffer
    accepted: List[int]
    landing: List[torch.Tensor]
    received_at: float


@dataclass
class SweepContext:
    """What the prefill scheduler hands to ``peer_demote_tick``."""

    tree_cache: Any
    host_pool: Any  # the anchor host pool (TpuRaidenHostKVCache)
    page_size: int
    low_watermark: float = 0.03
    high_watermark: float = 0.05
    batch_pages: int = 128
    period_s: float = 30.0
    timeout_s: float = 120.0
    parallelism: int = 1
    # list[int] -> list[int] reduced over the attention-TP ranks; None alone.
    reduce_min: Any = None
    reduce_max: Any = None


@dataclass
class LandingContext:
    """What the decode scheduler hands to ``peer_demote_serve``."""

    tree_cache: Any
    host_pool: Any
    page_size: int
    lease_timeout_s: float = 180.0
    # all-gather over the attention-TP ranks: gather(obj) -> [obj per rank].
    gather: Any = None


def _full_component():
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    return ComponentType.FULL


def _tree_core(tree):
    core = getattr(tree, "tree_core", None)
    if core is None:
        raise RuntimeError(
            "peer demotion needs the unified radix cache (tree_core); the "
            "legacy HiRadixCache has no host-leaf API to demote from"
        )
    return core


def _path_tokens(node) -> List[int]:
    """Tokens from the root to the end of ``node``'s own segment."""
    segments = []
    cur = node
    while cur is not None and getattr(cur, "key", None) is not None:
        segments.append(cur.key.token_ids)
        cur = cur.parent
    out: List[int] = []
    for seg in reversed(segments):
        out.extend(int(t) for t in seg)
    return out


def _page_hashes(core, key: RadixKey, parent_len: int, page_size: int) -> List[str]:
    """Per-page hashes for ``key``: placeholders for the parent path the
    decode already holds, chained hashes for the inserted segment."""
    n_parent = parent_len // page_size
    leaf = [int(t) for t in key.token_ids[parent_len:]]
    try:
        from sglang.srt.mem_cache.utils import get_hash_str

        prior = None
        if parent_len > 0:
            _, last = _backed_prefix(core, key[:parent_len], page_size)
            if last is not None:
                prior = last.get_last_hash_value()
        hashes = get_hash_str(leaf, prior, page_size)
        if isinstance(hashes, str):
            hashes = [hashes]
        hashes = list(hashes)
    except Exception:  # noqa: BLE001 - hashes only feed storage and events
        hashes = [""] * (len(leaf) // page_size)
    return [""] * n_parent + hashes


def _backed_prefix(core, key: RadixKey, page_size: int):
    """Length of ``key`` covered by host-backed nodes from the root, and the
    last such node. The decode must hold a node's parent path on host before
    it can hang a host-only node below it."""
    full = _full_component()
    node = core.root_node
    length = 0
    last = None
    while len(key) > 0:
        child = node.children.get(key.child_key(page_size))
        if child is None:
            break
        if child.component_data[full].host_value is None:
            break
        n = child.key.match(key, page_size=page_size)
        if n <= 0:
            break
        length += n
        last = child
        if n < len(child.key):
            break
        key = key[n:]
        node = child
    return length, last


class PeerDemoteMixin:
    """Mixed into ``RaidenKVManager`` next to ``PeerPrefixMixin``. The prefill
    role sweeps and pushes; the decode role admits and inserts."""

    def _demote_init(self) -> None:
        self._demote_lock = threading.Lock()
        self._demote_seq = 0
        self._demote_batch: Optional[DemoteBatch] = None
        self._demote_draining: List[DemoteBatch] = []
        self._demote_episode = False
        self._demote_targets: List[str] = []
        self._demote_last_check = 0.0
        self._demote_nudged = False
        self._demote_acks: Dict[int, DemoteAck] = {}
        self._demote_offer_inbox: deque = deque()
        self._demote_done_inbox: deque = deque()
        self._demote_inbox_at: Dict[int, float] = {}
        self._demote_landing: Dict[int, DemoteLanding] = {}
        self._demote_stats: Dict[str, int] = {
            "episodes": 0,
            "offers": 0,
            "offered_pages": 0,
            "pushed_pages": 0,
            "existing_pages": 0,
            "evicted_pages": 0,
            "refused": 0,
            "failed": 0,
            "landed_pages": 0,
            "dropped_landings": 0,
        }
        self._message_handlers.update(
            {
                DemoteOffer.TAG: self._on_demote_offer,
                DemoteAck.TAG: self._on_demote_ack,
                DemoteDone.TAG: self._on_demote_done,
            }
        )

    # ------------------------------------------------------------------ #
    # Listener thread: enqueue only.
    # ------------------------------------------------------------------ #
    def _on_demote_offer(self, msg: List[bytes]) -> None:
        offer = DemoteOffer.from_frames(msg)
        with self._demote_lock:
            self._demote_offer_inbox.append(offer)
            self._demote_inbox_at[offer.seq] = time.monotonic()

    def _on_demote_ack(self, msg: List[bytes]) -> None:
        ack = DemoteAck.from_frames(msg)
        with self._demote_lock:
            self._demote_acks[ack.seq] = ack

    def _on_demote_done(self, msg: List[bytes]) -> None:
        done = DemoteDone.from_frames(msg)
        with self._demote_lock:
            self._demote_done_inbox.append(done)

    # ------------------------------------------------------------------ #
    # Prefill role: the sweep. Scheduler thread.
    # ------------------------------------------------------------------ #
    def peer_demote_nudge(self) -> None:
        """A host allocation just failed (the tree dropped pages): check the
        watermark at the next tick instead of waiting for the period."""
        self._demote_nudged = True

    def _reduce(self, fn, values: List[int]) -> List[int]:
        return list(values) if fn is None else list(fn(list(values)))

    def _agree_min(self, ctx, values):
        return self._reduce(ctx.reduce_min, values)

    def _agree_max(self, ctx, values):
        return self._reduce(ctx.reduce_max, values)

    def _free_ratio(self, ctx: SweepContext) -> float:
        total = int(getattr(ctx.host_pool, "size", 0) or 0)
        if total <= 0:
            return 1.0
        return float(ctx.host_pool.available_size()) / float(total)

    def peer_demote_tick(self, ctx: SweepContext) -> None:
        self._demote_drain()
        batch = self._demote_batch
        if batch is not None:
            self._demote_advance(ctx, batch)
            return
        now = time.monotonic()
        if not self._demote_episode:
            due = int(
                self._demote_nudged or now - self._demote_last_check >= ctx.period_s
            )
            if not self._agree_max(ctx, [due])[0]:
                return
            self._demote_nudged = False
            self._demote_last_check = now
            under = int(self._free_ratio(ctx) < ctx.low_watermark)
            if not self._agree_max(ctx, [under])[0]:
                return
            endpoint = getattr(self, "_peer_decode_endpoint", None)
            if endpoint is None:
                logger.info(
                    "peer demote: host pool under watermark, no decode peer yet"
                )
                return
            self._demote_episode = True
            self._demote_targets = [endpoint]
            self._demote_stats["episodes"] += 1
            logger.info(
                "peer demote episode start | free=%.3f low=%.3f high=%.3f",
                self._free_ratio(ctx),
                ctx.low_watermark,
                ctx.high_watermark,
            )
        else:
            recovered = int(self._free_ratio(ctx) >= ctx.high_watermark)
            if self._agree_min(ctx, [recovered])[0]:
                self._demote_end_episode(ctx, "recovered")
                return
            if not self._demote_targets:
                self._demote_end_episode(ctx, "no target left")
                return
        victims = self._demote_select(ctx)
        sig = [len(victims), sum(len(v.pages) for v in victims)]
        if self._agree_min(ctx, sig) != self._agree_max(ctx, sig):
            logger.warning("peer demote: ranks picked different batches; episode ends")
            self._demote_end_episode(ctx, "rank mismatch")
            return
        if not victims:
            self._demote_end_episode(ctx, "nothing evictable")
            return
        self._demote_offer(ctx, victims)

    def _demote_end_episode(self, ctx: SweepContext, reason: str) -> None:
        if self._demote_episode:
            logger.info(
                "peer demote episode end | reason=%s free=%.3f",
                reason,
                self._free_ratio(ctx),
            )
        self._demote_episode = False
        self._demote_targets = []

    def _demote_select(self, ctx: SweepContext) -> List[DemoteVictim]:
        """Cold host leaves, oldest first, up to one batch of pages. Mirrors
        ``GetEvictableKeys``: unpinned, LRU order, capped."""
        tree = ctx.tree_cache
        core = _tree_core(tree)
        if getattr(core, "is_eagle", False):
            return []
        full = _full_component()
        lru = core.host_lru_lists[full]
        victims: List[DemoteVictim] = []
        pages = 0
        node = lru.get_lru_no_host_lock()
        while node is not None and pages < ctx.batch_pages:
            if core._is_host_leaf(node):
                victim = self._demote_victim(ctx, node)
                if victim is not None:
                    if pages + len(victim.pages) > ctx.batch_pages:
                        break
                    victims.append(victim)
                    pages += len(victim.pages)
            node = lru.get_prev_no_host_lock(node)
        return victims

    def _demote_victim(self, ctx: SweepContext, node) -> Optional[DemoteVictim]:
        full = _full_component()
        host_value = node.component_data[full].host_value
        if host_value is None or node.key is None:
            return None
        if len(host_value) != len(node.key) or len(host_value) % ctx.page_size:
            return None
        if getattr(node, "write_through_pending_id", None) is not None:
            return None
        if getattr(node, "load_back_pending_id", None) is not None:
            return None
        try:
            pages = [int(p) for p in ctx.host_pool._to_page_ids(host_value)]
        except Exception:  # noqa: BLE001
            return None
        if not pages:
            return None
        return DemoteVictim(
            node=node,
            node_id=node.id,
            token_ids=_path_tokens(node),
            pages=pages,
            host_value=host_value,
            extra_key=getattr(node.key, "extra_key", None),
            cache_salt=getattr(node.key, "cache_salt", None),
        )

    def _demote_offer(self, ctx: SweepContext, victims: List[DemoteVictim]) -> None:
        tree = ctx.tree_cache
        endpoint = self._demote_targets[0]
        for v in victims:
            v.lock_params = tree.inc_host_lock_ref(v.node_id).to_dec_params()
        with self._demote_lock:
            self._demote_seq += 1
            seq = self._demote_seq
        payload = DemoteOffer(
            seq=seq,
            prefill_ip=self.local_ip,
            prefill_port=self.rank_port,
            nodes=[
                DemoteNode(
                    token_ids=v.token_ids,
                    pages=v.pages,
                    extra_key=v.extra_key,
                    cache_salt=v.cache_salt,
                )
                for v in victims
            ],
            deadline_ms=int(ctx.timeout_s * 1000),
        ).to_frames()
        batch = DemoteBatch(
            seq=seq, endpoint=endpoint, victims=victims, sent_at=time.monotonic()
        )
        self._demote_batch = batch
        self._demote_stats["offers"] += 1
        self._demote_stats["offered_pages"] += sum(len(v.pages) for v in victims)
        try:
            self._send_multipart_locked(endpoint, payload)
        except Exception as exc:  # noqa: BLE001
            batch.error = f"offer send: {exc}"
            batch.stage = STAGE_FAILED
        logger.info(
            "peer demote offered | seq=%d nodes=%d pages=%d free=%.3f",
            seq,
            len(victims),
            sum(len(v.pages) for v in victims),
            self._free_ratio(ctx),
        )

    def _demote_advance(self, ctx: SweepContext, batch: DemoteBatch) -> None:
        now = time.monotonic()
        with self._demote_lock:
            ack = self._demote_acks.pop(batch.seq, None)
        if batch.stage == STAGE_OFFERED:
            if ack is not None:
                if ack.error:
                    batch.error = ack.error
                    batch.stage = STAGE_FAILED
                else:
                    batch.accepted = [int(i) for i in ack.accepted]
                    batch.existing = [int(i) for i in ack.existing]
                    batch.landing = [[int(p) for p in pages] for pages in ack.landing]
                    batch.address = ack.address
                    bad = any(
                        i < 0 or i >= len(batch.victims)
                        for i in batch.accepted + batch.existing
                    )
                    if bad:
                        batch.error = "ack names an unknown node"
                        batch.stage = STAGE_FAILED
                    else:
                        batch.stage = STAGE_ACKED if batch.accepted else STAGE_NOTHING
            elif now - batch.sent_at > ctx.timeout_s:
                batch.error = "timed out waiting for the decode"
                batch.stage = STAGE_FAILED
        elif batch.stage == STAGE_PUSHING:
            if batch.push_done:
                if batch.push_error:
                    batch.error = batch.push_error
                    batch.stage = STAGE_FAILED
                else:
                    batch.stage = STAGE_PUSHED
            elif now - batch.sent_at > 2 * ctx.timeout_s:
                batch.error = "push did not return in time"
                batch.stage = STAGE_FAILED
        stages = [batch.stage]
        mn = self._agree_min(ctx, stages)[0]
        mx = self._agree_max(ctx, stages)[0]
        if mx == STAGE_FAILED:
            self._demote_abort(ctx, batch)
            return
        if mn == STAGE_ACKED and mx == STAGE_ACKED:
            self._demote_start_push(ctx, batch)
            return
        if mn == STAGE_ACKED and mx == STAGE_NOTHING:
            batch.error = batch.error or "ranks received different acks"
            self._demote_abort(ctx, batch)
            return
        if mn in (STAGE_PUSHED, STAGE_NOTHING) and mx in (STAGE_PUSHED, STAGE_NOTHING):
            check = [
                len(batch.accepted),
                sum(batch.accepted),
                len(batch.existing),
                sum(batch.existing),
            ]
            if self._agree_min(ctx, check) != self._agree_max(ctx, check):
                batch.error = "ranks accepted different nodes"
                self._demote_abort(ctx, batch)
                return
            self._demote_finish(ctx, batch)

    def _demote_start_push(self, ctx: SweepContext, batch: DemoteBatch) -> None:
        src: List[int] = []
        dst: List[int] = []
        for i, pages in zip(batch.accepted, batch.landing):
            src.extend(batch.victims[i].pages)
            dst.extend(pages)
        if len(src) != len(dst) or not src or not batch.address:
            batch.error = "landing pages do not match the offered pages"
            batch.stage = STAGE_FAILED
            return
        uuid = DEMOTE_UUID_BASE + batch.seq
        manager = ctx.host_pool.transfer._mgr
        parallelism = ctx.parallelism

        def run() -> None:
            try:
                manager.push_registered_plan(
                    uuid, batch.address, src, dst, -1, parallelism
                )
                batch.pushed_pages = len(src)
            except Exception as exc:  # noqa: BLE001
                batch.push_error = f"push: {exc}"
            finally:
                batch.push_done = True

        batch.stage = STAGE_PUSHING
        batch.push_thread = threading.Thread(
            target=run, name=f"peer-demote-push-{batch.seq}", daemon=True
        )
        batch.push_thread.start()

    def _demote_finish(self, ctx: SweepContext, batch: DemoteBatch) -> None:
        if batch.accepted:
            try:
                self._send_multipart_locked(
                    batch.endpoint,
                    DemoteDone(
                        seq=batch.seq, ok=True, accepted=batch.accepted
                    ).to_frames(),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("peer demote done for seq %d failed: %s", batch.seq, exc)
        done = set(batch.accepted) | set(batch.existing)
        evicted = self._demote_evict(ctx, batch, done)
        self._demote_unpin(ctx, batch)
        refused = len(batch.victims) - len(done)
        self._demote_stats["pushed_pages"] += batch.pushed_pages
        self._demote_stats["existing_pages"] += sum(
            len(batch.victims[i].pages) for i in batch.existing
        )
        self._demote_stats["evicted_pages"] += evicted
        self._demote_stats["refused"] += refused
        logger.info(
            "peer demote done | seq=%d pushed=%d existing=%d refused=%d evicted_pages=%d "
            "took=%.2fs free=%.3f",
            batch.seq,
            len(batch.accepted),
            len(batch.existing),
            refused,
            evicted,
            time.monotonic() - batch.sent_at,
            self._free_ratio(ctx),
        )
        self._demote_batch = None
        if not done:
            # The target took nothing: it is out for the rest of the episode,
            # as a store node that answers RESOURCE_EXHAUSTED is.
            self._demote_targets.pop(0)

    def _demote_evict(self, ctx: SweepContext, batch: DemoteBatch, idx: set) -> int:
        """Drop the demoted leaves from the prefill tree and free their host
        pages. A node that stopped being a cold leaf meanwhile, or whose host
        pages moved, is left alone: the decode holds a copy, the prefill
        keeps its own, as a store keeps a block a reader pinned."""
        if not idx:
            return 0
        tree = ctx.tree_cache
        core = _tree_core(tree)
        full = _full_component()
        from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
            DriveHostEvictionResult,
        )

        result = DriveHostEvictionResult()
        evicted = 0
        for i in sorted(idx):
            v = batch.victims[i]
            self._demote_unpin_one(tree, v)
            node = v.node
            current = node.component_data[full].host_value
            same = current is v.host_value or (
                current is not None
                and len(current) == len(v.host_value)
                and bool(torch.equal(current, v.host_value))
            )
            if not same or not core._is_host_leaf(node):
                continue
            core._evict_host_leaf(
                node, result.tracker, result.device_frees, result.host_frees
            )
            evicted += len(v.pages)
        tree._free_values(result.device_frees, result.host_frees)
        return evicted

    def _demote_unpin_one(self, tree, v: DemoteVictim) -> None:
        if v.lock_params is None:
            return
        params, v.lock_params = v.lock_params, None
        try:
            tree.dec_host_lock_ref(v.node_id, params)
        except Exception:  # noqa: BLE001
            logger.exception("peer demote: unpin failed for node %s", v.node_id)

    def _demote_unpin(self, ctx: SweepContext, batch: DemoteBatch) -> None:
        for v in batch.victims:
            self._demote_unpin_one(ctx.tree_cache, v)

    def _demote_abort(self, ctx: SweepContext, batch: DemoteBatch) -> None:
        self._demote_stats["failed"] += 1
        logger.warning(
            "peer demote abandoned | seq=%d stage=%d reason=%s",
            batch.seq,
            batch.stage,
            batch.error or "a rank failed",
        )
        if batch.accepted and batch.stage in (
            STAGE_ACKED,
            STAGE_PUSHING,
            STAGE_PUSHED,
            STAGE_FAILED,
        ):
            try:
                self._send_multipart_locked(
                    batch.endpoint,
                    DemoteDone(
                        seq=batch.seq, ok=False, accepted=batch.accepted
                    ).to_frames(),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "peer demote abort for seq %d failed: %s", batch.seq, exc
                )
        self._demote_batch = None
        if batch.push_thread is not None and batch.push_thread.is_alive():
            # The push still reads the source pages: keep them pinned until
            # it returns, then unpin from the drain.
            self._demote_draining.append(batch)
        else:
            self._demote_unpin(ctx, batch)
        self._demote_batch_ctx = ctx
        self._demote_end_episode(ctx, "offer failed")

    def _demote_drain(self) -> None:
        if not self._demote_draining:
            return
        keep = []
        for batch in self._demote_draining:
            if batch.push_thread is not None and batch.push_thread.is_alive():
                keep.append(batch)
            else:
                ctx = getattr(self, "_demote_batch_ctx", None)
                if ctx is not None:
                    self._demote_unpin(ctx, batch)
        self._demote_draining = keep

    # ------------------------------------------------------------------ #
    # Decode role: admit, land, insert. Scheduler thread.
    # ------------------------------------------------------------------ #
    def peer_demote_serve(self, ctx: LandingContext) -> None:
        with self._demote_lock:
            offers = list(self._demote_offer_inbox)
            self._demote_offer_inbox.clear()
            dones = list(self._demote_done_inbox)
            self._demote_done_inbox.clear()
        if ctx.gather is not None:
            offers, dones = self._demote_agree(ctx, offers, dones)
        for done in dones:
            try:
                self._demote_land(ctx, done)
            except Exception:  # noqa: BLE001
                logger.exception("peer demote: landing failed for seq %d", done.seq)
                self._demote_free_landing(ctx, done.seq)
        for offer in offers:
            try:
                self._demote_admit(ctx, offer)
            except Exception as exc:  # noqa: BLE001
                logger.exception("peer demote: admit failed for seq %d", offer.seq)
                self._demote_reply(
                    offer,
                    DemoteAck(
                        seq=offer.seq, accepted=[], landing=[], error=f"admit: {exc}"
                    ),
                )
        self._demote_expire(ctx)

    def _demote_agree(self, ctx: LandingContext, offers, dones):
        """Keep only the offers and dones every rank has received, so the
        ranks' trees change in the same tick; the rest wait in the inbox."""
        local = (
            tuple(sorted(o.seq for o in offers)),
            tuple(sorted(d.seq for d in dones)),
        )
        views = ctx.gather(local)
        seqs = set(local[0]).intersection(*(set(v[0]) for v in views))
        done_seqs = set(local[1]).intersection(*(set(v[1]) for v in views))
        now = time.monotonic()
        take, keep = [], []
        for offer in offers:
            if offer.seq in seqs:
                take.append(offer)
            elif now - self._demote_inbox_at.get(offer.seq, now) > ctx.lease_timeout_s:
                self._demote_reply(
                    offer,
                    DemoteAck(
                        seq=offer.seq, accepted=[], landing=[], error="inbox timeout"
                    ),
                )
            else:
                keep.append(offer)
        take_done = [d for d in dones if d.seq in done_seqs]
        keep_done = [d for d in dones if d.seq not in done_seqs]
        with self._demote_lock:
            self._demote_offer_inbox.extendleft(reversed(keep))
            self._demote_done_inbox.extendleft(reversed(keep_done))
        for offer in take:
            self._demote_inbox_at.pop(offer.seq, None)
        take.sort(key=lambda o: o.seq)
        take_done.sort(key=lambda d: d.seq)
        return take, take_done

    def _demote_key(self, node: DemoteNode) -> RadixKey:
        return RadixKey(
            array("q", node.token_ids),
            node.extra_key,
            is_bigram=False,
            cache_salt=node.cache_salt,
        )

    def _demote_decide(self, ctx: LandingContext, node: DemoteNode) -> int:
        """0 refuse, 1 accept, 2 already held."""
        tree = ctx.tree_cache
        core = _tree_core(tree)
        ps = ctx.page_size
        total = len(node.token_ids)
        leaf = len(node.pages) * ps
        parent_len = total - leaf
        if leaf <= 0 or parent_len < 0 or total % ps or parent_len % ps:
            return 0
        key = self._demote_key(node)
        backed, _ = _backed_prefix(core, key, ps)
        device = 0
        try:
            res = tree.match_prefix(MatchPrefixParams(key=key))
            device = page_floor(len(res.device_indices), ps)
        except Exception:  # noqa: BLE001
            device = 0
        if backed >= total or device >= total:
            return 2
        if backed < parent_len:
            return 0
        return 1

    def _demote_admit(self, ctx: LandingContext, offer: DemoteOffer) -> None:
        ps = ctx.page_size
        decisions = [self._demote_decide(ctx, n) for n in offer.nodes]
        needed = sum(
            len(n.pages) * ps for n, d in zip(offer.nodes, decisions) if d == 1
        )
        # No self-eviction to make room, as the store node has none.
        if needed > 0 and needed > ctx.host_pool.available_size():
            decisions = [0 if d == 1 else d for d in decisions]
        if ctx.gather is not None:
            views = ctx.gather(tuple(decisions))
            decisions = [
                d if all(v[i] == d for v in views) else 0
                for i, d in enumerate(decisions)
            ]
        accepted = [i for i, d in enumerate(decisions) if d == 1]
        existing = [i for i, d in enumerate(decisions) if d == 2]
        landing: List[torch.Tensor] = []
        for i in accepted:
            idx = ctx.host_pool.alloc(len(offer.nodes[i].pages) * ps)
            if idx is None:
                for t in landing:
                    ctx.host_pool.free(t)
                self._demote_reply(
                    offer,
                    DemoteAck(
                        seq=offer.seq, accepted=[], landing=[], error="host pool full"
                    ),
                )
                self._demote_stats["refused"] += 1
                return
            landing.append(idx)
        address = ""
        if accepted:
            address = str(ctx.host_pool.transfer._mgr.transfer_address)
            self._demote_landing[offer.seq] = DemoteLanding(
                offer=offer,
                accepted=accepted,
                landing=landing,
                received_at=time.monotonic(),
            )
        self._demote_reply(
            offer,
            DemoteAck(
                seq=offer.seq,
                accepted=accepted,
                landing=[
                    [int(p) for p in ctx.host_pool._to_page_ids(t)] for t in landing
                ],
                address=address,
                existing=existing,
            ),
        )
        logger.info(
            "peer demote admitted | seq=%d accepted=%d existing=%d refused=%d",
            offer.seq,
            len(accepted),
            len(existing),
            len(offer.nodes) - len(accepted) - len(existing),
        )

    def _demote_reply(self, offer: DemoteOffer, ack: DemoteAck) -> None:
        try:
            self._send_multipart_locked(
                _tcp_endpoint(ip=offer.prefill_ip, port=offer.prefill_port),
                ack.to_frames(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("peer demote ack for seq %d failed: %s", offer.seq, exc)

    def _demote_free_landing(self, ctx: LandingContext, seq: int) -> None:
        landing = self._demote_landing.pop(seq, None)
        if landing is None:
            return
        for t in landing.landing:
            try:
                ctx.host_pool.free(t)
            except Exception:  # noqa: BLE001
                logger.exception("peer demote: freeing landing pages failed")

    def _demote_land(self, ctx: LandingContext, done: DemoteDone) -> None:
        landing = self._demote_landing.pop(done.seq, None)
        if landing is None:
            return
        if not done.ok:
            for t in landing.landing:
                ctx.host_pool.free(t)
            self._demote_stats["dropped_landings"] += len(landing.landing)
            return
        tree = ctx.tree_cache
        core = _tree_core(tree)
        ps = ctx.page_size
        done_set = set(int(i) for i in done.accepted)
        for node_idx, idx_tensor in zip(landing.accepted, landing.landing):
            if node_idx not in done_set:
                ctx.host_pool.free(idx_tensor)
                continue
            node = landing.offer.nodes[node_idx]
            key = self._demote_key(node)
            total = len(node.token_ids)
            parent_len = total - len(node.pages) * ps
            backed, _ = _backed_prefix(core, key, ps)
            if backed >= total or backed < parent_len:
                # Held meanwhile, or the parent path left the host tier while
                # the push ran: the bytes have nowhere to hang.
                ctx.host_pool.free(idx_tensor)
                self._demote_stats["dropped_landings"] += 1
                continue
            placeholder = torch.full((parent_len,), -1, dtype=idx_tensor.dtype)
            host_value = torch.cat([placeholder, idx_tensor])
            hashes = _page_hashes(core, key, parent_len, ps)
            result = core.insert_host(core.root_node.id, key, host_value, hashes)
            apply = getattr(tree, "_apply_cache_actions", None)
            actions = getattr(result, "cache_actions", None)
            if apply is not None and actions:
                apply(actions)
            if getattr(result, "host_insert_dropped", False) or (
                getattr(result, "inserted_host_node", None) is None
            ):
                ctx.host_pool.free(idx_tensor)
                self._demote_stats["dropped_landings"] += 1
                continue
            self._demote_stats["landed_pages"] += len(node.pages)
        logger.info(
            "peer demote landed | seq=%d nodes=%d landed_pages=%d",
            done.seq,
            len(landing.accepted),
            self._demote_stats["landed_pages"],
        )

    def _demote_expire(self, ctx: LandingContext) -> None:
        now = time.monotonic()
        for seq, landing in list(self._demote_landing.items()):
            if now - landing.received_at > ctx.lease_timeout_s:
                logger.info("peer demote landing expired | seq=%d", seq)
                self._demote_free_landing(ctx, seq)

    # ------------------------------------------------------------------ #
    # Introspection, for tests and metrics.
    # ------------------------------------------------------------------ #
    def peer_demote_stats(self) -> dict:
        out = dict(self._demote_stats)
        out["episode"] = self._demote_episode
        out["inflight"] = self._demote_batch is not None
        out["landings"] = len(self._demote_landing)
        return out
