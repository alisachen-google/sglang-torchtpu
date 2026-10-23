"""Victim push: the decode's RAM is the cache tier below the prefill's RAM.

Why: core has no per-request disaggregation cache policy or queue extension.
How: backend puller subclass, plus small lifecycle wrappers listed below.
Upstream PR or issue: none yet; queue/cache lifecycle seams need upstreaming.
Remove when: core supplies equivalent checkpoint and decode-queue callbacks.

SGLANG_DISAGG_PREFIX_VICTIM_PUSH=1 on both roles, with the decode-held prefix
pull (SGLANG_DISAGG_PREFILL_PULL_DECODE_PREFIX) on.

- Decode: does not cache the requests it serves; its radix tree holds only the
  sessions prefills push to it.
- Prefill: caches every session as usual and records, at each request's
  bootstrap commit, which decode serves the session. When its host pools run
  low it takes the tracked session with the oldest path end, locks it, loads
  it back to HBM if it is host-only (raiden reads device memory), registers
  the pages for a read and sends them to the ranks of that decode only
  (VICTIM_PUSH). On the decode's ack it unlocks the session and, if stored,
  makes it its own next eviction victim.
- Decode: every ``_CADENCE`` scheduler iterations its ranks agree on the
  pushes all of them received, allocate pages, read them, insert the
  sessions and ack (VICTIM_ACK).

Sticky routing sends the session's next turn to the same decode, whose prefix
offer lets the prefill pull it back. Routes attach to exact cached branches,
including their salt/adapter namespace. TP/CP ranks agree on both identity and
content before changing the tree. Each prefill incarnation owns a distinct
native read namespace. Sessions longer than a staging slot are not pushed.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import secrets
import time
import weakref
from array import array
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import msgspec
import torch
import torch.distributed as dist
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeTransferQueue
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.common import kv_to_page_indices
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager
from sglang_tpu.disaggregation.raiden.prefix_offer import window_tokens
from sglang_tpu.disaggregation.raiden.prefix_pull import DecodePrefixPuller
from sglang_tpu.envs import envs as tpu_envs

logger = logging.getLogger(__name__)

# Arbitrary: bounds the HBM a prefill pins and a decode reads into at once.
_MAX_INFLIGHT = 4
# Arbitrary: push once the host pools could take only about two more sessions.
_PUSH_FULL_FREE_TOKENS = 65536
_PUSH_SWA_FREE_WINDOWS = 4
# Arbitrary, well past a raiden read of one staging slot.
_ACK_TIMEOUT_S = 60.0
# Arbitrary: bootstrap steps a push waits for a load slot and HBM room.
_MAX_LOAD_TRIES = 2000
# Arbitrary: decode iterations between agreement rounds, and pushes a rank
# offers per round; a push not every rank received in 8 rounds is dropped.
_CADENCE = 16
_AGREE_SLOTS = 8
_MAX_UNAGREED_ROUNDS = 8
# Same reserve the prefix pull keeps for admission.
_MIN_FREE_FRACTION = 0.25
# Arbitrary: often enough to follow a sweep, rare enough to stay quiet.
_LOG_INTERVAL_S = 30.0

_is_installed = False


def _all_ranks_reduce(
    groups: Sequence[Any], rows: List[List[int]], op
) -> List[List[int]]:
    if not groups or not rows:
        return [list(row) for row in rows]
    values = torch.tensor(rows, dtype=torch.int64)
    for group in groups:
        dist.all_reduce(values, op=op, group=group)
    return values.tolist()


def _all_ranks_common(groups: Sequence[Any], keys: list) -> list:
    """Agree on bounded manifests, including content, over the full TP/CP mesh.

    Feed each group's intersection into the next group. Intersecting separate
    local neighbour lists does not establish agreement over a Cartesian mesh.
    """
    common = set(sorted(keys)[:_AGREE_SLOTS])
    for group in groups:
        gathered = [None] * dist.get_world_size(group)
        dist.all_gather_object(gathered, sorted(common), group=group)
        common.intersection_update(*(set(row) for row in gathered))
    return sorted(common)


def _manifest(tokens, extra_key=None, cache_salt=None, is_eagle=False, *geometry):
    payload = msgspec.msgpack.encode(
        (tokens, extra_key, cache_salt, is_eagle, geometry)
    )
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    return tuple(int.from_bytes(digest[i : i + 8], "little") >> 1 for i in (0, 8))


def _agree(groups, row):
    low = _all_ranks_reduce(groups, [row], dist.ReduceOp.MIN)
    high = _all_ranks_reduce(groups, [row], dist.ReduceOp.MAX)
    return low == high


def _cpu_groups(scheduler: Any) -> List[Any]:
    return [
        group
        for group in (scheduler.attn_tp_cpu_group, scheduler.attn_cp_cpu_group)
        if group is not None and dist.get_world_size(group) > 1
    ]


def _depth(node: Any, root: Any) -> int:
    depth = 0
    while node is not root:
        depth += len(node.key)
        node = node.parent
    return depth


def _path_tokens(root: Any, leaf: Any) -> List[int]:
    """Reconstruct raw tokens, joining overlapping EAGLE boundaries once."""
    parts = []
    node = leaf
    while node is not root:
        parts.append(node.key.token_ids)
        node = node.parent
    tokens = []
    for part in reversed(parts):
        tokens.extend(part[1:] if tokens and leaf.key.is_bigram else part)
    return tokens


def _maybe_log(stats: Dict[str, int], role: str, last: List[float]) -> None:
    now = time.monotonic()
    if now - last[0] < _LOG_INTERVAL_S:
        return
    last[0] = now
    logger.info(
        "[victim-push] role=%s %s",
        role,
        " ".join(f"{k}={v}" for k, v in stats.items() if v),
    )


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------


class _Push(msgspec.Struct):
    seq: int
    key: int
    decode: Tuple[str, int]
    tokens: List[int]
    path_len: int
    anchor_id: Any
    anchor_receipt: Any
    anchor_host_receipt: Any
    source_node: Any = None
    extra_key: Optional[str] = None
    cache_salt: Optional[str] = None
    is_eagle: bool = False
    state: str = "load"
    ready_receipt: Any = None
    load_tries: int = 0
    sent_at: float = 0.0
    stored: Optional[bool] = None
    producer_ready: bool = False

    @property
    def num_tokens(self) -> int:
        return len(self.tokens) - int(self.is_eagle)

    def radix_key(self):
        return RadixKey(
            array("q", self.tokens),
            extra_key=self.extra_key,
            cache_salt=self.cache_salt,
            is_bigram=self.is_eagle,
        )


class PrefillPusher:
    """Pushes sessions near eviction to their decodes, for one prefill puller."""

    def __init__(self, puller: DecodePrefixPuller) -> None:
        self._mgr = puller._mgr
        self._cache = puller._tree_cache
        self._allocator = puller._allocator
        self._page_size = puller._page_size
        self._groups = list(puller._groups)
        # Incarnation ID shared by this prefill's TP/CP ranks; independent
        # prefills and restarts cannot alias native read IDs on one decode.
        epoch = _all_ranks_reduce(
            self._groups, [[secrets.randbits(62)]], dist.ReduceOp.MIN
        )[0][0]
        self._mgr.victim_namespace = f"{epoch:x}"

        window_pages = puller._window_pages
        self._window_tokens = (
            0 if window_pages is None else window_pages * self._page_size
        )
        self._max_tokens = self._mgr._full_engine_max_blocks * self._page_size
        # Exact cached leaf -> (decode peer, semantic fingerprint). Weak keys
        # disappear with eviction and retain independent shared-prefix branches.
        self._routes = weakref.WeakKeyDictionary()
        _pushers_by_cache[self._cache] = weakref.ref(self)
        self._pushes: Dict[int, _Push] = {}
        # Path length last pushed, or found unpushable, per session.
        self._pushed_len = weakref.WeakKeyDictionary()
        self._next_seq = 1
        self.stats = dict.fromkeys(
            ("started", "loaded_back", "sent", "stored", "declined", "aborted"), 0
        )
        self._last_log = [0.0]

    def record(self, req: Any) -> None:
        """Carry the actual decode peer to the request's cache checkpoint.

        Ownership belongs to a cached branch, including its namespace, rather
        than to a first-page hash shared by unrelated chat sessions.
        """
        peer = self._mgr.decode_peer(req.bootstrap_room)
        if peer is not None:
            req._pd_victim_decode_peer = peer
            self.remember(req)

    def remember(self, req: Any) -> None:
        peer = getattr(req, "_pd_victim_decode_peer", None)
        if peer is None:
            return
        tree_core = self._cache.tree_core
        node_id = getattr(req, "last_node", None)
        if node_id is None:
            key = RadixKey(
                array("q", req.origin_input_ids),
                extra_key=getattr(req, "extra_key", None),
                cache_salt=getattr(req, "cache_salt", None),
                is_bigram=tree_core.is_eagle,
            ).page_aligned(self._page_size)
            match = self._cache.match_prefix(MatchPrefixParams(key=key))
            node_id = match.best_match_node
        node = tree_core.node_by_id(node_id)
        if node is not tree_core.root_node and not node.children:
            tokens = _path_tokens(tree_core.root_node, node)
            self._routes[node] = (
                peer,
                _manifest(
                    tokens, node.key.extra_key, node.key.cache_salt, tree_core.is_eagle
                ),
            )

    def step(self) -> None:
        """Every bootstrap step, every rank; no tree decision uses a local clock."""
        if self._pushes:
            self._mgr._drain_engine()
        self._settle_sent()
        for push in list(self._pushes.values()):
            self._advance(push)
        eligible = int(
            len(self._pushes) < _MAX_INFLIGHT and self._under_host_pressure()
        )
        if _all_ranks_reduce(self._groups, [[eligible]], dist.ReduceOp.MIN)[0][0]:
            self._maybe_start()
        _maybe_log(self.stats, "prefill", self._last_log)

    def release_all(self) -> None:
        safe = all(
            p.state != "loading" and self._mgr.victim_send_settled(p.seq)
            for p in self._pushes.values()
        )
        safe = _all_ranks_reduce(self._groups, [[int(safe)]], dist.ReduceOp.MIN)[0][0]
        if not safe:
            raise RuntimeError(
                "Cannot release KV memory while victim transfers are active; retry after they drain"
            )
        for push in list(self._pushes.values()):
            self._finish(push, stored=False)

    def _settle_sent(self) -> None:
        sent = [p for p in self._pushes.values() if p.state == "sent"]
        if not sent:
            return
        now = time.monotonic()
        rows = []
        for push in sent:
            if push.stored is None:
                push.stored = self._mgr.take_victim_ack(push.seq)
            expired = now - push.sent_at > _ACK_TIMEOUT_S
            # An ACK timeout is not a native cancellation. Even on decline, the
            # source lease remains until every registered read has settled.
            settled = (
                push.stored is not None or expired
            ) and self._mgr.victim_send_settled(push.seq)
            rows.append([int(settled), int(push.stored is True)])
        agreed = _all_ranks_reduce(self._groups, rows, dist.ReduceOp.MIN)
        for push, (settled, stored) in zip(sent, agreed):
            if settled:
                self.stats["stored" if stored else "declined"] += 1
                self._finish(push, stored=bool(stored))

    def _finish(self, push: _Push, *, stored: bool) -> None:
        self._pushes.pop(push.seq, None)
        self._pushed_len[push.source_node] = push.path_len
        if push.ready_receipt is not None:
            self._cache.dec_lock_ref(push.anchor_id, push.ready_receipt)
        self._cache.dec_lock_ref(push.anchor_id, push.anchor_receipt)
        self._cache.dec_host_lock_ref(push.anchor_id, push.anchor_host_receipt)
        self._mgr.forget_victim_reads(push.seq)
        if stored:
            self._make_next_victim(push.anchor_id)

    def _make_next_victim(self, anchor_id: Any) -> None:
        """The decode holds the session now: make its path here the oldest in
        every eviction order, below any branch other sessions share."""
        tree_core = self._cache.tree_core
        anchor = tree_core.node_by_id(anchor_id)
        nodes = []
        stack = [anchor]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children.values())
        node = anchor.parent
        while node is not tree_core.root_node and len(node.children) == 1:
            nodes.append(node)
            node = node.parent
        lrus = [*tree_core.lru_lists.values(), *tree_core.host_lru_lists.values()]
        for node in nodes:
            node.last_access_time = 0
            for lru in lrus:
                if lru.in_list(node):
                    lru.reset_node_lru(node)

    def _under_host_pressure(self) -> bool:
        controller = self._cache.cache_controller
        if controller is None:
            return False
        if controller.mem_pool_host.available_size() < _PUSH_FULL_FREE_TOKENS:
            return True
        if not self._window_tokens:
            return False
        swa_pool = self._cache.host_pool_group.get_pool(PoolName.SWA)
        return (
            swa_pool is not None
            and swa_pool.available_size() < _PUSH_SWA_FREE_WINDOWS * self._window_tokens
        )

    def _maybe_start(self) -> None:
        picked = self._pick()
        if picked is None:
            signature = [-1, -1]
        else:
            _, decode, leaf = picked
            tree_core = self._cache.tree_core
            path = _path_tokens(tree_core.root_node, leaf)
            signature = list(
                _manifest(
                    path, leaf.key.extra_key, leaf.key.cache_salt, tree_core.is_eagle
                )
            )
        if not _agree(self._groups, signature) or picked is None:
            return
        key, decode, leaf = picked
        radix_key = RadixKey(
            array("q", path),
            extra_key=leaf.key.extra_key,
            cache_salt=leaf.key.cache_salt,
            is_bigram=tree_core.is_eagle,
        )
        match = self._cache.match_prefix(MatchPrefixParams(key=radix_key))
        anchor_id = match.best_match_node
        anchor = tree_core.node_by_id(anchor_id)
        num_tokens = _depth(anchor, tree_core.root_node)
        if not _agree(self._groups, [num_tokens]):
            return
        if (
            not max(self._page_size, self._window_tokens)
            <= num_tokens
            <= self._max_tokens
        ):
            self._pushed_len[leaf] = _depth(leaf, tree_core.root_node)
            return
        seq = self._next_seq
        self._next_seq += 1
        self._pushes[seq] = _Push(
            seq=seq,
            key=key,
            decode=decode,
            tokens=path[: num_tokens + int(tree_core.is_eagle)],
            path_len=_depth(leaf, tree_core.root_node),
            source_node=leaf,
            extra_key=leaf.key.extra_key,
            cache_salt=leaf.key.cache_salt,
            is_eagle=tree_core.is_eagle,
            anchor_id=anchor_id,
            anchor_host_receipt=self._cache.inc_host_lock_ref(
                anchor_id
            ).to_dec_params(),
            anchor_receipt=self._cache.inc_lock_ref(anchor_id).to_dec_params(),
        )
        self.stats["started"] += 1

    def _pick(self):
        """Oldest complete branch owned by a known decode, without grouping siblings."""
        tree_core = self._cache.tree_core
        busy = {push.source_node for push in self._pushes.values()}
        candidates = []
        for node, (peer, fingerprint) in list(self._routes.items()):
            try:
                if tree_core.node_by_id(node.id) is not node:
                    continue
            except KeyError:
                self._routes.pop(node, None)
                continue
            if node.children or node in busy:
                continue
            full = node.component_data[ComponentType.FULL]
            if full.value is None and full.host_value is None:
                continue
            depth = _depth(node, tree_core.root_node)
            if self._pushed_len.get(node) == depth:
                continue
            candidates.append((node.last_access_time, fingerprint, peer, node))
        if not candidates:
            return None
        _, fingerprint, peer, node = min(candidates, key=lambda x: x[:2])
        return fingerprint[0], peer, node

    def _advance(self, push: _Push) -> None:
        if push.state == "load":
            self._load(push)
        elif push.state == "loading":
            loaded = push.anchor_id not in self._cache.ongoing_load_back
            if _all_ranks_reduce(self._groups, [[int(loaded)]], dist.ReduceOp.MIN)[0][0]:
                self._ready(push)
        elif push.state == "ready":
            self._send(push)

    def _ready(self, push: _Push) -> None:
        push.state = "ready"
        push.ready_receipt = self._cache.inc_lock_ref(push.anchor_id).to_dec_params()
        # Native DMA reads do not share compute's ordering. A newly cached
        # request can still be written by an overlapping forward. Wait without
        # synchronizing the scheduler, while retaining the source locks.
        fence = getattr(self._allocator, "kv_free_fence", None)
        if fence is None:
            push.producer_ready = True
        else:
            fence.after_in_flight(lambda: setattr(push, "producer_ready", True))

    def _load(self, push: _Push) -> None:
        match = self._cache.match_prefix(MatchPrefixParams(key=push.radix_key()))
        ready = (
            len(match.device_indices) == push.num_tokens
            and not match.swa_host_hit_length
        )
        if _all_ranks_reduce(self._groups, [[int(ready)]], dist.ReduceOp.MIN)[0][0]:
            self._ready(push)
            return
        eligible = self._cache.has_free_load_back_slot() and self._has_device_room(
            push.num_tokens
        )
        if _all_ranks_reduce(self._groups, [[int(eligible)]], dist.ReduceOp.MIN)[0][0]:
            loaded = self._cache.load_back(push.anchor_id)
            if not _agree(self._groups, [int(loaded)]):
                raise RuntimeError("Victim load-back allocation diverged across ranks")
            if loaded:
                self._cache.ready_to_load_host_cache()
                push.state = "loading"
                self.stats["loaded_back"] += 1
            else:
                self._abort(push)
            return
        push.load_tries += 1
        if push.load_tries > _MAX_LOAD_TRIES:
            self._abort(push)

    def _has_device_room(self, num_tokens: int) -> bool:
        if not self._window_tokens:
            return (
                self._allocator.available_size() + self._cache.full_evictable_size()
                >= num_tokens
            )
        full_room = (
            self._allocator.full_available_size() + self._cache.full_evictable_size()
        )
        swa_room = (
            self._allocator.swa_available_size() + self._cache.swa_evictable_size()
        )
        return full_room >= num_tokens and swa_room >= self._window_tokens

    def _send(self, push: _Push) -> None:
        ready = _all_ranks_reduce(
            self._groups, [[int(push.producer_ready)]], dist.ReduceOp.MIN
        )[0][0]
        if not ready:
            return
        pages = self._pages(push)
        complete = _all_ranks_reduce(
            self._groups, [[int(pages is not None)]], dist.ReduceOp.MIN
        )[0][0]
        if not complete:
            self._abort(push)
            return
        blocks, swa_blocks = pages
        push.state = "sent"
        push.sent_at = time.monotonic()
        self.stats["sent"] += 1
        # A rank whose send fails gets no ack and settles at the timeout; the
        # decode stores only what all of its ranks received.
        sent = self._mgr.send_victim_push(
            decode=push.decode,
            seq=push.seq,
            tokens=push.tokens,
            blocks=blocks,
            swa_blocks=swa_blocks,
            extra_key=push.extra_key,
            cache_salt=push.cache_salt,
            is_eagle=push.is_eagle,
        )
        if not sent:
            push.stored = False

    def _pages(self, push: _Push) -> Optional[Tuple[List[int], List[int]]]:
        """Full page ids of the path and window page ids of its tail, None if
        the device does not hold both. Same on every rank."""
        num_tokens = push.num_tokens
        match = self._cache.match_prefix(MatchPrefixParams(key=push.radix_key()))
        full = match.device_indices
        if len(full) != num_tokens:
            return None
        swa_blocks: List[int] = []
        if self._window_tokens:
            swa = self._allocator.translate_swa_indices_for_transfer(
                full[num_tokens - self._window_tokens :]
            )
            if bool((swa <= 0).any()):
                return None
            swa_blocks = kv_to_page_indices(swa, self._page_size).tolist()
        return kv_to_page_indices(full, self._page_size).tolist(), swa_blocks

    def _abort(self, push: _Push) -> None:
        self.stats["aborted"] += 1
        self._finish(push, stored=False)


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


class _Incoming(msgspec.Struct):
    push: Any
    indices: Any
    swa_tokens: int
    # "allocated" until raiden admits the read, "reading", or "failed".
    state: str = "allocated"
    retries: int = 0


class DecodeReceiver:
    """Stores the sessions prefills push to one decode."""

    def __init__(self, *, queue: DecodeTransferQueue, mgr: RaidenKVManager) -> None:
        scheduler = queue.scheduler
        self._mgr = mgr
        self._cache = queue.tree_cache
        self._allocator = scheduler.token_to_kv_pool_allocator
        self._page_size = int(self._allocator.page_size)
        window = scheduler.sliding_window_size
        self._window_tokens = (
            window_tokens(window_size=window, page_size=self._page_size)
            if window
            else 0
        )
        self._tree_device = scheduler.req_to_token_pool.req_to_token.device
        self._groups = _cpu_groups(scheduler)
        self._iterations = 0
        self._rounds = 0
        # Round a received push was first seen in, by (prefill, seq).
        self._first_seen: Dict[Tuple[str, int], int] = {}
        self._incoming: Dict[Tuple[str, int], _Incoming] = {}
        self.stats = dict.fromkeys(
            ("received", "stored", "stored_tokens", "no_room", "failed", "dropped"), 0
        )
        self._last_log = [0.0]

    def holds_pages(self) -> bool:
        return bool(self._incoming)

    def step(self) -> None:
        """Every decode iteration, every rank; acts once per ``_CADENCE``."""
        self._iterations += 1
        if self._iterations % _CADENCE:
            return
        self._rounds += 1
        self._admit_agreed()
        if self._incoming:
            self._mgr._drain_engine()
            self._poll()
        _maybe_log(self.stats, "decode", self._last_log)

    def _admit_agreed(self) -> None:
        received = {
            key: push
            for key, push in self._mgr.victim_pushes().items()
            if key not in self._incoming
        }
        for key in received:
            if self._first_seen.setdefault(key, self._rounds) == self._rounds:
                self.stats["received"] += 1
        manifests = {
            (
                *key,
                *_manifest(
                    push.tokens,
                    push.extra_key,
                    push.cache_salt,
                    push.is_eagle,
                    len(push.blocks),
                    len(push.swa_blocks),
                ),
            ): key
            for key, push in received.items()
        }
        for manifest in _all_ranks_common(self._groups, list(manifests)):
            key = manifests[manifest]
            push = self._mgr.take_victim_push(key)
            self._first_seen.pop(key, None)
            if len(self._incoming) >= _MAX_INFLIGHT or not self._admit(key, push):
                self.stats["no_room"] += 1
                self._mgr.send_victim_ack(
                    endpoint=push.ack_endpoint, seq=push.seq, stored=False
                )
        for key, first in list(self._first_seen.items()):
            if self._rounds - first > _MAX_UNAGREED_ROUNDS:
                self._first_seen.pop(key)
                if self._mgr.take_victim_push(key) is not None:
                    self.stats["dropped"] += 1

    def _admit(self, key: Tuple[str, int], push: Any) -> bool:
        num_tokens = len(push.tokens) - int(push.is_eagle)
        swa_tokens = len(push.swa_blocks) * self._page_size
        valid = (
            num_tokens > 0
            and num_tokens == len(push.blocks) * self._page_size
            and push.is_eagle == self._cache.tree_core.is_eagle
            and swa_tokens == min(num_tokens, self._window_tokens)
        )
        valid = _all_ranks_reduce(self._groups, [[int(valid)]], dist.ReduceOp.MIN)[0][0]
        if not valid:
            return False
        indices = self._alloc(num_tokens=num_tokens, swa_tokens=swa_tokens)
        allocated = _all_ranks_reduce(
            self._groups, [[int(indices is not None)]], dist.ReduceOp.MIN
        )[0][0]
        if not allocated:
            if indices is not None:
                self._allocator.free(indices)
            return False
        incoming = _Incoming(push=push, indices=indices, swa_tokens=swa_tokens)
        self._incoming[key] = incoming
        self._start(incoming)
        return True

    def _alloc(self, *, num_tokens: int, swa_tokens: int):
        allocator = self._allocator
        cache = self._cache
        if not self._window_tokens:
            room = allocator.available_size() + cache.full_evictable_size()
            if room - num_tokens < _MIN_FREE_FRACTION * allocator.size:
                return None
            short = num_tokens - allocator.available_size()
            if short > 0:
                cache.evict(EvictParams(num_tokens=short))
            return allocator.alloc(num_tokens)
        full_room = allocator.full_available_size() + cache.full_evictable_size()
        swa_room = allocator.swa_available_size() + cache.swa_evictable_size()
        if (
            full_room - num_tokens < _MIN_FREE_FRACTION * allocator.size_full
            or swa_room - swa_tokens < _MIN_FREE_FRACTION * allocator.size_swa
        ):
            return None
        full_short = max(0, num_tokens - allocator.full_available_size())
        swa_short = max(0, swa_tokens - allocator.swa_available_size())
        if full_short or swa_short:
            cache.evict_for_alloc(
                EvictParams(num_tokens=full_short, swa_num_tokens=swa_short)
            )
        device = allocator.device
        return allocator.alloc_extend_swa_tail(
            prefix_lens=torch.tensor([0], dtype=torch.int64, device=device),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([num_tokens], dtype=torch.int64, device=device),
            seq_lens_cpu=torch.tensor([num_tokens], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64, device=device),
            extend_num_tokens=num_tokens,
            swa_tail_len=swa_tokens,
        )

    def _start(self, incoming: _Incoming) -> None:
        push = incoming.push
        local = [int(x) for x in kv_to_page_indices(incoming.indices, self._page_size)]
        swa_local: List[int] = []
        if incoming.swa_tokens:
            swa = self._allocator.translate_swa_indices_for_transfer(
                incoming.indices[
                    len(push.tokens) - int(push.is_eagle) - incoming.swa_tokens :
                ]
            )
            swa_local = [int(x) for x in kv_to_page_indices(swa, self._page_size)]
        outcome = self._mgr.start_victim_read(
            push=push, local=local, swa_local=swa_local
        )
        if outcome != "retry":
            incoming.state = "reading" if outcome == "started" else "failed"

    def _poll(self) -> None:
        items = list(self._incoming.items())
        rows = []
        for _, incoming in items:
            if incoming.state == "allocated":
                incoming.retries += 1
                if incoming.retries > _MAX_LOAD_TRIES:
                    incoming.state = "failed"
                else:
                    self._start(incoming)
            if incoming.state == "reading":
                local = self._mgr.victim_read_state(_read_ids(incoming.push))
            else:
                local = "failed" if incoming.state == "failed" else "pending"
            rows.append([int(local != "pending"), int(local == "done")])
        agreed = _all_ranks_reduce(self._groups, rows, dist.ReduceOp.MIN)
        for (key, incoming), (settled, done) in zip(items, agreed):
            if settled:
                self._settle(key, incoming, done=bool(done))

    def _settle(self, key: Tuple[str, int], incoming: _Incoming, *, done: bool) -> None:
        push = incoming.push
        self._incoming.pop(key)
        self._mgr.forget_victim_read(_read_ids(push))
        if done:
            self._insert(incoming)
            self.stats["stored"] += 1
            self.stats["stored_tokens"] += len(push.tokens) - int(push.is_eagle)
        else:
            self.stats["failed"] += 1
            self._allocator.free(incoming.indices)
        self._mgr.send_victim_ack(endpoint=push.ack_endpoint, seq=push.seq, stored=done)

    def _insert(self, incoming: _Incoming) -> None:
        num_tokens = len(incoming.push.tokens) - int(incoming.push.is_eagle)
        # The allocation bound window slots only for the tail, as the SWA
        # evicted length says. Pages duplicating nodes the tree holds are
        # freed by the insert.
        evicted = (
            {ComponentType.SWA: num_tokens - incoming.swa_tokens}
            if self._window_tokens
            else {}
        )
        self._cache.insert(
            InsertParams(
                key=RadixKey(
                    array("q", incoming.push.tokens),
                    extra_key=incoming.push.extra_key,
                    cache_salt=incoming.push.cache_salt,
                    is_bigram=incoming.push.is_eagle,
                ),
                value=incoming.indices.to(self._tree_device).to(torch.int64),
                prev_prefix_len=0,
                inserted_len=num_tokens,
                component_evicted_seqlens=evicted,
            )
        )


class VictimPrefixPuller(DecodePrefixPuller):
    """Prefill orchestration selected by Raiden's existing backend factory."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pusher = PrefillPusher(self)

    def commit(self, req):
        super().commit(req)
        self.pusher.record(req)

    def advance(self, reqs, *, waiting_reqs=()):
        super().advance(reqs, waiting_reqs=waiting_reqs)
        self.pusher.step()

    def release_all(self):
        self.pusher.release_all()
        super().release_all()


def _read_ids(push: Any) -> List[str]:
    return [rid for rid in (push.req_id, push.swa_req_id) if rid]


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

_pushers_by_cache = weakref.WeakKeyDictionary()

_receivers: weakref.WeakKeyDictionary[DecodeTransferQueue, DecodeReceiver] = (
    weakref.WeakKeyDictionary()
)


def _decode_add(
    upstream: Callable, queue: DecodePreallocQueue, req: Any, *args, **kwargs
):
    """Wraps ``DecodePreallocQueue.add``: the decode caches only pushed sessions."""
    if (
        isinstance(queue.kv_manager, RaidenKVManager)
        and queue.kv_manager.prefix_pull_active
        and isinstance(queue.tree_cache, UnifiedRadixCache)
    ):
        from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore

        if isinstance(queue.tree_cache.tree_core, UnifiedTreeCore):
            if queue.tree_cache.cache_controller is None:
                raise ValueError(
                    "Victim prefix caching requires --enable-hierarchical-cache on both roles"
                )
            if (
                queue.tree_cache.supports_swa()
                and not queue.tree_cache.tree_core.has_swa_host_pool
            ):
                raise ValueError(
                    "Victim SWA prefix caching requires SGLANG_TPU_PD_DECODE_SWA_HOST_WINDOWS > 0"
                )
            req.skip_radix_cache_insert = True
    return upstream(queue, req, *args, **kwargs)


def _receiver(queue: DecodeTransferQueue) -> Optional[DecodeReceiver]:
    receiver = _receivers.get(queue)
    if receiver is None:
        mgr = queue.scheduler.disagg_decode_prealloc_queue.kv_manager
        if (
            not isinstance(mgr, RaidenKVManager)
            or not mgr.prefix_pull_active
            or not isinstance(queue.tree_cache, UnifiedRadixCache)
        ):
            return None
        receiver = _receivers[queue] = DecodeReceiver(queue=queue, mgr=mgr)
    return receiver


def _resolve_deferred_releases(upstream: Callable, queue: DecodeTransferQueue) -> Any:
    """Wraps ``DecodeTransferQueue.resolve_deferred_releases``, which every rank
    runs each decode iteration: drive the receiver."""
    result = upstream(queue)
    receiver = _receiver(queue)
    if receiver is not None:
        receiver.step()
    return result


def _has_pending_deferred_releases(
    upstream: Callable, queue: DecodeTransferQueue
) -> bool:
    """Wraps ``DecodeTransferQueue.has_pending_deferred_releases``: pages read
    for pushed sessions sit outside the tree by design too, so the idle pool
    leak check waits for them as for deferred releases."""
    receiver = _receivers.get(queue)
    return upstream(queue) or (receiver is not None and receiver.holds_pages())


def _checkpoint(upstream, cache, req, **kwargs):
    result = upstream(cache, req, **kwargs)
    ref = _pushers_by_cache.get(cache)
    pusher = ref() if ref is not None else None
    if pusher is not None:
        pusher.remember(req)
    return result


def _reset_cache(upstream, cache):
    ref = _pushers_by_cache.get(cache)
    pusher = ref() if ref is not None else None
    if pusher is not None:
        pusher.release_all()
        pusher._routes.clear()
        pusher._pushed_len.clear()
    if any(
        receiver._cache is cache and receiver.holds_pages()
        for receiver in list(_receivers.values())
    ):
        raise RuntimeError(
            "Cannot reset KV memory while victim reads are active; retry after they drain"
        )
    return upstream(cache)


def _patch_targets() -> tuple[tuple[Any, str, Callable], ...]:
    return (
        (UnifiedRadixCache, "reset", _reset_cache),
        (UnifiedRadixCache, "checkpoint", _checkpoint),
        (DecodePreallocQueue, "add", _decode_add),
        (DecodeTransferQueue, "resolve_deferred_releases", _resolve_deferred_releases),
        (
            DecodeTransferQueue,
            "has_pending_deferred_releases",
            _has_pending_deferred_releases,
        ),
    )


def _wrap(owner: Any, name: str, impl: Callable) -> None:
    upstream = getattr(owner, name)

    @functools.wraps(upstream)
    def wrapper(*args, **kwargs):
        return impl(upstream, *args, **kwargs)

    setattr(owner, name, wrapper)


def install() -> None:
    global _is_installed
    if _is_installed or not tpu_envs.SGLANG_DISAGG_PREFIX_VICTIM_PUSH.get():
        return
    _is_installed = True
    for owner, name, impl in _patch_targets():
        _wrap(owner, name, impl)
