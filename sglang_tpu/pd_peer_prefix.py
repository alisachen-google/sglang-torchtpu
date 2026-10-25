# SPDX-License-Identifier: Apache-2.0
"""Scheduler side of the reverse prefix pull (see
``disaggregation/raiden/peer_prefix.py``).

Decode engine: after each ``process_decode_queue`` the manager serves the
prefix requests of its prefill peers from the decode radix tree.

Prefill engine: a request whose prefix is not in the prefill's own tiers asks
the decode peer when it enters the queue. ``pop_bootstrapped`` holds the
request until the pull lands, inserts the pulled pages into the radix tree and
only then lets the request be scheduled, so it prefills the remaining tokens.

All decisions that change what a request does are taken from values reduced
across the attention-TP ranks, so every rank inserts the same key and
schedules the same request in the same step.

Enable with ``SGLANG_TPU_PEER_PREFIX=1`` on both engines. The decode engine
needs ``--disaggregation-decode-enable-radix-cache --enable-hierarchical-cache``.
"""

from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, List, Optional

import torch
import torch.distributed as dist

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_policy import _match_prefix_result
from sglang.srt.mem_cache.allocation import evict_from_tree_cache
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.common import kv_to_page_indices
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.runtime_context import get_disagg
from sglang_tpu.disaggregation.raiden.peer_demote import (
    LandingContext,
    SweepContext,
)
from sglang_tpu.disaggregation.raiden.peer_prefix import (
    STAGE_DONE,
    STAGE_FAILED,
    STAGE_REPLIED,
    ProducerContext,
    page_floor,
)

logger = logging.getLogger("sglang_tpu.pd_peer_prefix")

ENV_ENABLE = "SGLANG_TPU_PEER_PREFIX"
ENV_TIMEOUT = "SGLANG_TPU_PEER_PREFIX_TIMEOUT_S"
ENV_MIN_TOKENS = "SGLANG_TPU_PEER_PREFIX_MIN_TOKENS"
ENV_OFFER = "SGLANG_TPU_PEER_PREFIX_OFFER"
ENV_OFFER_HEADROOM = "SGLANG_TPU_PEER_PREFIX_OFFER_HEADROOM"


def offers_enabled() -> bool:
    return os.environ.get(ENV_OFFER, "1") == "1"


def _offer_headroom() -> float:
    """Fraction of the prefill pool that must stay free after a speculative pull."""
    return float(os.environ.get(ENV_OFFER_HEADROOM, "0.5"))


def enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "0") == "1"


def _timeout_s() -> float:
    return float(os.environ.get(ENV_TIMEOUT, "120"))


ENV_GATE_TIMEOUT = "SGLANG_TPU_PEER_PREFIX_GATE_TIMEOUT_S"


def _gate_timeout_s() -> float:
    """A pull decided at the gate holds the prefill's next slot; recompute
    is a few seconds away, so it waits only briefly for the decode."""
    return float(os.environ.get(ENV_GATE_TIMEOUT, "20"))


ENV_JIT = "SGLANG_TPU_PEER_PREFIX_JIT"
ENV_JIT_DEPTH = "SGLANG_TPU_PEER_PREFIX_JIT_DEPTH"


def jit_enabled() -> bool:
    return os.environ.get(ENV_JIT, "1") == "1"


def _jit_depth() -> int:
    return max(1, int(os.environ.get(ENV_JIT_DEPTH, "1")))


def _min_tokens(page_size: int) -> int:
    return max(page_size, int(os.environ.get(ENV_MIN_TOKENS, "8192")))


# Pressure-driven demotion of the prefill's cold host tier into the decode's
# host tier (vllm-torchtpu's evict sweep; see raiden/peer_demote.py). Off by
# default; needs SGLANG_TPU_PEER_PREFIX=1 on both engines as well.
ENV_DEMOTE = "SGLANG_TPU_PEER_DEMOTE"
ENV_DEMOTE_LOW = "SGLANG_TPU_PEER_DEMOTE_LOW"  # start an episode under this free ratio
ENV_DEMOTE_HIGH = "SGLANG_TPU_PEER_DEMOTE_HIGH"  # end it at this free ratio
ENV_DEMOTE_BATCH = "SGLANG_TPU_PEER_DEMOTE_BATCH"  # pages per offer
ENV_DEMOTE_PERIOD = "SGLANG_TPU_PEER_DEMOTE_PERIOD_S"  # idle watermark check period
ENV_DEMOTE_TIMEOUT = "SGLANG_TPU_PEER_DEMOTE_TIMEOUT_S"  # ack and push deadline


def demote_enabled() -> bool:
    return enabled() and os.environ.get(ENV_DEMOTE, "0") == "1"


def _demote_low() -> float:
    return float(os.environ.get(ENV_DEMOTE_LOW, "0.03"))


def _demote_high() -> float:
    return float(os.environ.get(ENV_DEMOTE_HIGH, "0.05"))


def _demote_batch_pages() -> int:
    return max(1, int(os.environ.get(ENV_DEMOTE_BATCH, "128")))


def _demote_period_s() -> float:
    return float(os.environ.get(ENV_DEMOTE_PERIOD, "30"))


def _demote_timeout_s() -> float:
    return float(os.environ.get(ENV_DEMOTE_TIMEOUT, str(_timeout_s())))


def _all_reduce(values: List[int], group, op) -> List[int]:
    if not values:
        return values
    t = torch.tensor(values, dtype=torch.int64)
    if group is not None and dist.is_initialized() and dist.get_world_size(group) > 1:
        dist.all_reduce(t, op=op, group=group)
    return t.tolist()


# --------------------------------------------------------------------------- #
# Decode engine
# --------------------------------------------------------------------------- #
def _producer_context(scheduler) -> Optional[ProducerContext]:
    ctx = getattr(scheduler, "_peer_prefix_ctx", None)
    if ctx is not None:
        return ctx
    tree = scheduler.tree_cache
    allocator = scheduler.token_to_kv_pool_allocator
    hybrid = bool(getattr(tree, "is_swa_enabled", False))
    page_size = allocator.page_size
    ctx = ProducerContext(
        tree_cache=tree,
        allocator=allocator,
        page_size=page_size,
        hybrid=hybrid,
        sliding_window_size=getattr(scheduler, "sliding_window_size", None),
        # The decode's host copy of a hybrid model holds full-attention KV
        # only unless SGLANG_TPU_DECODE_SWA_HOST=1 keeps the window pages too;
        # a restored prefix without them is useless to the prefill.
        restore_allowed=bool(scheduler.enable_decode_hicache)
        and (not hybrid or os.environ.get("SGLANG_TPU_DECODE_SWA_HOST", "0") == "1"),
        min_hit_tokens=_min_tokens(page_size),
        lease_timeout_s=max(
            _timeout_s() + 60.0,
            float(os.environ.get("SGLANG_RAIDEN_TIMEOUT_S", "600")) + 60.0,
        ),
        is_eagle=bool(getattr(getattr(tree, "tree_core", None), "is_eagle", False)),
        lookup_room=_room_lookup(scheduler),
        gather=_make_gather(getattr(scheduler, "attn_tp_cpu_group", None)),
    )
    scheduler._peer_prefix_ctx = ctx
    return ctx


def _make_gather(group):
    """all_gather_object over the attention-TP CPU group, or None alone."""
    if group is None or not dist.is_initialized() or dist.get_world_size(group) <= 1:
        return None
    world = dist.get_world_size(group)

    def gather(obj):
        out = [None] * world
        dist.all_gather_object(out, obj, group=group)
        return out

    return gather


def _room_lookup(scheduler):
    """The decode's DecodeRequest for a room while it is pre-allocated or in
    transfer; None once it runs or when it is unknown."""

    def lookup(room):
        for name in ("disagg_decode_prealloc_queue", "disagg_decode_transfer_queue"):
            queue = getattr(scheduler, name, None)
            for dr in getattr(queue, "queue", None) or ():
                if getattr(getattr(dr, "req", None), "bootstrap_room", None) == room:
                    return dr
        return None

    return lookup


_OFFER_MANAGERS: dict = {}  # id(tree_cache) -> kv manager, decode side


def _checkpoint(upstream: Callable, cache, req, *args, **kwargs):
    """Wraps ``UnifiedRadixCache.checkpoint``: a finished request on a decode
    that serves pulls is offered to the prefill peer."""
    out = upstream(cache, req, *args, **kwargs)
    manager = _OFFER_MANAGERS.get(id(cache))
    if manager is None or not offers_enabled():
        return out
    try:
        finished = getattr(req, "finished", None)
        if callable(finished) and not finished():
            return out
        room = getattr(req, "bootstrap_room", None)
        if room is None or getattr(req, "is_retracted", False):
            return out
        page_size = getattr(cache, "page_size", 1)
        tokens = list(req.origin_input_ids) + list(getattr(req, "output_ids", []) or [])
        n = page_floor(len(tokens), page_size)
        if n >= _min_tokens(page_size):
            manager.peer_offer(room, tokens[:n], req.extra_key, req.cache_salt)
    except Exception:  # noqa: BLE001
        logger.exception("peer prefix: offer failed for %s", getattr(req, "rid", "?"))
    return out


def _process_decode_queue(upstream: Callable, scheduler) -> None:
    upstream(scheduler)
    queue = getattr(scheduler, "disagg_decode_prealloc_queue", None)
    manager = getattr(queue, "kv_manager", None)
    serve = getattr(manager, "peer_prefix_serve", None)
    if serve is None:
        return
    if not get_disagg().disaggregation_decode_enable_radix_cache:
        return
    ctx = _producer_context(scheduler)
    if ctx is not None:
        _OFFER_MANAGERS.setdefault(id(scheduler.tree_cache), manager)
        serve(ctx)
    if demote_enabled() and hasattr(manager, "peer_demote_serve"):
        land_ctx = _landing_context(scheduler)
        if land_ctx is not None:
            try:
                manager.peer_demote_serve(land_ctx)
            except Exception:  # noqa: BLE001
                logger.exception("peer demote: serve failed")


# --------------------------------------------------------------------------- #
# Prefill engine
# --------------------------------------------------------------------------- #
# Pages a pull has allocated but not yet handed to the radix tree. The idle
# pool invariant counts only tree and request pages, so these are reported as
# "uncached" through the wrapped invariant checks below.
_INFLIGHT: dict = {}


def _inflight_tokens() -> int:
    return sum(
        len(f.new_indices) for f in _INFLIGHT.values() if f.new_indices is not None
    )


def _check_full_pool(upstream: Callable, checker, ps, uncached: int = 0):
    return upstream(checker, ps, uncached=uncached + _inflight_tokens())


def _check_swa_pool(upstream: Callable, checker, ps, uncached: int = 0):
    return upstream(checker, ps, uncached=uncached + _inflight_tokens())


@dataclass
class _Fetch:
    room: int
    matched_len: int
    want_len: int
    hit_len: int = 0
    new_indices: Optional[torch.Tensor] = None
    lock_node: Any = None
    lock_params: Any = None
    prefix_indices: Optional[torch.Tensor] = None
    reading: bool = False
    remote_blocks: List[int] = None
    local_blocks: List[int] = None
    remote_swa_blocks: List[int] = None
    local_swa_blocks: List[int] = None
    timeout: float = 0.0


def _add_request_to_queue(
    upstream: Callable, scheduler, req, is_retracted: bool = False
):
    upstream(scheduler, req, is_retracted)
    try:
        _maybe_begin_fetch(scheduler, req, is_retracted)
    except Exception:  # noqa: BLE001
        logger.exception("peer prefix: begin failed for %s", getattr(req, "rid", "?"))


def _maybe_begin_fetch(
    scheduler, req, is_retracted: bool, *, at_pop: bool = False
) -> None:
    """Decide a pull for ``req``: at arrival, and again when it leaves the
    bootstrap queue. The wait there is long when the decode is full, and the
    prefill's own tiers can lose the prefix meanwhile; the decode keeps it."""
    if scheduler.disaggregation_mode != DisaggregationMode.PREFILL or is_retracted:
        return
    if getattr(req, "bootstrap_room", None) is None or req.finished():
        return
    queue = scheduler.disagg_prefill_bootstrap_queue
    manager = getattr(queue, "kv_manager", None)
    if manager is None or not hasattr(manager, "peer_fetch_begin"):
        return
    tree = scheduler.tree_cache
    allocator = scheduler.token_to_kv_pool_allocator
    page_size = allocator.page_size
    tokens = req.origin_input_ids
    result = _match_prefix_result(tree, req, tokens)
    matched = page_floor(len(result.device_indices), page_size)
    want = page_floor(req._compute_max_prefix_len(len(tokens)), page_size)
    # Every rank must take the same decision. The local match is in lockstep
    # already; whether the decode peer is known is not, until the first pull.
    local_ok = int(
        result.host_hit_length == 0
        and want - matched >= _min_tokens(page_size)
        and manager.peer_known()
    )
    ok = _all_reduce([local_ok], scheduler.attn_tp_cpu_group, dist.ReduceOp.MIN)[0]
    if not ok:
        return
    if not manager.peer_fetch_begin(
        req.bootstrap_room, tokens[:want], matched, req.extra_key, req.cache_salt
    ):
        # The send failed on this rank only; the hold below reduces stages, so
        # a missing fetch surfaces as a failure on every rank.
        pass
    fetch = _Fetch(room=req.bootstrap_room, matched_len=matched, want_len=want)
    fetch.timeout = _gate_timeout_s() if at_pop else _timeout_s()
    if at_pop:
        logger.info(
            "peer prefix pull at schedule time | room=%s matched=%d want=%d",
            req.bootstrap_room,
            matched,
            want,
        )
    # Snapshot of the local prefix. It is not locked while the pull is in
    # flight: a lock would pin a whole context per held request and starve
    # the prefill's own scheduling. The commit re-matches and gives up if the
    # prefix moved.
    fetch.prefix_indices = result.device_indices[:matched].cpu().to(torch.int64)
    req._peer_fetch = fetch


def _pop_bootstrapped(upstream: Callable, queue, *args, **kwargs):
    out = upstream(queue, *args, **kwargs)
    if isinstance(out, tuple):
        reqs, failed = out
    else:
        reqs, failed = out, None
    held = queue.__dict__.setdefault("_peer_held", [])
    deferred = queue.__dict__.setdefault("_peer_deferred", [])
    passthrough = []
    for req in reqs:
        if getattr(req, "_peer_fetch", None) is not None:
            held.append(req)
        elif getattr(req, "_peer_offer", False):
            passthrough.append(req)
        elif jit_enabled():
            # Just in time: a popped request would wait in the waiting queue
            # behind the requests computing ahead of it, long enough for the
            # prefill's tiers to evict its prefix. It is released, and the
            # pull decided, only when the waiting queue has room for it.
            deferred.append(req)
        else:
            _decide_at_pop(queue, req)
            (
                held if getattr(req, "_peer_fetch", None) is not None else passthrough
            ).append(req)
    # Speculative pulls from decode offers ride the same hold logic as
    # requests; they are never handed to the scheduler.
    _admit_offer(queue, held)
    # _advance_held reduces every decision across ranks and must not raise;
    # an escaping error here would desynchronize the ranks' collectives.
    for req in _advance_held(queue, held):
        if getattr(req, "_peer_offer", False):
            continue
        abort = getattr(req, "_peer_abort_req", None)
        if abort is not None and _UPSTREAM_ABORT is not None:
            # Hand it to the scheduler's own abort path, which expects the
            # request in the waiting queue.
            req._peer_abort_req = None
            queue.scheduler.waiting_queue.append(req)
            try:
                _UPSTREAM_ABORT(queue.scheduler, abort)
            except Exception:  # noqa: BLE001
                logger.exception("peer prefix: abort of %s failed", req.rid)
            continue
        # A request whose arrival-time fetch just settled, as a commit or a
        # miss, still goes through the gate: it waits for room like any other
        # and the pull is decided again when it is released.
        (deferred if jit_enabled() else passthrough).append(req)
    # The waiting queue and the held list are lockstep state, so every rank
    # releases the same deferred requests in the same tick.
    room = (
        _jit_depth()
        - len(getattr(queue.scheduler, "waiting_queue", ()) or ())
        - len(held)
    )
    while deferred and room > 0:
        req = deferred.pop(0)
        # One decision per request at the gate. A request whose gate pull
        # settled comes back through the deferred list and must then run,
        # or a prompt the decode does not hold would be asked for forever.
        if not getattr(req, "_peer_decided", False):
            req._peer_decided = True
            _decide_at_pop(queue, req)
        (held if getattr(req, "_peer_fetch", None) is not None else passthrough).append(
            req
        )
        room -= 1
    _release_scheduled_locks(queue, held, deferred)
    _demote_tick(queue)
    return passthrough if failed is None else (passthrough, failed)


def _lock_until_scheduled(queue, req, hit: int) -> None:
    """Pin a prefix the pull just inserted until the scheduler takes the
    request from the waiting queue: the chunked prefill running ahead of it
    allocates as it goes and could evict the fresh pages first."""
    tree = queue.scheduler.tree_cache
    try:
        res = _match_prefix_result(tree, req, req.origin_input_ids[:hit])
        node = res.last_device_node
        req._peer_lock = (node, tree.inc_lock_ref(node).to_dec_params())
        queue.__dict__.setdefault("_peer_locked", []).append(req)
    except Exception:  # noqa: BLE001
        logger.exception("peer prefix: lock after commit failed for %s", req.rid)


def _release_scheduled_locks(queue, held, deferred) -> None:
    locked = queue.__dict__.get("_peer_locked")
    if not locked:
        return
    waiting = getattr(queue.scheduler, "waiting_queue", ()) or ()
    tree = queue.scheduler.tree_cache
    for req in list(locked):
        if req in waiting or req in held or req in deferred:
            continue
        lock = getattr(req, "_peer_lock", None)
        req._peer_lock = None
        locked.remove(req)
        if lock is None:
            continue
        try:
            tree.dec_lock_ref(*lock)
        except Exception:  # noqa: BLE001
            logger.exception("peer prefix: unlock failed for %s", req.rid)


def _decide_at_pop(queue, req) -> None:
    """The popped list is all-reduced upstream, so this per-request decision
    runs in lockstep on every rank."""
    try:
        _maybe_begin_fetch(queue.scheduler, req, False, at_pop=True)
    except Exception:  # noqa: BLE001
        logger.exception(
            "peer prefix: schedule-time begin failed for %s", getattr(req, "rid", "?")
        )


def _admit_offer(queue, held: List[Any]) -> None:
    """Admit at most one decode offer per iteration, the same one on every rank.

    Rank 0 names the oldest offer it holds; the others answer whether they
    hold it too. Only when every rank has it does each rank start the pull,
    so the held list stays identical across ranks."""
    if not offers_enabled():
        return
    scheduler = queue.scheduler
    manager = queue.kv_manager
    group = scheduler.attn_tp_cpu_group
    if any(getattr(r, "_peer_offer", False) for r in held):
        return  # one speculative pull at a time
    try:
        rank = (
            dist.get_rank(group) if group is not None and dist.is_initialized() else 0
        )
        world = (
            dist.get_world_size(group)
            if group is not None and dist.is_initialized()
            else 1
        )
    except Exception:  # noqa: BLE001
        rank, world = 0, 1
    if world > 1:
        room_t = torch.tensor(
            [manager.peer_peek_offer_room() if rank == 0 else -1], dtype=torch.int64
        )
        dist.broadcast(
            room_t,
            src=dist.get_global_rank(group, 0)
            if hasattr(dist, "get_global_rank")
            else 0,
            group=group,
        )
        room = int(room_t[0])
        if room < 0:
            return
        have = _all_reduce([int(_has_offer(manager, room))], group, dist.ReduceOp.MIN)[
            0
        ]
        if not have:
            return
        offer = manager.peer_take_offer(room)
    else:
        offer = manager.peer_take_offer()
    if offer is None:
        return
    try:
        _begin_offer_fetch(queue, held, offer)
    except Exception:  # noqa: BLE001
        logger.exception("peer prefix: offer admission failed for room %s", offer.room)


def _has_offer(manager, room: int) -> bool:
    with manager._peer_lock:
        return any(o.room == room for o in manager._peer_offers)


def _begin_offer_fetch(queue, held: List[Any], offer) -> None:
    scheduler = queue.scheduler
    manager = queue.kv_manager
    tree = scheduler.tree_cache
    allocator = scheduler.token_to_kv_pool_allocator
    page_size = allocator.page_size
    from array import array

    tokens = array("q", offer.token_ids)
    req = SimpleNamespace(
        rid=f"offer:{offer.room}",
        bootstrap_room=offer.room,
        origin_input_ids=tokens,
        extra_key=offer.extra_key,
        cache_salt=offer.cache_salt,
        _peer_fetch=None,
        _peer_offer=True,
        finished=lambda: False,
    )
    result = _match_prefix_result(tree, req, tokens)
    matched = page_floor(len(result.device_indices), page_size)
    want = page_floor(len(tokens), page_size)
    gap = want - matched
    # Only pull when the pages fit with headroom to spare; the pulled pages
    # are evictable the moment they are inserted, so this is a hint, not a lock.
    free = allocator.available_size()
    total = getattr(allocator, "size", None) or free
    ok = (
        result.host_hit_length == 0
        and gap >= _min_tokens(page_size)
        and free - gap >= _offer_headroom() * total
        and manager.peer_known()
    )
    ok = _all_reduce([int(ok)], scheduler.attn_tp_cpu_group, dist.ReduceOp.MIN)[0]
    if not ok:
        return
    manager.peer_fetch_begin(
        offer.room, tokens[:want], matched, offer.extra_key, offer.cache_salt
    )
    fetch = _Fetch(room=offer.room, matched_len=matched, want_len=want)
    fetch.prefix_indices = result.device_indices[:matched].cpu().to(torch.int64)
    req._peer_fetch = fetch
    held.append(req)
    logger.info(
        "peer prefix offer accepted | room=%s matched=%d want=%d",
        offer.room,
        matched,
        want,
    )


def _advance_held(queue, held: List[Any]) -> List[Any]:
    """Advance every held request one step. Each step's decision is taken
    from values reduced over the attention-TP ranks, so the ranks hold, pull,
    insert and release the same requests in the same scheduler iteration. A
    rank-local problem only marks that rank's fetch failed; the next
    iteration's MAX reduce then makes every rank abandon the pull."""
    if not held:
        _reap_draining(queue)
        return []
    scheduler = queue.scheduler
    manager = queue.kv_manager
    group = scheduler.attn_tp_cpu_group
    timeout = _timeout_s()
    # Aborted while the pull was in flight (flagged in lockstep by the
    # abort handler): give the pull up and hand the request back.
    aborted = [r for r in held if getattr(r, "_peer_abort_req", None) is not None]
    for req in aborted:
        _abandon(queue, req)
        held.remove(req)
    if not held:
        _reap_draining(queue)
        return aborted
    stages, hits = [], []
    for req in held:
        fetch = req._peer_fetch
        stage = manager.peer_fetch_poll(
            fetch.room, getattr(fetch, "timeout", 0.0) or timeout
        )
        peer = manager.peer_fetch(fetch.room)
        stages.append(stage)
        hits.append(peer.hit_len if peer is not None else 0)
    min_stage = _all_reduce(stages, group, dist.ReduceOp.MIN)
    max_stage = _all_reduce(stages, group, dist.ReduceOp.MAX)
    min_hit = _all_reduce(hits, group, dist.ReduceOp.MIN)
    released = []
    # Replied on every rank: prepare the pull locally, then agree on it.
    # Every rank sees the same in-flight set, so the budget gate is consistent.
    # A pull larger than the budget is still admitted when nothing is in
    # flight, or it would wait forever; the budget bounds the overlap, not
    # the size of one pull.
    inflight = manager.peer_inflight_pages()
    budget = manager.peer_read_budget_pages() - inflight
    page_size = scheduler.token_to_kv_pool_allocator.page_size
    prepare_idx = []
    for i in range(len(held)):
        if not (min_stage[i] == STAGE_REPLIED and max_stage[i] == STAGE_REPLIED):
            continue
        need = max(0, min_hit[i] - held[i]._peer_fetch.matched_len) // page_size
        if need > budget and (inflight > 0 or prepare_idx):
            continue  # wait for in-flight reads to settle
        budget -= need
        prepare_idx.append(i)
    if prepare_idx:
        ready = []
        for i in prepare_idx:
            try:
                ready.append(int(_prepare_read(queue, held[i], min_hit[i])))
            except Exception:  # noqa: BLE001
                logger.exception("peer prefix: prepare failed for %s", held[i].rid)
                ready.append(0)
        agreed = _all_reduce(ready, group, dist.ReduceOp.MIN)
        for i, ok in zip(prepare_idx, agreed):
            req = held[i]
            if ok:
                try:
                    if not _issue_read(queue, req):
                        _mark_failed(manager, req, "start_read refused")
                except Exception:  # noqa: BLE001
                    logger.exception("peer prefix: start_read failed for %s", req.rid)
                    _mark_failed(manager, req, "start_read raised")
            else:
                _abandon(queue, req)
                released.append(req)
    for i, req in enumerate(list(held)):
        if req in released:
            continue
        if max_stage[i] == STAGE_FAILED:
            _abandon(queue, req)
            released.append(req)
        elif min_stage[i] == STAGE_DONE:
            try:
                _commit(queue, req)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "peer prefix: commit failed for %s; releasing", req.rid
                )
                _abandon(queue, req)
            released.append(req)
        # STAGE_SENT / STAGE_READING on some rank: wait.
    for req in released:
        held.remove(req)
    _reap_draining(queue)
    return aborted + released


def _mark_failed(manager, req, reason: str) -> None:
    fetch = getattr(req, "_peer_fetch", None)
    peer = manager.peer_fetch(fetch.room) if fetch is not None else None
    if peer is not None:
        peer.error = peer.error or reason
        peer.stage = STAGE_FAILED


def _prepare_read(queue, req, hit: int) -> bool:
    """Allocate this rank's landing pages for a pull of ``hit`` tokens and
    check them against the decode's reply. Rank-local; the caller reduces
    the result before any read starts."""
    scheduler = queue.scheduler
    manager = queue.kv_manager
    allocator = scheduler.token_to_kv_pool_allocator
    page_size = allocator.page_size
    fetch = req._peer_fetch
    hit = min(page_floor(hit, page_size), fetch.want_len)
    hybrid = bool(getattr(scheduler.tree_cache, "is_swa_enabled", False))
    window = getattr(scheduler, "sliding_window_size", None)

    def refuse(reason: str) -> bool:
        # A plain miss (hit 0) is routine; log the real refusals.
        (logger.debug if hit == 0 else logger.info)(
            "peer prefix prepare refused | room=%s reason=%s hit=%d matched=%d",
            fetch.room,
            reason,
            hit,
            fetch.matched_len,
        )
        return False

    if hybrid and (
        window is None
        or hit - fetch.matched_len < page_floor(window, page_size) + page_size
    ):
        return refuse("window short")
    if hit <= fetch.matched_len:
        return refuse("no gain")
    peer = manager.peer_fetch(fetch.room)
    if peer is None:
        return refuse("no reply")
    n_new = hit - fetch.matched_len
    if len(peer.blocks) != n_new // page_size:
        return refuse("block count mismatch")
    last_loc = int(fetch.prefix_indices[-1]) if fetch.matched_len > 0 else -1
    # The landing pages come from the same pool the chunked prefill fills;
    # evict cache pages for the shortfall first, as every extend allocation
    # does. Lockstep: every rank prepares the same pulls in the same tick.
    tree = scheduler.tree_cache
    if hasattr(tree, "token_to_kv_pool_allocator"):
        try:
            evict_from_tree_cache(tree, n_new + page_size)
        except Exception:  # noqa: BLE001
            logger.exception("peer prefix: evict before pull failed")
    new_indices = allocator.alloc_extend(
        torch.tensor([fetch.matched_len], dtype=torch.int64),
        torch.tensor([fetch.matched_len], dtype=torch.int64),
        torch.tensor([hit], dtype=torch.int64),
        torch.tensor([hit], dtype=torch.int64),
        torch.tensor([last_loc], dtype=torch.int64),
        n_new,
    )
    if new_indices is None:
        return refuse("alloc failed")
    new_indices = new_indices.cpu().to(torch.int64)
    fetch.hit_len = hit
    fetch.new_indices = new_indices
    _INFLIGHT[id(fetch)] = fetch
    local_blocks = [int(b) for b in kv_to_page_indices(new_indices, page_size)]
    swa_blocks: List[int] = []
    if hybrid:
        window_start = page_floor(max(0, hit - window), page_size)
        swa_idx = allocator.translate_loc_from_full_to_swa(
            new_indices[window_start - fetch.matched_len :]
        )
        swa_blocks = [int(b) for b in kv_to_page_indices(swa_idx, page_size)]
    # Every rank must pull the same token range, so the decode ranks must have
    # answered with the same hit. A rank whose decode found more or fewer pages
    # than the agreed hit makes every rank give up on this pull.
    remote_blocks = list(peer.blocks)
    remote_swa = list(peer.swa_blocks)
    if len(remote_blocks) != len(local_blocks) or len(remote_swa) != len(swa_blocks):
        _release_pages(allocator, fetch)
        return refuse("rank block mismatch")
    fetch.remote_blocks = remote_blocks
    fetch.local_blocks = local_blocks
    fetch.remote_swa_blocks = remote_swa
    fetch.local_swa_blocks = swa_blocks
    return True


def _issue_read(queue, req) -> bool:
    manager = queue.kv_manager
    fetch = req._peer_fetch
    peer = manager.peer_fetch(fetch.room)
    if peer is None:
        return False
    peer.blocks = list(fetch.remote_blocks)
    peer.swa_blocks = list(fetch.remote_swa_blocks)
    if not manager.peer_fetch_start_read(
        fetch.room, fetch.local_blocks, fetch.local_swa_blocks
    ):
        return False
    fetch.reading = True
    return True


def _release_pages(allocator, fetch) -> None:
    """Return a fetch's pages to the allocator and drop them from the in-flight set."""
    _INFLIGHT.pop(id(fetch), None)
    if fetch.new_indices is not None:
        allocator.free(fetch.new_indices)
        fetch.new_indices = None


def _commit(queue, req) -> None:
    """Every rank's read settled: hand the pulled pages to the radix tree."""
    scheduler = queue.scheduler
    manager = queue.kv_manager
    tree = scheduler.tree_cache
    allocator = scheduler.token_to_kv_pool_allocator
    page_size = allocator.page_size
    fetch = req._peer_fetch
    hit = fetch.hit_len
    tokens = req.origin_input_ids[:hit]
    if fetch.matched_len > 0:
        # The local prefix was not locked during the pull. It must still be
        # the same device pages, or the pulled tail has nothing to attach to.
        again = _match_prefix_result(
            tree, req, req.origin_input_ids[: fetch.matched_len]
        )
        current = again.device_indices[: fetch.matched_len].cpu().to(torch.int64)
        if len(current) != fetch.matched_len or not torch.equal(
            current, fetch.prefix_indices
        ):
            logger.info(
                "peer prefix dropped | rid=%s local prefix moved (%d -> %d tokens)",
                req.rid,
                fetch.matched_len,
                len(current),
            )
            _abandon(queue, req)
            return
    values = torch.cat([fetch.prefix_indices, fetch.new_indices])
    params = InsertParams(
        key=RadixKey(
            tokens,
            req.extra_key,
            is_bigram=bool(
                getattr(getattr(tree, "tree_core", None), "is_eagle", False)
            ),
            cache_salt=req.cache_salt,
        ),
        value=values,
        prev_prefix_len=fetch.matched_len,
        inserted_len=fetch.matched_len,
    )
    hybrid = bool(getattr(tree, "is_swa_enabled", False))
    if hybrid:
        window = scheduler.sliding_window_size
        window_start = page_floor(max(0, hit - window), page_size)
        if window_start > fetch.matched_len:
            allocator.free_swa_segment(
                fetch.new_indices[: window_start - fetch.matched_len],
                start_pos=fetch.matched_len,
            )
        params.set_evicted_seqlen(ComponentType.SWA, window_start)
    result = tree.insert(params)
    declined = bool(getattr(result, "rotation_tail_declined", False))
    if declined:
        _release_pages(allocator, fetch)
    else:
        _INFLIGHT.pop(id(fetch), None)  # the tree owns the pages now
    _unlock(tree, fetch)
    manager.peer_fetch_end(fetch.room)
    req._peer_fetch = None
    if not declined and jit_enabled():
        _lock_until_scheduled(queue, req, hit)
    logger.info(
        "peer prefix committed | rid=%s room=%s matched=%d hit=%d declined=%s",
        req.rid,
        fetch.room,
        fetch.matched_len,
        hit,
        declined,
    )


def _unlock(tree, fetch) -> None:
    if fetch.lock_node is not None and fetch.lock_params is not None:
        tree.dec_lock_ref(fetch.lock_node, fetch.lock_params)
        fetch.lock_node = None
        fetch.lock_params = None


def _abandon(queue, req, free_pages: bool = True) -> None:
    """Give up on the pull; the request prefills what it misses."""
    try:
        _abandon_impl(queue, req, free_pages)
    except Exception:  # noqa: BLE001
        logger.exception("peer prefix: abandon failed for %s", getattr(req, "rid", "?"))
        req._peer_fetch = None


def _abandon_impl(queue, req, free_pages: bool) -> None:
    scheduler = queue.scheduler
    manager = queue.kv_manager
    fetch = getattr(req, "_peer_fetch", None)
    if fetch is None:
        return
    _unlock(scheduler.tree_cache, fetch)
    if fetch.new_indices is not None:
        if fetch.reading and not manager.peer_fetch_reads_settled(fetch.room):
            # A read may still land in these pages. Keep them out of the
            # allocator until Raiden settles it.
            queue.__dict__.setdefault("_peer_draining", []).append(fetch)
            manager.peer_fetch_end(fetch.room, notify=True)
            req._peer_fetch = None
            return
        if free_pages:
            _release_pages(scheduler.token_to_kv_pool_allocator, fetch)
    manager.peer_fetch_end(fetch.room, notify=True)
    req._peer_fetch = None


def _reap_draining(queue) -> None:
    draining = queue.__dict__.get("_peer_draining")
    if not draining:
        return
    manager = queue.kv_manager
    allocator = queue.scheduler.token_to_kv_pool_allocator
    keep = []
    for fetch in draining:
        # peer_fetch_end already dropped the fetch; settle by req id.
        settled = manager.peer_fetch_reads_settled(fetch.room)
        if settled:
            _release_pages(allocator, fetch)
        else:
            keep.append(fetch)
    queue._peer_draining = keep


# --------------------------------------------------------------------------- #
# Demotion: prefill host tier -> decode host tier under pressure
# --------------------------------------------------------------------------- #
# Trees whose host allocation failed since the last tick (the tree dropped
# pages itself): the sweep checks its watermark at once instead of on its
# period, as a failed AllocateBlockIds nudges vllm-torchtpu's sweep.
_DEMOTE_NUDGE: set = set()


def _evict_host(upstream: Callable, cache, *args, **kwargs):
    """Wraps ``UnifiedRadixCache.evict_host``: the drop still happens here,
    the sweep is told the pool is under pressure."""
    _DEMOTE_NUDGE.add(id(cache))
    return upstream(cache, *args, **kwargs)


def _host_anchor(tree):
    """The anchor host pool of a tree's HiCache, or None without one."""
    controller = getattr(tree, "cache_controller", None)
    pool = getattr(controller, "mem_pool_host", None)
    if pool is None:
        return None
    entry = getattr(pool, "anchor_entry", None)
    if entry is not None:
        return entry.host_pool
    return pool


def _sweep_context(scheduler) -> Optional[SweepContext]:
    ctx = getattr(scheduler, "_peer_demote_ctx", None)
    if ctx is not None:
        return ctx
    tree = scheduler.tree_cache
    host_pool = _host_anchor(tree)
    if host_pool is None or getattr(tree, "tree_core", None) is None:
        if not getattr(scheduler, "_peer_demote_warned", False):
            scheduler._peer_demote_warned = True
            logger.warning(
                "peer demote: no unified HiCache host pool on this engine; demotion off"
            )
        return None
    group = getattr(scheduler, "attn_tp_cpu_group", None)
    ctx = SweepContext(
        tree_cache=tree,
        host_pool=host_pool,
        page_size=scheduler.token_to_kv_pool_allocator.page_size,
        low_watermark=_demote_low(),
        high_watermark=_demote_high(),
        batch_pages=_demote_batch_pages(),
        period_s=_demote_period_s(),
        timeout_s=_demote_timeout_s(),
        parallelism=max(1, int(getattr(scheduler, "_peer_demote_parallelism", 1))),
        reduce_min=lambda vals: _all_reduce(vals, group, dist.ReduceOp.MIN),
        reduce_max=lambda vals: _all_reduce(vals, group, dist.ReduceOp.MAX),
    )
    scheduler._peer_demote_ctx = ctx
    return ctx


def _landing_context(scheduler) -> Optional[LandingContext]:
    ctx = getattr(scheduler, "_peer_demote_land_ctx", None)
    if ctx is not None:
        return ctx
    tree = scheduler.tree_cache
    host_pool = _host_anchor(tree)
    if host_pool is None or getattr(tree, "tree_core", None) is None:
        return None
    ctx = LandingContext(
        tree_cache=tree,
        host_pool=host_pool,
        page_size=scheduler.token_to_kv_pool_allocator.page_size,
        lease_timeout_s=2 * _demote_timeout_s() + 60.0,
        gather=_make_gather(getattr(scheduler, "attn_tp_cpu_group", None)),
    )
    scheduler._peer_demote_land_ctx = ctx
    return ctx


def _demote_tick(queue) -> None:
    """One sweep step on the prefill, after the bootstrap queue's lockstep
    work of this iteration. Must not raise: the step reduces across ranks."""
    if not demote_enabled():
        return
    manager = getattr(queue, "kv_manager", None)
    tick = getattr(manager, "peer_demote_tick", None)
    if tick is None:
        return
    scheduler = queue.scheduler
    if scheduler.disaggregation_mode != DisaggregationMode.PREFILL:
        return
    ctx = _sweep_context(scheduler)
    if ctx is None:
        return
    if id(scheduler.tree_cache) in _DEMOTE_NUDGE:
        _DEMOTE_NUDGE.discard(id(scheduler.tree_cache))
        manager.peer_demote_nudge()
    try:
        tick(ctx)
    except Exception:  # noqa: BLE001
        logger.exception("peer demote: tick failed")


# --------------------------------------------------------------------------- #
# Installation
# --------------------------------------------------------------------------- #
_UPSTREAM_ABORT: Optional[Callable] = None


def _abort_request(upstream: Callable, scheduler, recv_req):
    """Wraps ``Scheduler.abort_request`` on the prefill. Requests the gate
    holds outside the waiting queue must still be reachable by an abort, or
    an aborted request is computed later and its transfer waits for a
    decode that gave the room up."""
    queue = getattr(scheduler, "disagg_prefill_bootstrap_queue", None)
    state = getattr(queue, "__dict__", None) if queue is not None else None
    if state is None or scheduler.disaggregation_mode != DisaggregationMode.PREFILL:
        return upstream(scheduler, recv_req)

    def matches(req) -> bool:
        return bool(getattr(recv_req, "abort_all", False)) or req.rid.startswith(
            recv_req.rid
        )

    # A pull in flight: flag it; the next tick abandons the pull and hands
    # the request to this same handler through the waiting queue.
    for req in state.get("_peer_held") or []:
        if not getattr(req, "_peer_offer", False) and matches(req):
            req._peer_abort_req = recv_req
    # Deferred requests: let upstream find them in the waiting queue, then
    # put the survivors back, in order.
    deferred = state.get("_peer_deferred") or []
    exposed = list(deferred)
    scheduler.waiting_queue.extend(exposed)
    try:
        return upstream(scheduler, recv_req)
    finally:
        survivors = [r for r in deferred if r in scheduler.waiting_queue]
        scheduler.waiting_queue = [
            r for r in scheduler.waiting_queue if r not in exposed
        ]
        deferred[:] = survivors


def _wrap(owner: Any, name: str, impl: Callable) -> None:
    upstream = getattr(owner, name)
    if getattr(upstream, "_peer_prefix_wrapped", False):
        return

    @functools.wraps(upstream)
    def wrapper(*args, **kwargs):
        return impl(upstream, *args, **kwargs)

    wrapper._peer_prefix_wrapped = True
    setattr(owner, name, wrapper)


def install() -> None:
    if not enabled():
        return
    from sglang.srt.disaggregation import decode as decode_mod
    from sglang.srt.disaggregation import prefill as prefill_mod
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_components.invariant_checker import (
        SchedulerInvariantChecker,
    )

    _wrap(
        decode_mod.SchedulerDisaggregationDecodeMixin,
        "process_decode_queue",
        _process_decode_queue,
    )
    _wrap(prefill_mod.PrefillBootstrapQueue, "pop_bootstrapped", _pop_bootstrapped)
    _wrap(Scheduler, "_add_request_to_queue", _add_request_to_queue)
    global _UPSTREAM_ABORT
    if not getattr(Scheduler.abort_request, "_peer_prefix_wrapped", False):
        _UPSTREAM_ABORT = Scheduler.abort_request
    _wrap(Scheduler, "abort_request", _abort_request)

    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    _wrap(UnifiedRadixCache, "checkpoint", _checkpoint)
    if demote_enabled():
        _wrap(UnifiedRadixCache, "evict_host", _evict_host)
    _wrap(SchedulerInvariantChecker, "_check_full_pool", _check_full_pool)
    _wrap(SchedulerInvariantChecker, "_check_swa_pool", _check_swa_pool)
    logger.info(
        "peer prefix pull installed (timeout %.0fs, demotion %s)",
        _timeout_s(),
        "on" if demote_enabled() else "off",
    )
