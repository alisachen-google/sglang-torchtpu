"""CPU tests for the reverse prefix pull (Plan C): the manager mixin's producer
and consumer state machines with fake engines and a fake radix tree, and the
prefill scheduler hook that holds, pulls, inserts and releases a request."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu import pd_peer_prefix as hook
from sglang_tpu.disaggregation.raiden import peer_prefix as pp
from sglang_tpu.disaggregation.raiden.serde import (
    PrefixBlocks,
    PrefixRelease,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE = 4


class FakeEngine:
    def __init__(self):
        self.registered = []
        self.reads = []
        self.fail_start = False

    def register_read(self, req_id, uuid, blocks):
        self.registered.append((req_id, uuid, list(blocks)))
        return bool(blocks)

    def start_read(self, req_id, uuid, endpoint, remote, local, parallelism=1):
        if self.fail_start:
            raise RuntimeError("no staging")
        assert len(remote) == len(local)
        self.reads.append((req_id, uuid, endpoint, list(remote), list(local)))


class FakeLock:
    def __init__(self, node):
        self.node = node

    def to_dec_params(self):
        return ("dec", self.node)


class FakeTree:
    """Holds one prefix as device-resident tokens plus an optional host tail."""

    def __init__(self, device_tokens, host_tokens=0, page=PAGE):
        self.device_tokens = device_tokens
        self.host_tokens = host_tokens
        self.page = page
        self.locks = []
        self.unlocks = []
        self.inserts = []
        self.load_back_slot = True
        self.event_done = True
        self.is_swa_enabled = False
        self.swa_missing = []
        self.tree_core = SimpleNamespace(
            is_eagle=False,
            swa_tombstone_ranges=lambda key, start, end: list(self.swa_missing),
        )

    def swa_reprefill_tail_tokens(self):
        return 0

    def match_prefix(self, params):
        n = min(len(params.key.token_ids), self.device_tokens)
        n = (n // self.page) * self.page
        host = min(self.host_tokens, len(params.key.token_ids) - n)
        host = (host // self.page) * self.page
        return SimpleNamespace(
            device_indices=torch.arange(1000, 1000 + n, dtype=torch.int64),
            host_hit_length=host,
            last_device_node=("dev", n),
            best_match_node=("host", n + host),
            swa_branching_seqlen=None,
        )

    def inc_lock_ref(self, node):
        self.locks.append(node)
        return FakeLock(node)

    def dec_lock_ref(self, node, params):
        self.unlocks.append((node, params))

    def has_free_load_back_slot(self):
        return self.load_back_slot

    def init_load_back(self, params):
        n = params.host_hit_length
        return torch.arange(5000, 5000 + n, dtype=torch.int64), ("restored", n)

    def ready_to_load_host_cache(self):
        return 7

    def is_load_back_event_done(self, idx):
        return self.event_done

    def insert(self, params):
        self.inserts.append(params)
        return SimpleNamespace(
            prefix_len=len(params.key.token_ids), rotation_tail_declined=False
        )


class StubManager(pp.PeerPrefixMixin):
    """Just enough of RaidenKVManager for the mixin."""

    def __init__(self, name):
        self.name = name
        self._message_handlers = {}
        self._state_lock = threading.Lock()
        self._completed_recv = set()
        self._completed_send = set()
        self._send_done_at = {}
        self._failed = set()
        self._native_failed = set()
        self.local_ip = "10.0.0.1"
        self.rank_port = 31000
        self.raiden_endpoint = f"{name}-raiden"
        self.swa_raiden_endpoint = ""
        self._transfer_parallelism = 1
        self.engine = FakeEngine()
        self._swa_engine = None
        self.sent = []
        self.peer = None
        self.drained = 0
        self._peer_init()

    def _ensure_engine(self):
        return self.engine

    def _send_multipart_locked(self, endpoint, parts, is_ipv6=False):
        self.sent.append((endpoint, parts))
        if self.peer is not None:
            self.peer._message_handlers[bytes(parts[0])](parts)

    def _drain_engine(self):
        self.drained += 1

    def _on_pull_req(self, msg):  # the real one lives in the sender mixin
        pass


def pair():
    prefill, decode = StubManager("prefill"), StubManager("decode")
    prefill.peer, decode.peer = decode, prefill
    prefill._peer_decode_endpoint = "tcp://10.0.0.2:31000"
    return prefill, decode


def ctx(tree, **kw):
    kwargs = dict(tree_cache=tree, allocator=None, page_size=PAGE, min_hit_tokens=PAGE)
    kwargs.update(kw)
    return pp.ProducerContext(**kwargs)


def test_serve_registers_locks_and_releases_on_notice():
    prefill, decode = pair()
    tree = FakeTree(device_tokens=16)
    tokens = list(range(20))
    assert prefill.peer_fetch_begin(5, tokens, 0, None, None)
    decode.peer_prefix_serve(ctx(tree))
    fetch = prefill.peer_fetch(5)
    assert fetch.stage == pp.STAGE_REPLIED and fetch.hit_len == 16
    assert fetch.blocks == [250, 251, 252, 253] and fetch.uuid == pp.peer_uuid(5)
    assert decode.engine.registered == [
        ("peer:5", pp.peer_uuid(5), [250, 251, 252, 253])
    ]
    assert tree.locks == [("dev", 16)] and tree.unlocks == []
    assert prefill.peer_fetch_start_read(5, [10, 11, 12, 13], [])
    assert prefill.engine.reads[0][2] == "decode-raiden"
    assert prefill.peer_fetch_poll(5, 10) == pp.STAGE_READING
    prefill._completed_recv.add("peer:5")
    assert prefill.peer_fetch_poll(5, 10) == pp.STAGE_DONE
    prefill.peer_fetch_end(5)
    assert "peer:5" not in prefill._completed_recv
    decode.peer_prefix_serve(ctx(tree))
    assert tree.unlocks == [(("dev", 16), ("dec", ("dev", 16)))]
    assert decode.peer_stats()["serving"] == 0


def test_short_hit_is_a_miss_without_locks():
    prefill, decode = pair()
    tree = FakeTree(device_tokens=4)
    prefill.peer_fetch_begin(1, list(range(20)), 4, None, None)
    decode.peer_prefix_serve(ctx(tree))
    assert prefill.peer_fetch(1).stage == pp.STAGE_REPLIED
    assert prefill.peer_fetch(1).hit_len == 0
    assert tree.locks == [] and decode.engine.registered == []


def test_host_part_is_restored_before_register():
    prefill, decode = pair()
    tree = FakeTree(device_tokens=8, host_tokens=8)
    tree.event_done = False
    prefill.peer_fetch_begin(2, list(range(20)), 0, None, None)
    decode.peer_prefix_serve(ctx(tree))
    assert prefill.peer_fetch(2).stage == pp.STAGE_SENT  # waiting for the H2D
    assert tree.locks == [("dev", 8), ("restored", 8)]
    tree.event_done = True
    decode.peer_prefix_serve(ctx(tree))
    fetch = prefill.peer_fetch(2)
    assert fetch.hit_len == 16
    assert fetch.blocks == [250, 251, 1250, 1251]


def test_hybrid_serves_hbm_only_and_window_pages():
    prefill, decode = pair()
    swa = FakeEngine()
    decode._swa_engine = swa
    decode.swa_raiden_endpoint = "decode-swa"
    tree = FakeTree(device_tokens=16, host_tokens=8)
    mapping = {i: 9000 + i for i in range(1000, 1016)}
    allocator = SimpleNamespace(
        translate_loc_from_full_to_swa=lambda idx: torch.tensor(
            [mapping[int(i)] for i in idx]
        )
    )
    prefill.peer_fetch_begin(3, list(range(24)), 0, None, None)
    decode.peer_prefix_serve(
        ctx(
            tree,
            hybrid=True,
            sliding_window_size=8,
            allocator=allocator,
            restore_allowed=False,
        )
    )
    fetch = prefill.peer_fetch(3)
    assert fetch.hit_len == 16  # host part not served for hybrid models
    assert fetch.swa_blocks == [2502, 2503] and fetch.swa_endpoint == "decode-swa"
    assert swa.registered[0][0] == "peerswa:3"


def test_lease_released_when_raiden_reports_the_send():
    prefill, decode = pair()
    tree = FakeTree(device_tokens=16)
    prefill.peer_fetch_begin(6, list(range(20)), 0, None, None)
    decode.peer_prefix_serve(ctx(tree))
    decode._completed_send.add("peer:6")
    decode.peer_prefix_serve(ctx(tree))
    assert tree.unlocks and "peer:6" not in decode._completed_send


def test_start_read_failure_and_timeout_fail_closed():
    prefill, decode = pair()
    tree = FakeTree(device_tokens=16)
    prefill.peer_fetch_begin(7, list(range(20)), 0, None, None)
    decode.peer_prefix_serve(ctx(tree))
    prefill.engine.fail_start = True
    assert not prefill.peer_fetch_start_read(7, [1, 2, 3, 4], [])
    assert prefill.peer_fetch(7).stage == pp.STAGE_FAILED
    prefill.peer_fetch_end(7)
    prefill.peer_fetch_begin(8, list(range(20)), 0, None, None)
    prefill.peer_fetch(8).sent_at -= 1000
    assert prefill.peer_fetch_poll(8, 1) == pp.STAGE_FAILED


# --------------------------------------------------------------------------- #
# Scheduler hook (prefill), world size 1.
# --------------------------------------------------------------------------- #
class FakeAllocator:
    def __init__(self):
        self.page_size = PAGE
        self.freed = []
        self.next = 3000

    def alloc_extend(self, pl, plc, sl, slc, last_loc, n):
        out = torch.arange(self.next, self.next + n, dtype=torch.int32)
        self.next += n
        return out

    def free(self, idx):
        self.freed.append(idx.tolist())


def make_queue(tree, manager):
    scheduler = SimpleNamespace(
        tree_cache=tree,
        token_to_kv_pool_allocator=FakeAllocator(),
        attn_tp_cpu_group=None,
        sliding_window_size=None,
    )
    return SimpleNamespace(scheduler=scheduler, kv_manager=manager, __dict__={})


def make_req(room, n=20):
    return SimpleNamespace(
        rid=f"r{room}",
        bootstrap_room=room,
        origin_input_ids=list(range(n)),
        extra_key=None,
        cache_salt=None,
        # No _peer_fetch attribute, as on a real Req before any fetch.
    )


def test_hook_pulls_inserts_and_releases(monkeypatch):
    prefill, decode = pair()
    tree = FakeTree(device_tokens=16)
    tree.token_to_kv_pool_allocator = None  # marks a real pool-backed tree
    evictions = []
    monkeypatch.setattr(hook, "evict_from_tree_cache", lambda t, n: evictions.append(n))
    queue = make_queue(tree, prefill)
    req = make_req(11)
    fetch = hook._Fetch(room=11, matched_len=4, want_len=16)
    fetch.prefix_indices = torch.arange(1000, 1004, dtype=torch.int64)
    fetch.lock_node = ("dev", 4)
    fetch.lock_params = ("dec", ("dev", 4))
    req._peer_fetch = fetch
    prefill.peer_fetch_begin(11, req.origin_input_ids[:16], 4, None, None)
    held = [req]
    assert hook._advance_held(queue, held) == [] and held == [req]  # waiting
    decode.peer_prefix_serve(ctx(tree))
    assert hook._advance_held(queue, held) == []  # replied -> reading
    assert prefill.peer_fetch(11).stage == pp.STAGE_READING
    assert evictions == [12 + PAGE]  # room for the 12 pulled tokens, evicted first
    assert prefill.engine.reads[0][4] == [750, 751, 752]
    prefill._completed_recv.add("peer:11")
    out = hook._advance_held(queue, held)
    assert out == [req] and held == []
    params = tree.inserts[0]
    assert list(params.key.token_ids) == list(range(16))
    assert params.value.tolist() == [1000, 1001, 1002, 1003] + list(range(3000, 3012))
    assert params.prev_prefix_len == 4 and params.inserted_len == 4
    assert tree.unlocks == [(("dev", 4), ("dec", ("dev", 4)))]
    assert req._peer_fetch is None and prefill.peer_fetch(11) is None
    # The inserted prefix stays pinned until the scheduler takes the request.
    assert tree.locks[-1] == ("dev", 16) and req._peer_lock is not None
    queue.scheduler.waiting_queue = [req]
    hook._release_scheduled_locks(queue, [], [])
    assert len(tree.unlocks) == 1
    queue.scheduler.waiting_queue = []
    hook._release_scheduled_locks(queue, [], [])
    assert tree.unlocks[-1] == (("dev", 16), ("dec", ("dev", 16)))
    assert req._peer_lock is None and queue.__dict__["_peer_locked"] == []
    assert decode.sent[-1][1][0] == PrefixBlocks.TAG or any(
        p[1][0] == PrefixRelease.TAG for p in prefill.sent
    )


def test_hook_abandons_on_miss_and_frees_on_failure():
    prefill, decode = pair()
    tree = FakeTree(device_tokens=4)
    queue = make_queue(tree, prefill)
    req = make_req(12)
    fetch = hook._Fetch(room=12, matched_len=4, want_len=16)
    fetch.prefix_indices = torch.arange(1000, 1004, dtype=torch.int64)
    req._peer_fetch = fetch
    prefill.peer_fetch_begin(12, req.origin_input_ids[:16], 4, None, None)
    decode.peer_prefix_serve(ctx(tree))  # miss
    held = [req]
    assert hook._advance_held(queue, held) == [req]
    assert req._peer_fetch is None and tree.inserts == []
    # failure after the read started: pages drain until Raiden settles
    tree2 = FakeTree(device_tokens=16)
    queue2 = make_queue(tree2, prefill)
    req2 = make_req(13)
    f2 = hook._Fetch(room=13, matched_len=0, want_len=16)
    f2.prefix_indices = torch.empty((0,), dtype=torch.int64)
    req2._peer_fetch = f2
    prefill.peer_fetch_begin(13, req2.origin_input_ids[:16], 0, None, None)
    decode.peer_prefix_serve(ctx(tree2))
    held = [req2]
    hook._advance_held(queue2, held)  # reading
    prefill._failed.add("peer:13")
    assert hook._advance_held(queue2, held) == [req2]
    assert queue2.scheduler.token_to_kv_pool_allocator.freed == [
        list(range(3000, 3016))
    ]


def test_decode_hook_serves_through_the_scheduler(monkeypatch):
    """The decode wrapper builds its context from the scheduler and calls serve."""
    prefill, decode = pair()
    tree = FakeTree(device_tokens=16)
    scheduler = SimpleNamespace(
        tree_cache=tree,
        token_to_kv_pool_allocator=SimpleNamespace(page_size=PAGE),
        sliding_window_size=None,
        enable_decode_hicache=True,
        disagg_decode_prealloc_queue=SimpleNamespace(kv_manager=decode),
    )
    monkeypatch.setattr(
        hook,
        "get_disagg",
        lambda: SimpleNamespace(disaggregation_decode_enable_radix_cache=True),
    )
    monkeypatch.setattr(hook, "_min_tokens", lambda page: PAGE)
    calls = []
    hook._process_decode_queue(lambda s: calls.append(s), scheduler)
    assert calls == [scheduler]
    prefill.peer_fetch_begin(21, list(range(20)), 0, None, None)
    hook._process_decode_queue(lambda s: None, scheduler)
    assert prefill.peer_fetch(21).hit_len == 16
    assert scheduler._peer_prefix_ctx.restore_allowed is True


def test_inflight_pages_count_as_uncached_until_committed():
    prefill, decode = pair()
    tree = FakeTree(device_tokens=16)
    queue = make_queue(tree, prefill)
    req = make_req(31)
    fetch = hook._Fetch(room=31, matched_len=0, want_len=16)
    fetch.prefix_indices = torch.empty((0,), dtype=torch.int64)
    req._peer_fetch = fetch
    prefill.peer_fetch_begin(31, req.origin_input_ids[:16], 0, None, None)
    decode.peer_prefix_serve(ctx(tree))
    held = [req]
    hook._advance_held(queue, held)  # replied -> reading: pages allocated
    seen = {}

    def upstream(checker, ps, uncached=0):
        seen["uncached"] = uncached
        return False, ""

    hook._check_full_pool(upstream, object(), None, uncached=5)
    assert seen["uncached"] == 5 + 16
    prefill._completed_recv.add("peer:31")
    hook._advance_held(queue, held)  # committed: the tree owns them
    hook._check_full_pool(upstream, object(), None)
    assert seen["uncached"] == 0


def test_mismatched_reply_abandons_and_frees_before_any_read():
    """A decode reply whose page count differs from the agreed hit is refused
    on this rank; the reduced decision then abandons on every rank."""
    prefill, decode = pair()
    tree = FakeTree(device_tokens=16)
    queue = make_queue(tree, prefill)
    req = make_req(41)
    fetch = hook._Fetch(room=41, matched_len=0, want_len=16)
    fetch.prefix_indices = torch.empty((0,), dtype=torch.int64)
    req._peer_fetch = fetch
    prefill.peer_fetch_begin(41, req.origin_input_ids[:16], 0, None, None)
    decode.peer_prefix_serve(ctx(tree))
    peer = prefill.peer_fetch(41)
    peer.blocks = peer.blocks[:-1]  # the decode on this rank answered one page short
    held = [req]
    assert hook._advance_held(queue, held) == [req]
    assert prefill.engine.reads == [] and req._peer_fetch is None
    # The mismatch is seen before any page is allocated: nothing to free.
    assert queue.scheduler.token_to_kv_pool_allocator.freed == []
    assert hook._inflight_tokens() == 0


def test_commit_failure_releases_without_raising():
    prefill, decode = pair()
    tree = FakeTree(device_tokens=16)
    tree.insert = lambda params: (_ for _ in ()).throw(RuntimeError("tree says no"))
    queue = make_queue(tree, prefill)
    req = make_req(42)
    fetch = hook._Fetch(room=42, matched_len=0, want_len=16)
    fetch.prefix_indices = torch.empty((0,), dtype=torch.int64)
    req._peer_fetch = fetch
    prefill.peer_fetch_begin(42, req.origin_input_ids[:16], 0, None, None)
    decode.peer_prefix_serve(ctx(tree))
    held = [req]
    hook._advance_held(queue, held)  # reading
    prefill._completed_recv.add("peer:42")
    assert hook._advance_held(queue, held) == [req]
    assert req._peer_fetch is None and hook._inflight_tokens() == 0


def test_read_budget_holds_a_second_pull_until_the_first_settles():
    prefill, decode = pair()
    prefill._full_engine_max_blocks = 4  # budget = 4 pages: one 16-token pull at a time
    tree = FakeTree(device_tokens=16)
    queue = make_queue(tree, prefill)
    reqs = []
    for room in (51, 52):
        req = make_req(room)
        fetch = hook._Fetch(room=room, matched_len=0, want_len=16)
        fetch.prefix_indices = torch.empty((0,), dtype=torch.int64)
        req._peer_fetch = fetch
        prefill.peer_fetch_begin(room, req.origin_input_ids[:16], 0, None, None)
        reqs.append(req)
    decode.peer_prefix_serve(ctx(tree))
    held = list(reqs)
    assert hook._advance_held(queue, held) == []
    stages = [prefill.peer_fetch(r._peer_fetch.room).stage for r in reqs]
    assert stages == [pp.STAGE_READING, pp.STAGE_REPLIED]  # the second waits
    assert prefill.peer_inflight_pages() == 4
    prefill._completed_recv.add("peer:51")
    out = hook._advance_held(queue, held)
    assert out == [reqs[0]]
    assert (
        prefill.peer_fetch(52).stage == pp.STAGE_READING
    )  # admitted after the first settled


def test_hybrid_restore_serves_window_when_present_and_misses_when_tombstoned():
    prefill, decode = pair()
    swa = FakeEngine()
    decode._swa_engine = swa
    decode.swa_raiden_endpoint = "decode-swa"
    tree = FakeTree(device_tokens=8, host_tokens=8)
    mapping = {i: 9000 + i for i in list(range(1000, 1008)) + list(range(5000, 5008))}
    allocator = SimpleNamespace(
        translate_loc_from_full_to_swa=lambda idx: torch.tensor(
            [mapping[int(i)] for i in idx]
        )
    )
    c = ctx(
        tree,
        hybrid=True,
        sliding_window_size=8,
        allocator=allocator,
        restore_allowed=True,
    )
    prefill.peer_fetch_begin(61, list(range(24)), 0, None, None)
    decode.peer_prefix_serve(c)
    fetch = prefill.peer_fetch(61)
    assert fetch.hit_len == 16 and fetch.blocks == [250, 251, 1250, 1251]
    assert fetch.swa_blocks == [
        3500,
        3501,
    ]  # the window sits in the restored part (5000.. -> 14000..)
    assert tree.locks == [("dev", 8), ("restored", 8)]
    prefill.peer_fetch_end(61)
    decode.peer_prefix_serve(c)
    tree.swa_missing = [(8, 16)]
    prefill.peer_fetch_begin(62, list(range(24)), 0, None, None)
    decode.peer_prefix_serve(c)
    assert prefill.peer_fetch(62).hit_len == 0
    assert prefill.peer_fetch(62).error == "SWA window missing"
    assert decode.peer_stats()["serving"] == 0


def test_decode_offer_is_pulled_speculatively_and_inserted(monkeypatch):
    monkeypatch.setattr(hook, "_min_tokens", lambda page: PAGE)
    prefill, decode = pair()
    decode._peer_prefill_endpoint = "tcp://10.0.0.1:31000"
    tree = FakeTree(device_tokens=16)  # the decode holds the prefix
    queue = make_queue(FakeTree(device_tokens=0), prefill)  # the prefill holds nothing
    queue.scheduler.token_to_kv_pool_allocator.size = 1000

    def available_size():
        return 900

    queue.scheduler.token_to_kv_pool_allocator.available_size = available_size
    # the decode finished a request: offer its prefix
    assert decode.peer_offer(71, list(range(16)), None, None)
    assert prefill.peer_peek_offer_room() == 71
    held = []
    hook._admit_offer(queue, held)  # rank 0, world size 1: admit at once
    assert len(held) == 1 and held[0]._peer_offer and held[0]._peer_fetch.room == 71
    assert prefill.peer_fetch(71).stage == pp.STAGE_SENT
    decode.peer_prefix_serve(ctx(tree))  # the decode serves its own offer
    assert hook._advance_held(queue, held) == []  # reading
    prefill._completed_recv.add("peer:71")
    out = hook._advance_held(queue, held)
    assert [r.rid for r in out] == ["offer:71"] and held == []
    assert list(queue.scheduler.tree_cache.inserts[0].key.token_ids) == list(range(16))
    assert prefill.peer_fetch(71) is None


def test_offer_is_declined_without_headroom(monkeypatch):
    prefill, decode = pair()
    tree = FakeTree(device_tokens=16)
    queue = make_queue(tree, prefill)
    queue.scheduler.token_to_kv_pool_allocator.size = 1000
    queue.scheduler.token_to_kv_pool_allocator.available_size = lambda: 100
    prefill._peer_offers.append(
        SimpleNamespace(
            room=72, token_ids=list(range(16)), extra_key=None, cache_salt=None
        )
    )
    held = []
    hook._admit_offer(queue, held)
    assert held == [] and prefill.peer_fetch(72) is None


def test_decode_finished_request_triggers_an_offer(monkeypatch):
    monkeypatch.setattr(hook, "_min_tokens", lambda page: PAGE)
    prefill, decode = pair()
    decode._peer_prefill_endpoint = "tcp://10.0.0.1:31000"
    cache = SimpleNamespace(page_size=PAGE)
    hook._OFFER_MANAGERS[id(cache)] = decode

    def req_ns(room, finished):
        return SimpleNamespace(
            rid=f"r{room}",
            bootstrap_room=room,
            origin_input_ids=list(range(12)),
            output_ids=[7, 8, 9, 10],
            extra_key=None,
            cache_salt=None,
            is_retracted=False,
            finished=lambda: finished,
        )

    calls = []
    req = req_ns(73, True)
    hook._checkpoint(
        lambda c, r, up_to=None: calls.append((r, up_to)), cache, req, up_to=16
    )
    assert calls == [(req, 16)]
    offer = prefill.peer_take_offer()
    assert offer is not None and offer.room == 73 and len(offer.token_ids) == 16
    # an unfinished checkpoint offers nothing
    hook._checkpoint(lambda c, r, up_to=None: None, cache, req_ns(74, False), up_to=12)
    assert prefill.peer_take_offer() is None


def test_pull_larger_than_budget_is_admitted_when_nothing_is_in_flight():
    prefill, decode = pair()
    prefill._full_engine_max_blocks = 2  # budget smaller than one 16-token pull
    tree = FakeTree(device_tokens=16)
    queue = make_queue(tree, prefill)
    req = make_req(81)
    fetch = hook._Fetch(room=81, matched_len=0, want_len=16)
    fetch.prefix_indices = torch.empty((0,), dtype=torch.int64)
    req._peer_fetch = fetch
    prefill.peer_fetch_begin(81, req.origin_input_ids[:16], 0, None, None)
    decode.peer_prefix_serve(ctx(tree))
    held = [req]
    hook._advance_held(queue, held)
    assert prefill.peer_fetch(81).stage == pp.STAGE_READING


def test_install_wraps_every_target(monkeypatch):
    """The hook wraps real classes; a wrong target name must fail here, not at engine start."""
    monkeypatch.setenv("SGLANG_TPU_PEER_PREFIX", "1")
    hook.install()
    from sglang.srt.disaggregation import decode as decode_mod
    from sglang.srt.disaggregation import prefill as prefill_mod
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_components.invariant_checker import (
        SchedulerInvariantChecker,
    )
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    for owner, name in (
        (decode_mod.SchedulerDisaggregationDecodeMixin, "process_decode_queue"),
        (prefill_mod.PrefillBootstrapQueue, "pop_bootstrapped"),
        (Scheduler, "_add_request_to_queue"),
        (Scheduler, "abort_request"),
        (UnifiedRadixCache, "checkpoint"),
        (SchedulerInvariantChecker, "_check_full_pool"),
        (SchedulerInvariantChecker, "_check_swa_pool"),
    ):
        assert getattr(getattr(owner, name), "_peer_prefix_wrapped", False), name


# ---------------------------------------------------------------------------
# Step 4: schedule-time pulls served from the decode's own restore.
# ---------------------------------------------------------------------------


def make_decode_request(room, l1=8, l2=8, status="pending"):
    """A fake DecodeRequest as the decode's prealloc/transfer queues hold it."""
    pm = SimpleNamespace(
        l1_prefix_len=l1,
        l2_host_hit_length=l2,
        needs_local_restore=l2 > 0,
        prefix_indices=torch.arange(1000, 1000 + l1, dtype=torch.int64),
        last_device_node=("dev", l1),
    )
    return SimpleNamespace(
        req=SimpleNamespace(bootstrap_room=room, last_node=("dev", l1)),
        prefix_match=pm,
        hicache_restore_status=SimpleNamespace(value=status),
        hicache_restored_kv_indices=None,
        hicache_restored_node=None,
        hicache_load_consumer_index=-1,
    )


def test_serve_rides_the_decodes_own_restore_for_a_preallocated_room():
    prefill, decode = pair()
    tree = FakeTree(device_tokens=0)  # the tree path would find nothing
    tree.load_back_slot = False  # and could not restore either
    dr = make_decode_request(21)
    rooms = {21: dr}
    c = ctx(tree, lookup_room=rooms.get)
    assert prefill.peer_fetch_begin(21, list(range(20)), 0, None, None)
    decode.peer_prefix_serve(c)
    # Restore not queued yet by the decode: wait, L1 locked, no reply.
    assert prefill.peer_fetch(21).stage == pp.STAGE_SENT
    assert tree.locks == [("dev", 8)] and decode.engine.registered == []
    # The decode queues its load-back; its state machine has not seen the
    # DMA land yet (the event poll is a collective, so it is not asked here).
    dr.hicache_restored_kv_indices = torch.arange(5000, 5008, dtype=torch.int64)
    dr.hicache_restored_node = ("restored", 16)
    dr.hicache_load_consumer_index = 3
    decode.peer_prefix_serve(c)
    assert prefill.peer_fetch(21).stage == pp.STAGE_SENT
    # READY: served from L1 + the restored pages, both locked for the lease.
    dr.hicache_restore_status = SimpleNamespace(value="ready")
    decode.peer_prefix_serve(c)
    fetch = prefill.peer_fetch(21)
    assert fetch.stage == pp.STAGE_REPLIED and fetch.hit_len == 16
    assert fetch.blocks == [250, 251, 1250, 1251]
    assert tree.locks == [("dev", 8), ("restored", 16)]
    decode._peer_lease_release(c, 21)
    assert sorted(n for n, _ in tree.unlocks) == [("dev", 8), ("restored", 16)]


def test_piggyback_restore_failure_serves_hbm_part_or_misses():
    prefill, decode = pair()
    tree = FakeTree(device_tokens=0)
    dr = make_decode_request(22, l1=8, l2=8, status="failed")
    c = ctx(tree, lookup_room={22: dr}.get)
    prefill.peer_fetch_begin(22, list(range(20)), 0, None, None)
    decode.peer_prefix_serve(c)
    fetch = prefill.peer_fetch(22)
    assert fetch.stage == pp.STAGE_REPLIED and fetch.hit_len == 8
    assert fetch.blocks == [250, 251]
    # Nothing beyond what the prefill holds: a miss, lease released.
    prefill2, decode2 = pair()
    tree2 = FakeTree(device_tokens=0)
    dr2 = make_decode_request(23, l1=8, l2=8, status="failed")
    c2 = ctx(tree2, lookup_room={23: dr2}.get)
    prefill2.peer_fetch_begin(23, list(range(20)), 8, None, None)
    decode2.peer_prefix_serve(c2)
    assert prefill2.peer_fetch(23).hit_len == 0
    assert decode2.peer_stats()["serving"] == 0 and len(tree2.unlocks) == 1


def test_piggyback_misses_when_the_room_is_gone_before_the_restore_lands():
    prefill, decode = pair()
    tree = FakeTree(device_tokens=0)
    dr = make_decode_request(24)
    rooms = {24: dr}
    c = ctx(tree, lookup_room=rooms.get)
    prefill.peer_fetch_begin(24, list(range(20)), 0, None, None)
    decode.peer_prefix_serve(c)
    rooms.clear()  # the decode aborted the request
    decode.peer_prefix_serve(c)
    # An error reply fails the fetch closed; the prefill recomputes.
    assert prefill.peer_fetch(24).stage == pp.STAGE_FAILED
    assert prefill.peer_fetch(24).error == "room gone"
    assert decode.peer_stats()["serving"] == 0


def test_pop_decides_a_pull_when_the_prefix_was_lost_during_bootstrap(monkeypatch):
    prefill, decode = pair()
    tree = FakeTree(device_tokens=0)  # the prefill lost the prefix meanwhile
    queue = make_queue(tree, prefill)
    queue.scheduler.disaggregation_mode = DisaggregationMode.PREFILL
    queue.scheduler.disagg_prefill_bootstrap_queue = queue
    monkeypatch.setattr(hook, "_min_tokens", lambda page_size: PAGE)
    monkeypatch.setattr(
        hook,
        "_match_prefix_result",
        lambda tree, req, tokens: tree.match_prefix(
            SimpleNamespace(key=SimpleNamespace(token_ids=tokens))
        ),
    )
    req = make_req(31)
    req._compute_max_prefix_len = lambda n: n
    req.finished = lambda: False
    monkeypatch.setenv(hook.ENV_JIT, "0")
    out = hook._pop_bootstrapped(lambda q: [req], queue)
    assert out == [] and queue.__dict__["_peer_held"] == [req]
    fetch = req._peer_fetch
    assert fetch is not None and fetch.matched_len == 0 and fetch.want_len == 20
    assert prefill.peer_fetch(31).stage == pp.STAGE_SENT
    # The decode serves 16 tokens; the request is released once committed.
    decode.peer_prefix_serve(ctx(FakeTree(device_tokens=16)))
    assert hook._pop_bootstrapped(lambda q: [], queue) == []
    assert prefill.peer_fetch(31).stage == pp.STAGE_READING
    prefill._completed_recv.add("peer:31")
    assert hook._pop_bootstrapped(lambda q: [], queue) == [req]
    assert req._peer_fetch is None and len(tree.inserts[0].value) == 16


def test_inbox_is_served_only_when_every_rank_has_the_room():
    prefill, decode = pair()
    tree = FakeTree(device_tokens=16)
    views = []
    c = ctx(tree, gather=lambda local: views)
    prefill.peer_fetch_begin(41, list(range(20)), 0, None, None)
    # The other rank has not received room 41 yet: nothing is served.
    s41 = prefill.peer_fetch(41).seq
    views[:] = [(((41, s41),), ()), ((), ())]
    decode.peer_prefix_serve(c)
    assert prefill.peer_fetch(41).stage == pp.STAGE_SENT
    assert decode.peer_stats()["inbox"] == 1 and tree.locks == []
    # Both ranks have it: served, in room order.
    prefill.peer_fetch_begin(40, list(range(20)), 0, None, None)
    s40 = prefill.peer_fetch(40).seq
    views[:] = [(((40, s40), (41, s41)), ()), (((40, s40), (41, s41)), ())]
    decode.peer_prefix_serve(c)
    assert [r[0] for r in decode.engine.registered] == ["peer:40", "peer:41"]
    assert decode.peer_stats()["inbox"] == 0


def test_inbox_entry_missing_on_a_rank_times_out_with_a_miss():
    prefill, decode = pair()
    tree = FakeTree(device_tokens=16)
    c = ctx(tree, gather=lambda local: [local, ((), ())], lease_timeout_s=0.0)
    prefill.peer_fetch_begin(42, list(range(20)), 0, None, None)
    decode._peer_inbox_at[(42, prefill.peer_fetch(42).seq)] -= 1.0
    decode.peer_prefix_serve(c)
    assert prefill.peer_fetch(42).stage == pp.STAGE_FAILED
    assert prefill.peer_fetch(42).error == "inbox timeout"
    assert decode.peer_stats()["inbox"] == 0 and tree.locks == []


def test_popped_requests_are_released_just_in_time(monkeypatch):
    prefill, decode = pair()
    tree = FakeTree(device_tokens=20)  # prefix intact: no pull needed
    queue = make_queue(tree, prefill)
    queue.scheduler.disaggregation_mode = DisaggregationMode.PREFILL
    queue.scheduler.disagg_prefill_bootstrap_queue = queue
    queue.scheduler.waiting_queue = ["computing"]
    monkeypatch.setattr(hook, "_min_tokens", lambda page_size: PAGE)
    monkeypatch.setattr(
        hook,
        "_match_prefix_result",
        lambda tree, req, tokens: tree.match_prefix(
            SimpleNamespace(key=SimpleNamespace(token_ids=tokens))
        ),
    )
    a, b = make_req(51), make_req(52)
    for r in (a, b):
        r._compute_max_prefix_len = lambda n: n
        r.finished = lambda: False
    # No room: both wait outside the waiting queue, in order.
    assert hook._pop_bootstrapped(lambda q: [a, b], queue) == []
    assert queue.__dict__["_peer_deferred"] == [a, b]
    # One slot: the oldest is released; its prefix is present, so no pull.
    queue.scheduler.waiting_queue = []
    assert hook._pop_bootstrapped(lambda q: [], queue) == [a]
    assert getattr(a, "_peer_fetch", None) is None and prefill.peer_fetch(51) is None
    # The prefix of the next one is gone by the time it is released: pull.
    tree.device_tokens = 0
    assert hook._pop_bootstrapped(lambda q: [], queue) == []
    assert b._peer_fetch is not None and prefill.peer_fetch(52).stage == pp.STAGE_SENT
    assert (
        queue.__dict__["_peer_held"] == [b] and queue.__dict__["_peer_deferred"] == []
    )
    # A pull in flight occupies the slot: a newly popped request waits.
    c = make_req(53)
    c._compute_max_prefix_len = lambda n: n
    c.finished = lambda: False
    assert hook._pop_bootstrapped(lambda q: [c], queue) == []
    assert queue.__dict__["_peer_deferred"] == [c]


def test_arrival_fetch_that_misses_still_waits_at_the_gate(monkeypatch):
    prefill, decode = pair()
    tree = FakeTree(device_tokens=20)
    queue = make_queue(tree, prefill)
    queue.scheduler.disaggregation_mode = DisaggregationMode.PREFILL
    queue.scheduler.disagg_prefill_bootstrap_queue = queue
    queue.scheduler.waiting_queue = ["computing"]
    monkeypatch.setattr(hook, "_min_tokens", lambda page_size: PAGE)
    monkeypatch.setattr(
        hook,
        "_match_prefix_result",
        lambda tree, req, tokens: tree.match_prefix(
            SimpleNamespace(key=SimpleNamespace(token_ids=tokens))
        ),
    )
    req = make_req(61)
    req._compute_max_prefix_len = lambda n: n
    req.finished = lambda: False
    # Asked at arrival with a near-full local match; the decode says short hit.
    req._peer_fetch = hook._Fetch(room=61, matched_len=16, want_len=20)
    prefill.peer_fetch_begin(61, req.origin_input_ids, 16, None, None)
    decode.peer_prefix_serve(ctx(FakeTree(device_tokens=16)))
    assert hook._pop_bootstrapped(lambda q: [req], queue) == []
    assert req._peer_fetch is None  # the miss settled the fetch
    assert queue.__dict__["_peer_deferred"] == [req]  # but the gate holds it
    queue.scheduler.waiting_queue = []
    assert hook._pop_bootstrapped(lambda q: [], queue) == [req]


def test_a_prompt_the_decode_lacks_is_asked_once_at_the_gate_then_runs(monkeypatch):
    prefill, decode = pair()
    tree = FakeTree(device_tokens=0)  # nothing cached on the prefill
    queue = make_queue(tree, prefill)
    queue.scheduler.disaggregation_mode = DisaggregationMode.PREFILL
    queue.scheduler.disagg_prefill_bootstrap_queue = queue
    queue.scheduler.waiting_queue = []
    monkeypatch.setattr(hook, "_min_tokens", lambda page_size: PAGE)
    monkeypatch.setattr(
        hook,
        "_match_prefix_result",
        lambda tree, req, tokens: tree.match_prefix(
            SimpleNamespace(key=SimpleNamespace(token_ids=tokens))
        ),
    )
    req = make_req(71)
    req._compute_max_prefix_len = lambda n: n
    req.finished = lambda: False
    from sglang_tpu.disaggregation.raiden.serde import PrefixReq

    def asks():
        return sum(1 for _, frames in prefill.sent if frames[0] == PrefixReq.TAG)

    assert hook._pop_bootstrapped(lambda q: [req], queue) == []  # asked, held
    assert asks() == 1
    decode.peer_prefix_serve(ctx(FakeTree(device_tokens=0)))  # nothing there
    # The miss settles the fetch; the request runs and is not asked again.
    out = hook._pop_bootstrapped(lambda q: [], queue)
    assert out == [req] and req._peer_fetch is None
    for _ in range(3):
        assert hook._pop_bootstrapped(lambda q: [], queue) == []
    assert asks() == 1 and queue.__dict__["_peer_deferred"] == []


def test_stale_release_and_stale_reply_are_ignored():
    prefill, decode = pair()
    tree = FakeTree(device_tokens=16)
    c = ctx(tree)
    # First ask, served, then abandoned by the prefill: a release goes out.
    assert prefill.peer_fetch_begin(81, list(range(20)), 0, None, None)
    first_seq = prefill.peer_fetch(81).seq
    decode.peer_prefix_serve(c)
    assert prefill.peer_fetch(81).stage == pp.STAGE_REPLIED
    prefill.peer_fetch_end(81, notify=True)
    # Second ask for the same room arrives in the same decode tick as the
    # release of the first: the new serve must survive and be answered.
    assert prefill.peer_fetch_begin(81, list(range(20)), 0, None, None)
    assert prefill.peer_fetch(81).seq == first_seq + 1
    decode.peer_prefix_serve(c)
    assert prefill.peer_fetch(81).stage == pp.STAGE_REPLIED
    assert decode.peer_stats()["serving"] == 1
    # A late release carrying the first sequence does not end the new serve.
    decode._peer_lease_release(c, 81, seq=first_seq)
    assert decode.peer_stats()["serving"] == 1
    decode._peer_lease_release(c, 81, seq=first_seq + 1)
    assert decode.peer_stats()["serving"] == 0
    # A late reply for an earlier ask is ignored by the prefill.
    assert prefill.peer_fetch_begin(82, list(range(20)), 0, None, None)
    stale = PrefixBlocks(
        room=82, hit_len=16, uuid=1, blocks=[1], endpoint="x", seq=1
    ).to_frames()
    prefill._on_prefix_blocks(stale)
    assert prefill.peer_fetch(82).stage == pp.STAGE_SENT


def test_gate_pull_uses_the_short_timeout(monkeypatch):
    prefill, decode = pair()
    tree = FakeTree(device_tokens=0)
    queue = make_queue(tree, prefill)
    queue.scheduler.disaggregation_mode = DisaggregationMode.PREFILL
    queue.scheduler.disagg_prefill_bootstrap_queue = queue
    queue.scheduler.waiting_queue = []
    monkeypatch.setattr(hook, "_min_tokens", lambda page_size: PAGE)
    monkeypatch.setattr(
        hook,
        "_match_prefix_result",
        lambda tree, req, tokens: tree.match_prefix(
            SimpleNamespace(key=SimpleNamespace(token_ids=tokens))
        ),
    )
    monkeypatch.setenv(hook.ENV_GATE_TIMEOUT, "7")
    req = make_req(91)
    req._compute_max_prefix_len = lambda n: n
    req.finished = lambda: False
    assert hook._pop_bootstrapped(lambda q: [req], queue) == []
    assert req._peer_fetch.timeout == 7.0


def test_a_newer_ask_supersedes_an_older_one_waiting_in_the_inbox():
    prefill, decode = pair()
    tree = FakeTree(device_tokens=16)
    held_views = []
    c = ctx(tree, gather=lambda local: held_views)
    # First ask waits in the inbox (the other rank has not seen it).
    prefill.peer_fetch_begin(43, list(range(20)), 16, None, None)
    held_views[:] = [((), ()), ((), ())]
    decode.peer_prefix_serve(c)
    assert decode.peer_stats()["inbox"] == 1
    # The prefill gives up on it and asks again with a lower local match.
    prefill.peer_fetch_end(43, notify=True)
    prefill.peer_fetch_begin(43, list(range(20)), 0, None, None)
    assert decode.peer_stats()["inbox"] == 1  # the older ask is gone
    seq = prefill.peer_fetch(43).seq
    held_views[:] = [(((43, seq),), ()), (((43, seq),), ())]
    decode.peer_prefix_serve(c)
    fetch = prefill.peer_fetch(43)
    assert fetch.stage == pp.STAGE_REPLIED and fetch.hit_len == 16


class FakeAbortUpstream:
    """Upstream abort_request: pops matching requests from the waiting queue."""

    def __init__(self):
        self.aborted = []

    def __call__(self, scheduler, recv_req):
        keep = []
        for req in scheduler.waiting_queue:
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                self.aborted.append(req)
            else:
                keep.append(req)
        scheduler.waiting_queue = keep


def test_abort_reaches_a_deferred_request():
    prefill, decode = pair()
    tree = FakeTree(device_tokens=20)
    queue = make_queue(tree, prefill)
    queue.scheduler.disaggregation_mode = DisaggregationMode.PREFILL
    queue.scheduler.disagg_prefill_bootstrap_queue = queue
    computing = make_req(50)
    queue.scheduler.waiting_queue = [computing]
    a, b = make_req(52), make_req(53)
    queue.__dict__["_peer_deferred"] = [a, b]
    upstream = FakeAbortUpstream()
    hook._abort_request(
        upstream, queue.scheduler, SimpleNamespace(rid="r53", abort_all=False)
    )
    assert upstream.aborted == [b]
    assert queue.__dict__["_peer_deferred"] == [a]
    assert queue.scheduler.waiting_queue == [computing]


def test_abort_of_a_held_pull_abandons_it_and_aborts_the_request(monkeypatch):
    prefill, decode = pair()
    tree = FakeTree(device_tokens=0)
    queue = make_queue(tree, prefill)
    queue.scheduler.disaggregation_mode = DisaggregationMode.PREFILL
    queue.scheduler.disagg_prefill_bootstrap_queue = queue
    queue.scheduler.waiting_queue = []
    req = make_req(54)
    req._peer_fetch = hook._Fetch(room=54, matched_len=0, want_len=20)
    prefill.peer_fetch_begin(54, req.origin_input_ids, 0, None, None)
    queue.__dict__["_peer_held"] = [req]
    upstream = FakeAbortUpstream()
    monkeypatch.setattr(hook, "_UPSTREAM_ABORT", upstream)
    hook._abort_request(
        upstream, queue.scheduler, SimpleNamespace(rid="r54", abort_all=False)
    )
    assert req._peer_abort_req is not None  # flagged, pull still in flight
    out = hook._pop_bootstrapped(lambda q: [], queue)
    assert out == [] and upstream.aborted == [req]
    assert req._peer_fetch is None and prefill.peer_fetch(54) is None
    assert queue.__dict__["_peer_held"] == [] and queue.scheduler.waiting_queue == []
