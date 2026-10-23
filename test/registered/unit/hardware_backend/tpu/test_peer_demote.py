"""CPU tests for pressure-driven demotion (prefill host tier -> decode host
tier): the manager mixin's sweep and landing state machines against fake host
pools, a fake unified tree core and a fake Raiden transport, plus the
scheduler hooks and the wire messages."""

from __future__ import annotations

import threading
import time
from array import array
from collections import deque
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu import pd_peer_prefix as hook
from sglang_tpu.disaggregation.raiden import peer_demote as pd
from sglang_tpu.disaggregation.raiden import peer_prefix as pp
from sglang_tpu.disaggregation.raiden.serde import (
    DemoteAck,
    DemoteDone,
    DemoteNode,
    DemoteOffer,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE = 4
FULL = ComponentType.FULL
N_COMPONENTS = len(ComponentType)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeMgr:
    """The host pool's Raiden manager: records explicit-destination pushes."""

    def __init__(self, name):
        self.transfer_address = f"{name}:7000"
        self.pushes = []
        self.fail = False

    def push_registered_plan(self, uuid, peer, src, dst, layer_idx=-1, parallelism=1):
        if self.fail:
            raise RuntimeError("peer unreachable")
        assert len(src) == len(dst)
        self.pushes.append((uuid, peer, list(src), list(dst), layer_idx))


class FakeHostPool:
    """Page-granular host pool with the TpuRaidenHostKVCache surface used."""

    def __init__(self, name, pages, page_size=PAGE):
        self.page_size = page_size
        self.size = pages * page_size
        self.free_pages = deque(range(pages))
        self.transfer = SimpleNamespace(_mgr=FakeMgr(name))

    def available_size(self):
        return len(self.free_pages) * self.page_size

    def alloc(self, need_size):
        n = need_size // self.page_size
        if n > len(self.free_pages):
            return None
        pages = [self.free_pages.popleft() for _ in range(n)]
        base = torch.tensor(pages, dtype=torch.int64).repeat_interleave(self.page_size)
        offs = torch.arange(self.page_size, dtype=torch.int64).repeat(n)
        return base * self.page_size + offs

    def free(self, indices):
        for p in torch.unique(indices // self.page_size).tolist():
            self.free_pages.append(int(p))
        return len(indices)

    def _to_page_ids(self, indices):
        return (indices[:: self.page_size] // self.page_size).tolist()


class FakeNode:
    counter = 0

    def __init__(self, key=None, host_value=None, value=None):
        self.id = FakeNode.counter
        FakeNode.counter += 1
        self.key = key
        self.parent = None
        self.children = {}
        self.component_data = [
            SimpleNamespace(host_value=None, value=None, host_lock_ref=0, lock_ref=0)
            for _ in range(N_COMPONENTS)
        ]
        self.component_data[FULL].host_value = host_value
        self.component_data[FULL].value = value
        self.write_through_pending_id = None
        self.load_back_pending_id = None
        self.hash_value = None

    @property
    def evicted(self):
        return self.parent is not None and self.component_data[FULL].value is None

    def get_last_hash_value(self):
        return None


class FakeLRU:
    """MRU at the head, LRU at the tail, like UnifiedLRUList's host list."""

    def __init__(self, core):
        self.core = core

    def _nodes(self):
        out = []
        stack = list(self.core.root_node.children.values())
        while stack:
            n = stack.pop()
            if n.component_data[FULL].host_value is not None:
                out.append(n)
            stack.extend(n.children.values())
        # Oldest = deepest, as a host LRU after a chain was evicted leaf first.
        out.sort(key=lambda n: n.id)
        return out

    def get_lru_no_host_lock(self):
        for n in reversed(self._nodes()):
            if n.component_data[FULL].host_lock_ref == 0:
                return n
        return None

    def get_prev_no_host_lock(self, node):
        nodes = self._nodes()
        i = nodes.index(node) - 1
        while i >= 0:
            if nodes[i].component_data[FULL].host_lock_ref == 0:
                return nodes[i]
            i -= 1
        return None


class FakeCore:
    def __init__(self, page_size=PAGE):
        self.page_size = page_size
        self.root_node = FakeNode()
        self.is_eagle = False
        self.is_write_back = False
        self.host_lru_lists = {FULL: FakeLRU(self)}
        self.nodes = {self.root_node.id: self.root_node}
        self.evicted = []
        self.inserted = []

    def add(self, parent, tokens, host=True, device=False, extra_key=None):
        key = RadixKey(array("q", tokens), extra_key)
        n = len(tokens)
        node = FakeNode(
            key,
            host_value=torch.arange(
                100 * (len(self.nodes) + 1), 100 * (len(self.nodes) + 1) + n
            )
            if host
            else None,
            value=torch.arange(n) if device else None,
        )
        node.parent = parent
        parent.children[key.child_key(self.page_size)] = node
        self.nodes[node.id] = node
        return node

    def node_by_id(self, node_id):
        return self.nodes[node_id]

    def _is_host_leaf(self, node):
        if node is self.root_node or not node.evicted:
            return False
        cd = node.component_data[FULL]
        if cd.host_value is None or cd.host_lock_ref > 0 or cd.lock_ref > 0:
            return False
        return len(node.children) == 0

    def _evict_host_leaf(self, node, tracker, device_frees, host_frees):
        assert self._is_host_leaf(node)
        host_frees[FULL].append(node.component_data[FULL].host_value)
        tracker[FULL] += len(node.key)
        node.parent.children.pop(node.key.child_key(self.page_size))
        node.component_data[FULL].host_value = None
        self.nodes.pop(node.id, None)
        self.evicted.append(node)

    def insert_host(self, node_id, key, host_value, hash_value):
        node = self.node_by_id(node_id)
        matched = 0
        while len(key) > 0:
            child = node.children.get(key.child_key(self.page_size))
            if child is None or child.component_data[FULL].host_value is None:
                break
            n = child.key.match(key, page_size=self.page_size)
            if n < len(child.key):
                break
            matched += n
            key = key[n:]
            node = child
        if len(key) == 0:
            return SimpleNamespace(
                host_insert_dropped=False, inserted_host_node=node.id, cache_actions=[]
            )
        if node is not self.root_node and node.component_data[FULL].host_value is None:
            return SimpleNamespace(
                host_insert_dropped=True, inserted_host_node=None, cache_actions=[]
            )
        new = FakeNode(key, host_value=host_value[matched:].clone())
        new.parent = node
        new.hash_value = list(hash_value[matched // self.page_size :])
        node.children[key.child_key(self.page_size)] = new
        self.nodes[new.id] = new
        self.inserted.append(new)
        return SimpleNamespace(
            host_insert_dropped=False, inserted_host_node=new.id, cache_actions=[]
        )

    def path(self, tokens):
        """Host-backed node chain matching ``tokens`` from the root."""
        key = RadixKey(array("q", tokens), None)
        node, out = self.root_node, []
        while len(key) > 0:
            child = node.children.get(key.child_key(self.page_size))
            if child is None:
                break
            n = child.key.match(key, page_size=self.page_size)
            out.append(child)
            if n < len(child.key):
                break
            key, node = key[n:], child
        return out


class FakeTree:
    def __init__(self, pool, core=None):
        self.page_size = PAGE
        self.tree_core = core or FakeCore()
        self.pool = pool
        self.cache_controller = SimpleNamespace(
            mem_pool_host=SimpleNamespace(
                anchor_entry=SimpleNamespace(host_pool=pool, name="kv")
            )
        )
        self.locks = []
        self.unlocks = []
        self.freed = []

    def inc_host_lock_ref(self, node_id):
        self.tree_core.node_by_id(node_id).component_data[FULL].host_lock_ref += 1
        self.locks.append(node_id)
        return SimpleNamespace(to_dec_params=lambda: ("host", node_id))

    def dec_host_lock_ref(self, node_id, params):
        node = self.tree_core.nodes.get(node_id)
        if node is not None:
            node.component_data[FULL].host_lock_ref -= 1
        self.unlocks.append(node_id)

    def _free_values(self, device_frees, host_frees):
        # Consume the entries as the real drain does.
        for ct in list(host_frees):
            for t in host_frees.pop(ct):
                self.freed.append(t)
                self.pool.free(t)
        device_frees.clear()

    def match_prefix(self, params):
        device = 0
        for node in self.tree_core.path(list(params.key.token_ids)):
            if node.component_data[FULL].value is None:
                break
            device += len(node.key)
        return SimpleNamespace(device_indices=torch.arange(device), host_hit_length=0)

    def _apply_cache_actions(self, actions):
        pass


class StubManager(pd.PeerDemoteMixin, pp.PeerPrefixMixin):
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
        self._transfer_parallelism = 1
        self.sent = []
        self.peer = None
        self.drop_sends = False
        self._peer_init()
        self._demote_init()

    def _send_multipart_locked(self, endpoint, parts, is_ipv6=False):
        self.sent.append((endpoint, parts))
        if self.peer is not None and not self.drop_sends:
            self.peer._message_handlers[bytes(parts[0])](parts)

    def _drain_engine(self):
        pass

    def _on_pull_req(self, msg):
        pass


def pair():
    prefill, decode = StubManager("prefill"), StubManager("decode")
    prefill.peer, decode.peer = decode, prefill
    prefill._peer_decode_endpoint = "tcp://10.0.0.2:31000"
    return prefill, decode


A = list(range(0, 8))  # two pages
B = list(range(8, 16))  # two pages


def prefill_tree(pages=8):
    """Root -> A (host, evicted) -> B (host-only leaf). Pool nearly full."""
    pool = FakeHostPool("prefill", pages)
    core = FakeCore()
    a = core.add(core.root_node, A)
    b = core.add(a, B)
    # The two nodes' pages are the pool's own pages: take them from the pool.
    used = pool.alloc(len(A) + len(B))
    a.component_data[FULL].host_value = used[: len(A)]
    b.component_data[FULL].host_value = used[len(A) :]
    return FakeTree(pool, core), a, b


def decode_tree(pages=8, with_a=True, with_b=False):
    pool = FakeHostPool("decode", pages)
    core = FakeCore()
    a = None
    if with_a:
        a = core.add(core.root_node, A)
        a.component_data[FULL].host_value = pool.alloc(len(A))
        if with_b:
            b = core.add(a, B)
            b.component_data[FULL].host_value = pool.alloc(len(B))
    return FakeTree(pool, core)


def sweep_ctx(tree, **kw):
    kwargs = dict(
        tree_cache=tree,
        host_pool=tree.pool,
        page_size=PAGE,
        low_watermark=0.03,
        high_watermark=0.05,
        batch_pages=128,
        period_s=0.0,
        timeout_s=5.0,
    )
    kwargs.update(kw)
    return pd.SweepContext(**kwargs)


def land_ctx(tree, **kw):
    kwargs = dict(
        tree_cache=tree, host_pool=tree.pool, page_size=PAGE, lease_timeout_s=5.0
    )
    kwargs.update(kw)
    return pd.LandingContext(**kwargs)


def run_push(prefill):
    batch = prefill._demote_batch
    assert batch is not None and batch.stage == pd.STAGE_PUSHING
    batch.push_thread.join(5)


# --------------------------------------------------------------------------- #
# Wire
# --------------------------------------------------------------------------- #
def test_messages_round_trip():
    offer = DemoteOffer(
        seq=3,
        prefill_ip="10.0.0.1",
        prefill_port=31000,
        nodes=[DemoteNode(token_ids=[1, 2, 3, 4], pages=[7], cache_salt="s")],
        deadline_ms=1000,
    )
    back = DemoteOffer.from_frames(offer.to_frames())
    assert back.nodes[0].pages == [7] and back.nodes[0].cache_salt == "s"
    ack = DemoteAck(seq=3, accepted=[0], landing=[[9]], address="d:1", existing=[])
    assert DemoteAck.from_frames(ack.to_frames()).landing == [[9]]
    done = DemoteDone(seq=3, ok=False, accepted=[0])
    assert DemoteDone.from_frames(done.to_frames()).ok is False


# --------------------------------------------------------------------------- #
# Sweep + landing, end to end on two stub managers
# --------------------------------------------------------------------------- #
def test_round_trip_demotes_the_cold_leaf():
    prefill, decode = pair()
    ptree, a, b = prefill_tree(pages=8)  # 4 of 8 pages used -> free 0.5
    dtree = decode_tree(pages=8, with_a=True)
    ctx = sweep_ctx(ptree, low_watermark=0.6, high_watermark=0.7)
    lctx = land_ctx(dtree)

    prefill.peer_demote_tick(ctx)  # under the low watermark: offer B
    batch = prefill._demote_batch
    assert batch is not None and [v.node_id for v in batch.victims] == [b.id]
    assert batch.victims[0].token_ids == A + B
    assert b.component_data[FULL].host_lock_ref == 1  # pinned for the offer
    assert prefill.sent[-1][1][0] == DemoteOffer.TAG

    decode.peer_demote_serve(lctx)  # dedupe, allocate landing, ack
    ack = DemoteAck.from_frames(decode.sent[-1][1])
    assert ack.accepted == [0] and ack.existing == [] and ack.address == "decode:7000"
    assert (
        len(ack.landing[0]) == 2 and dtree.pool.available_size() == (8 - 2 - 2) * PAGE
    )

    prefill.peer_demote_tick(ctx)  # acked -> push off the scheduler thread
    run_push(prefill)
    (uuid, peer, src, dst, layer) = ptree.pool.transfer._mgr.pushes[0]
    assert uuid == pd.DEMOTE_UUID_BASE + 1 and peer == "decode:7000" and layer == -1
    assert src == batch.victims[0].pages and dst == ack.landing[0]

    prefill.peer_demote_tick(ctx)  # pushed -> done, evict locally, unpin
    assert DemoteDone.from_frames(prefill.sent[-1][1]).ok is True
    assert ptree.tree_core.evicted == [b] and a.children == {}
    assert B[0] not in [t for n in ptree.tree_core.path(A + B) for t in n.key.token_ids]
    assert ptree.pool.available_size() == (8 - 2) * PAGE  # B's two pages freed
    assert ptree.unlocks == [b.id] and prefill._demote_batch is None
    stats = prefill.peer_demote_stats()
    assert stats["pushed_pages"] == 2 and stats["evicted_pages"] == 2

    decode.peer_demote_serve(lctx)  # done -> insert the host-only node
    new = dtree.tree_core.inserted
    assert len(new) == 1 and list(new[0].key.token_ids) == B
    assert dtree.tree_core.path(A + B)[-1] is new[0]
    assert new[0].component_data[FULL].host_value.tolist() == [
        p * PAGE + i for p in ack.landing[0] for i in range(PAGE)
    ]
    assert decode.peer_demote_stats()["landings"] == 0


def test_a_node_the_decode_holds_is_evicted_without_a_push():
    prefill, decode = pair()
    ptree, a, b = prefill_tree(pages=8)
    dtree = decode_tree(pages=8, with_a=True, with_b=True)
    ctx = sweep_ctx(ptree, low_watermark=0.6, high_watermark=0.7)
    prefill.peer_demote_tick(ctx)
    decode.peer_demote_serve(land_ctx(dtree))
    ack = DemoteAck.from_frames(decode.sent[-1][1])
    assert ack.accepted == [] and ack.existing == [0]
    prefill.peer_demote_tick(ctx)
    assert ptree.pool.transfer._mgr.pushes == []
    assert ptree.tree_core.evicted == [b] and prefill._demote_batch is None
    assert prefill.peer_demote_stats()["existing_pages"] == 2
    assert prefill._demote_episode  # the target still takes nodes


def test_refusal_keeps_the_pages_and_ends_the_episode():
    prefill, decode = pair()
    ptree, a, b = prefill_tree(pages=8)
    dtree = decode_tree(pages=2, with_a=True)  # A fills the decode pool
    ctx = sweep_ctx(ptree, low_watermark=0.6, high_watermark=0.7)
    prefill.peer_demote_tick(ctx)
    decode.peer_demote_serve(land_ctx(dtree))
    ack = DemoteAck.from_frames(decode.sent[-1][1])
    assert ack.accepted == [] and ack.existing == [] and ack.error == ""
    prefill.peer_demote_tick(ctx)
    assert ptree.tree_core.evicted == [] and b.component_data[FULL].host_lock_ref == 0
    assert prefill._demote_batch is None and prefill._demote_targets == []
    prefill.peer_demote_tick(ctx)  # no target left: the episode ends
    assert not prefill._demote_episode
    assert prefill.peer_demote_stats()["refused"] == 1


def test_parent_not_on_the_decode_host_tier_is_refused():
    prefill, decode = pair()
    ptree, a, b = prefill_tree(pages=8)
    dtree = decode_tree(pages=8, with_a=False)
    prefill.peer_demote_tick(sweep_ctx(ptree, low_watermark=0.6, high_watermark=0.7))
    decode.peer_demote_serve(land_ctx(dtree))
    ack = DemoteAck.from_frames(decode.sent[-1][1])
    assert ack.accepted == [] and dtree.pool.available_size() == 8 * PAGE


def test_watermarks_gate_the_sweep_with_hysteresis():
    prefill, decode = pair()
    ptree, a, b = prefill_tree(pages=8)  # free 0.5
    dtree = decode_tree(pages=16, with_a=True)
    lctx = land_ctx(dtree)
    # Above the low watermark: nothing happens.
    prefill.peer_demote_tick(sweep_ctx(ptree, low_watermark=0.4, high_watermark=0.9))
    assert prefill._demote_batch is None and not prefill._demote_episode
    # Under it: B goes (free 0.75), still under high 0.9: A goes next (free 1.0).
    ctx = sweep_ctx(ptree, low_watermark=0.6, high_watermark=0.9)
    for _ in range(2):
        prefill.peer_demote_tick(ctx)  # offer
        decode.peer_demote_serve(lctx)  # ack
        prefill.peer_demote_tick(ctx)  # push, or finish at once when held
        if prefill._demote_batch is not None:
            run_push(prefill)
            prefill.peer_demote_tick(ctx)
        decode.peer_demote_serve(lctx)
    assert [n.id for n in ptree.tree_core.evicted] == [b.id, a.id]
    assert ptree.pool.available_size() == 8 * PAGE
    prefill.peer_demote_tick(ctx)  # recovered: episode over
    assert not prefill._demote_episode and prefill._demote_batch is None
    assert [list(n.key.token_ids) for n in dtree.tree_core.inserted] == [B]  # A existed


def test_timeout_abandons_the_offer_and_unpins():
    prefill, decode = pair()
    prefill.drop_sends = True  # the decode never hears the offer
    ptree, a, b = prefill_tree(pages=8)
    ctx = sweep_ctx(ptree, low_watermark=0.6, high_watermark=0.7, timeout_s=0.01)
    prefill.peer_demote_tick(ctx)
    assert b.component_data[FULL].host_lock_ref == 1
    time.sleep(0.02)
    prefill.peer_demote_tick(ctx)
    assert prefill._demote_batch is None and b.component_data[FULL].host_lock_ref == 0
    assert ptree.tree_core.evicted == [] and not prefill._demote_episode
    assert prefill.peer_demote_stats()["failed"] == 1


def test_push_failure_aborts_and_the_decode_frees_its_landing_pages():
    prefill, decode = pair()
    ptree, a, b = prefill_tree(pages=8)
    ptree.pool.transfer._mgr.fail = True
    dtree = decode_tree(pages=8, with_a=True)
    ctx = sweep_ctx(ptree, low_watermark=0.6, high_watermark=0.7)
    lctx = land_ctx(dtree)
    prefill.peer_demote_tick(ctx)
    decode.peer_demote_serve(lctx)
    assert dtree.pool.available_size() == 4 * PAGE
    prefill.peer_demote_tick(ctx)
    run_push(prefill)
    prefill.peer_demote_tick(ctx)
    done = DemoteDone.from_frames(prefill.sent[-1][1])
    assert done.ok is False and ptree.tree_core.evicted == []
    assert b.component_data[FULL].host_lock_ref == 0 and not prefill._demote_episode
    decode.peer_demote_serve(lctx)
    assert dtree.pool.available_size() == 6 * PAGE and dtree.tree_core.inserted == []
    assert decode.peer_demote_stats()["dropped_landings"] == 1


def test_nudge_runs_the_check_before_the_period():
    prefill, decode = pair()
    ptree, a, b = prefill_tree(pages=8)
    ctx = sweep_ctx(ptree, low_watermark=0.6, high_watermark=0.7, period_s=3600)
    prefill._demote_last_check = time.monotonic()
    prefill.peer_demote_tick(ctx)
    assert prefill._demote_batch is None  # not due yet
    prefill.peer_demote_nudge()
    prefill.peer_demote_tick(ctx)
    assert prefill._demote_batch is not None


def test_ranks_that_disagree_on_the_batch_do_not_offer():
    prefill, decode = pair()
    ptree, a, b = prefill_tree(pages=8)
    ctx = sweep_ctx(
        ptree,
        low_watermark=0.6,
        high_watermark=0.7,
        reduce_min=lambda v: [
            x - 1 if x > 0 else x for x in v
        ],  # another rank saw less
        reduce_max=lambda v: v,
    )
    prefill.peer_demote_tick(ctx)
    assert prefill._demote_batch is None and not prefill._demote_episode
    assert b.component_data[FULL].host_lock_ref == 0


def test_decode_serves_only_offers_every_rank_received():
    prefill, decode = pair()
    ptree, a, b = prefill_tree(pages=8)
    dtree = decode_tree(pages=8, with_a=True)
    prefill.peer_demote_tick(sweep_ctx(ptree, low_watermark=0.6, high_watermark=0.7))
    # This rank has seq 1, the other rank has nothing yet: wait.
    decode.peer_demote_serve(land_ctx(dtree, gather=lambda local: [local, ((), ())]))
    assert decode.sent == [] and len(decode._demote_offer_inbox) == 1
    decode.peer_demote_serve(land_ctx(dtree, gather=lambda local: [local, local]))
    assert DemoteAck.from_frames(decode.sent[-1][1]).accepted == [0]


def test_landing_expires_when_no_done_arrives():
    prefill, decode = pair()
    ptree, a, b = prefill_tree(pages=8)
    dtree = decode_tree(pages=8, with_a=True)
    prefill.peer_demote_tick(sweep_ctx(ptree, low_watermark=0.6, high_watermark=0.7))
    lctx = land_ctx(dtree, lease_timeout_s=0.01)
    decode.peer_demote_serve(lctx)
    assert dtree.pool.available_size() == 4 * PAGE
    time.sleep(0.02)
    decode.peer_demote_serve(lctx)
    assert dtree.pool.available_size() == 6 * PAGE and decode._demote_landing == {}


# --------------------------------------------------------------------------- #
# Scheduler hooks
# --------------------------------------------------------------------------- #
def test_prefill_tick_hook_builds_its_context_and_consumes_the_nudge(monkeypatch):
    monkeypatch.setenv("SGLANG_TPU_PEER_PREFIX", "1")
    monkeypatch.setenv("SGLANG_TPU_PEER_DEMOTE", "1")
    prefill, decode = pair()
    ptree, a, b = prefill_tree(pages=8)
    scheduler = SimpleNamespace(
        tree_cache=ptree,
        token_to_kv_pool_allocator=SimpleNamespace(page_size=PAGE),
        attn_tp_cpu_group=None,
        disaggregation_mode=hook.DisaggregationMode.PREFILL,
    )
    queue = SimpleNamespace(scheduler=scheduler, kv_manager=prefill)
    monkeypatch.setenv("SGLANG_TPU_PEER_DEMOTE_LOW", "0.6")
    monkeypatch.setenv("SGLANG_TPU_PEER_DEMOTE_HIGH", "0.7")
    monkeypatch.setenv("SGLANG_TPU_PEER_DEMOTE_PERIOD_S", "3600")
    prefill._demote_last_check = time.monotonic()
    hook._demote_tick(queue)
    assert prefill._demote_batch is None
    hook._evict_host(lambda cache, n: 0, ptree, 4)  # the tree dropped pages
    hook._demote_tick(queue)
    assert prefill._demote_batch is not None
    assert scheduler._peer_demote_ctx.low_watermark == 0.6


def test_decode_hook_lands_through_process_decode_queue(monkeypatch):
    monkeypatch.setenv("SGLANG_TPU_PEER_PREFIX", "1")
    monkeypatch.setenv("SGLANG_TPU_PEER_DEMOTE", "1")
    prefill, decode = pair()
    ptree, a, b = prefill_tree(pages=8)
    dtree = decode_tree(pages=8, with_a=True)
    scheduler = SimpleNamespace(
        tree_cache=dtree,
        token_to_kv_pool_allocator=SimpleNamespace(page_size=PAGE),
        sliding_window_size=None,
        enable_decode_hicache=True,
        attn_tp_cpu_group=None,
        disagg_decode_prealloc_queue=SimpleNamespace(kv_manager=decode),
    )
    monkeypatch.setattr(
        hook,
        "get_disagg",
        lambda: SimpleNamespace(disaggregation_decode_enable_radix_cache=True),
    )
    prefill.peer_demote_tick(sweep_ctx(ptree, low_watermark=0.6, high_watermark=0.7))
    hook._process_decode_queue(lambda s: None, scheduler)
    assert DemoteAck.from_frames(decode.sent[-1][1]).accepted == [0]


def test_install_wraps_evict_host_when_demotion_is_on(monkeypatch):
    monkeypatch.setenv("SGLANG_TPU_PEER_PREFIX", "1")
    monkeypatch.setenv("SGLANG_TPU_PEER_DEMOTE", "1")
    hook.install()
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    assert getattr(UnifiedRadixCache.evict_host, "_peer_prefix_wrapped", False)
