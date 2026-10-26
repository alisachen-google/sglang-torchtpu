"""Unit tests for victim push (sglang_tpu/pd_victim_push.py).

The prefill hands the session with the oldest path end to the decode that
served it, holding it locked until the decode answers, and makes it its own
next eviction victim once the decode stored it. The decode reads pushed pages
and inserts them. Real full-attention unified trees with a fake raiden
controller; the raiden manager is a fake.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from sglang.srt.managers.cache_controller import HiCacheAck
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_utils import CustomTestCase

from sglang_tpu import pd_victim_push as victim_push
from sglang_tpu.disaggregation.raiden.serde import VictimAck, VictimPush
from sglang_tpu.raiden_hybrid_cache_controller import TpuRaidenHybridCacheController
from sglang_tpu.raiden_transfer import NOOP_EVENT

FULL = ComponentType.FULL


class _FakeFuture:
    def __init__(self):
        self.ready = True

    def query(self) -> bool:
        return self.ready

    def synchronize(self) -> None:
        self.ready = True


class _FakeHostPool:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.used = 0

    def available_size(self) -> int:
        return self.capacity - self.used

    def alloc(self, num_tokens: int):
        if num_tokens > self.available_size():
            return None
        indices = torch.arange(self.used, self.used + num_tokens, dtype=torch.int64)
        self.used += num_tokens
        return indices

    def free(self, indices) -> None:
        self.used -= len(indices)

    def clear(self) -> None:
        self.used = 0


class _FakeRaidenController(TpuRaidenHybridCacheController):
    """The controller surface the cache's write-back paths touch."""

    def __init__(self, *, host_capacity: int):
        self.write_policy = "write_back"
        self.host = _FakeHostPool(host_capacity)
        self.mem_pool_host = SimpleNamespace(
            anchor_entry=SimpleNamespace(
                host_pool=SimpleNamespace(shared_allocation_domain=None),
                host_evict_fn=None,
            ),
            available_size=self.host.available_size,
            clear=self.host.clear,
        )
        self.write_queue: list = []
        self.ack_write_queue: list = []
        self.ack_load_queue: list = []
        self.extra_host_mem_release_queues: dict = {}

    def write(
        self, device_indices, priority=None, node_id=-1, extra_pools=None, flush=True
    ):
        host_indices = self.host.alloc(len(device_indices))
        if host_indices is None:
            return None
        self.write_queue.append((host_indices, device_indices, node_id))
        if flush:
            self.start_writing()
        return host_indices

    def start_writing(self) -> None:
        if not self.write_queue:
            return
        node_ids = [node_id for _, _, node_id in self.write_queue]
        num_tokens = sum(len(d) for _, d, _ in self.write_queue)
        self.write_queue.clear()
        self.ack_write_queue.append(
            HiCacheAck(
                start_event=NOOP_EVENT,
                finish_event=_FakeFuture(),
                node_ids=node_ids,
                num_tokens=num_tokens,
            )
        )

    def reset(self) -> None:
        self.write_queue.clear()
        self.ack_write_queue.clear()


def full_cache(*, kv_size=64, host_capacity=64):
    """A real FULL-only write-back tree over CPU pools, page size 1."""
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", page_size=1))
    req_to_token_pool = ReqToTokenPool(
        size=8, max_context_len=128, device="cpu", enable_memory_saver=False
    )
    kv_pool = MHATokenToKVPool(
        size=kv_size,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=2,
        head_dim=64,
        layer_num=2,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=kv_size,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            tree_components=(FULL,),
            tree_core_backend="python",
        )
    )
    controller = _FakeRaidenController(host_capacity=host_capacity)
    cache.cache_controller = controller
    cache.components[FULL]._full_kv_pool_host = controller.host
    cache.host_pool_group = SimpleNamespace(
        free=lambda indices, pool=None: controller.host.free(indices)
    )
    cache.tree_core.set_hicache_enabled()
    cache.tree_core.is_write_back = True
    cache.is_write_back = True
    cache._write_back_copy_ahead = None
    return cache, allocator


def insert_tokens(cache, allocator, tokens):
    cache.insert(
        InsertParams(
            key=RadixKey(array("q", tokens)), value=allocator.alloc(len(tokens))
        )
    )


def device_hit(cache, tokens) -> int:
    match = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", tokens))))
    return len(match.device_indices)


def root_node(cache, first_token):
    return next(
        c
        for c in cache.tree_core.root_node.children.values()
        if c.key.token_ids[0] == first_token
    )


class TestWire(CustomTestCase):
    def test_push_and_ack_round_trip(self):
        push = VictimPush(
            prefill="10.0.0.9",
            seq=3,
            ack_endpoint="tcp://10.0.0.9:7000",
            tokens=[1, 2],
            endpoint="e",
            req_id="v3#vx",
            uuid=2**62,
            blocks=[4],
        )
        self.assertEqual(VictimPush.from_frames(push.to_frames()), push)
        ack = VictimAck(seq=3, stored=True)
        self.assertEqual(VictimAck.from_frames(ack.to_frames()), ack)


class TestPrefillPusher(CustomTestCase):
    def setUp(self):
        self.cache, self.allocator = full_cache(host_capacity=8)
        self.mgr = MagicMock()
        self.mgr._full_engine_max_blocks = 64
        self.acks = {}
        self.mgr.take_victim_ack.side_effect = lambda seq: self.acks.pop(seq, None)
        self.peers = {}
        self.mgr.decode_peer.side_effect = lambda room: self.peers.get(room)
        puller = SimpleNamespace(
            _mgr=self.mgr,
            _tree_cache=self.cache,
            _allocator=self.allocator,
            _page_size=1,
            _groups=[],
            _window_pages=None,
        )
        self.pusher = victim_push.PrefillPusher(puller)
        self.a = list(range(100, 108))
        self.b = list(range(200, 208))
        # A is older: the next eviction would take it first.
        for room, tokens, decode in (
            (1, self.a, ("10.0.0.1", 5001)),
            (2, self.b, ("10.0.0.2", 5002)),
        ):
            insert_tokens(self.cache, self.allocator, tokens)
            self.peers[room] = decode
            self.pusher.record(
                SimpleNamespace(bootstrap_room=room, origin_input_ids=tokens)
            )

    def push_once(self):
        for _ in range(3):  # pick, on device, send
            self.pusher.step()
        self.mgr.send_victim_push.assert_called_once()
        return self.mgr.send_victim_push.call_args.kwargs

    def test_oldest_session_goes_to_its_own_decode_and_becomes_next_victim(self):
        sent = self.push_once()
        self.assertEqual(sent["decode"], ("10.0.0.1", 5001))
        self.assertEqual((sent["tokens"], len(sent["blocks"])), (self.a, len(self.a)))
        # Locked while the decode reads: eviction takes B instead.
        self.cache.evict(EvictParams(num_tokens=8))
        self.assertEqual(device_hit(self.cache, self.a), 8)
        insert_tokens(self.cache, self.allocator, self.b)
        # Stored: A goes first although the send made it the most recent.
        self.acks[sent["seq"]] = True
        self.pusher.step()
        self.cache.evict(EvictParams(num_tokens=8))
        self.assertTrue(root_node(self.cache, self.a[0]).evicted)
        self.assertFalse(root_node(self.cache, self.b[0]).evicted)

    def test_declined_push_unlocks_and_is_not_repeated(self):
        sent = self.push_once()
        self.acks[sent["seq"]] = False
        self.pusher.step()
        self.assertNotIn(sent["seq"], self.pusher._pushes)
        self.cache.evict(EvictParams(num_tokens=8))
        self.assertTrue(root_node(self.cache, self.a[0]).evicted)
        # Not again at the same length: the next push is B.
        for _ in range(3):
            self.pusher.step()
        self.assertEqual(self.mgr.send_victim_push.call_args.kwargs["tokens"], self.b)

    def test_no_push_without_host_pressure(self):
        with patch.object(victim_push, "_PUSH_FULL_FREE_TOKENS", 0):
            for _ in range(3):
                self.pusher.step()
        self.mgr.send_victim_push.assert_not_called()


class TestDecodeReceiver(CustomTestCase):
    def setUp(self):
        self.cache, self.allocator = full_cache()
        self.mgr = MagicMock()
        self.pushes = {}
        self.mgr.victim_pushes.side_effect = lambda: dict(self.pushes)
        self.mgr.take_victim_push.side_effect = lambda key: self.pushes.pop(key, None)
        self.mgr.start_victim_read.return_value = "started"
        self.read_state = "pending"
        self.mgr.victim_read_state.side_effect = lambda ids: self.read_state
        scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=self.allocator,
            sliding_window_size=None,
            req_to_token_pool=self.cache.req_to_token_pool,
            attn_tp_cpu_group=None,
            attn_cp_cpu_group=None,
        )
        queue = SimpleNamespace(scheduler=scheduler, tree_cache=self.cache)
        self.receiver = victim_push.DecodeReceiver(queue=queue, mgr=self.mgr)
        self.c = list(range(300, 308))
        self.pushes[("10.0.0.9", 1)] = VictimPush(
            prefill="10.0.0.9",
            seq=1,
            ack_endpoint="tcp://p",
            tokens=self.c,
            endpoint="e",
            req_id="v1#vx",
            uuid=5,
            blocks=list(range(8)),
        )

    def round(self):
        for _ in range(victim_push._CADENCE):
            self.receiver.step()

    def test_pushed_session_is_inserted_once_read(self):
        self.round()
        self.assertTrue(self.receiver.holds_pages())
        self.mgr.send_victim_ack.assert_not_called()
        self.read_state = "done"
        self.round()
        self.assertEqual(device_hit(self.cache, self.c), 8)
        self.assertFalse(self.receiver.holds_pages())
        self.mgr.send_victim_ack.assert_called_once_with(
            endpoint="tcp://p", seq=1, stored=True
        )

    def test_failed_read_frees_its_pages(self):
        free_before = self.allocator.available_size()
        self.round()
        self.read_state = "failed"
        self.round()
        self.assertEqual(self.allocator.available_size(), free_before)
        self.assertEqual(device_hit(self.cache, self.c), 0)
        self.mgr.send_victim_ack.assert_called_once_with(
            endpoint="tcp://p", seq=1, stored=False
        )


if __name__ == "__main__":
    unittest.main()
