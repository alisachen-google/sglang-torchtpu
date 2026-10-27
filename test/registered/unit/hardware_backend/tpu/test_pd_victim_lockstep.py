"""Lockstep tests for the P/D victim cache: every attention-TP rank must run the
same collectives, in the same order, with the same shapes.

A rank that skips or adds one leaves the others waiting in a gloo collective
forever. On hardware this surfaced as a prefill hang (ranks in different
all-reduces) followed, minutes later, by libtpu's telemetry harvester aborting
every rank. These tests reproduce the class on CPU in seconds: K simulated ranks
run the same component on K threads with rank-divergent inputs (an ack, a READY,
an allocation failure or a settled read that only some ranks see). The collective
helpers are replaced by a barrier that matches calls across ranks by kind, op,
shape and call site, computes the real result, and raises LockstepFault with
every rank's stack when they disagree or one rank never arrives.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import traceback
import unittest
from pathlib import Path
from unittest.mock import patch

import torch.distributed as dist

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import pd_victim_push as victim_push
from sglang_tpu.disaggregation.raiden.prefix_pull import DecodePrefixPuller

_HERE = Path(__file__).resolve().parent


def _load_sibling(name: str, path: Path):
    """Import a sibling test module for its fixtures without re-registering its tests."""
    with patch(
        "sglang.test.ci.ci_register.register_cpu_ci", lambda *a, **k: lambda c: c
    ):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


_vp = _load_sibling("_lockstep_victim_fixtures", _HERE / "test_pd_victim_push.py")
_pp = _load_sibling(
    "_lockstep_pull_fixtures",
    _HERE.parents[1] / "disaggregation" / "test_raiden_prefix_pull.py",
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

WORLD = 4


class LockstepFault(AssertionError):
    """The simulated ranks disagreed on a collective (or one never arrived)."""


class Lockstep:
    """Barrier standing in for the TP/CP collectives of K simulated ranks."""

    def __init__(self, world: int, timeout: float = 5.0):
        self.world = world
        self.timeout = timeout
        self._cv = threading.Condition()
        self._ranks: dict[int, int] = {}
        self._threads: dict[int, int] = {}
        self._pending: dict[int, tuple] = {}
        self._gen = 0
        self._result = None
        self.fault: str | None = None
        self.trace: dict[int, list] = {r: [] for r in range(world)}

    def bind(self, rank: int) -> None:
        self._ranks[threading.get_ident()] = rank
        self._threads[rank] = threading.get_ident()

    def rank(self) -> int:
        return self._ranks[threading.get_ident()]

    # The three helpers under test, with the real helpers' early returns.
    def reduce(self, rows, op):
        if not rows:
            return [list(row) for row in rows]
        return self._call("reduce", str(op), [list(r) for r in rows])

    def common(self, keys):
        return self._call("common", "", sorted(keys)[: victim_push._AGREE_SLOTS])

    @staticmethod
    def _site() -> str:
        for frame in traceback.extract_stack()[::-1]:
            if not frame.filename.endswith("test_pd_victim_lockstep.py"):
                return f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
        return "?"

    @staticmethod
    def _shape(kind, payload):
        if kind == "reduce":
            return (len(payload), len(payload[0]) if payload else 0)
        return ("common",)

    def _stacks(self, ranks) -> str:
        frames = sys._current_frames()
        out = []
        for r in ranks:
            tid = self._threads.get(r)
            if tid in frames:
                stack = "".join(traceback.format_stack(frames[tid])[-8:])
                out.append(f"--- rank {r} stack:\n{stack}")
            else:
                out.append(f"--- rank {r}: thread finished (ran fewer collectives)")
        return "\n".join(out)

    def _combine(self, kind, op, payloads):
        if kind == "common":
            common = set(payloads[0])
            for p in payloads[1:]:
                common &= set(p)
            return sorted(common)
        pick = min if op.endswith("MIN") else max
        return [
            [pick(p[i][j] for p in payloads) for j in range(len(payloads[0][i]))]
            for i in range(len(payloads[0]))
        ]

    def _call(self, kind, op, payload):
        rank = self.rank()
        site = self._site()
        shape = self._shape(kind, payload)
        with self._cv:
            self.trace[rank].append((kind, op, shape, site))
            if self.fault:
                raise LockstepFault(self.fault)
            gen = self._gen
            self._pending[rank] = (kind, op, shape, site, payload)
            if len(self._pending) == self.world:
                sigs = {v[:4] for v in self._pending.values()}
                if len(sigs) > 1:
                    listing = "\n".join(
                        f"  rank {r}: {v[0]} {v[1]} shape={v[2]} at {v[3]}"
                        for r, v in sorted(self._pending.items())
                    )
                    self.fault = f"ranks disagree on the collective:\n{listing}"
                else:
                    self._result = self._combine(
                        kind, op, [self._pending[r][4] for r in range(self.world)]
                    )
                self._pending = {}
                self._gen += 1
                self._cv.notify_all()
                if self.fault:
                    raise LockstepFault(self.fault)
                return self._result
            while self._gen == gen and not self.fault:
                if not self._cv.wait(timeout=self.timeout):
                    here = sorted(self._pending)
                    missing = [r for r in range(self.world) if r not in self._pending]
                    self.fault = (
                        f"ranks {here} called {kind} {op} shape={shape} at {site}; "
                        f"ranks {missing} did not within {self.timeout}s\n"
                        + self._stacks(missing)
                    )
                    self._cv.notify_all()
            if self.fault:
                raise LockstepFault(self.fault)
            return self._result


def run_ranks(world, body, timeout=30.0):
    """Run body(rank, lockstep) on one thread per rank with the collectives patched."""
    ls = Lockstep(world)
    errors: dict[int, str] = {}

    def main(rank):
        ls.bind(rank)
        try:
            body(rank, ls)
        except BaseException:  # noqa: BLE001 - every rank's failure is reported
            errors[rank] = traceback.format_exc()

    with (
        patch.object(
            victim_push,
            "_all_ranks_reduce",
            lambda groups, rows, op: ls.reduce(rows, op),
        ),
        patch.object(
            victim_push, "_all_ranks_common", lambda groups, keys: ls.common(keys)
        ),
        patch.object(
            DecodePrefixPuller,
            "_all_ranks_reduce",
            lambda self, rows, op: ls.reduce(rows, op),
        ),
    ):
        threads = [
            threading.Thread(target=main, args=(r,), name=f"rank{r}", daemon=True)
            for r in range(world)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout)
        alive = [t.name for t in threads if t.is_alive()]
    return ls, errors, alive


class LockstepCase(CustomTestCase):
    def assert_lockstep(self, ls, errors, alive):
        report = "\n".join(f"rank {r}:\n{tb}" for r, tb in sorted(errors.items()))
        self.assertFalse(errors, f"ranks raised:\n{report}\nfault={ls.fault}")
        self.assertFalse(alive, f"ranks still running: {alive}\nfault={ls.fault}")
        self.assertIsNone(ls.fault)
        traces = [ls.trace[r] for r in range(ls.world)]
        for r in range(1, ls.world):
            self.assertEqual(
                traces[r], traces[0], f"rank {r} ran different collectives than rank 0"
            )
        return traces[0]


class TestLockstepHarness(LockstepCase):
    def test_a_rank_that_skips_a_collective_is_reported(self):
        def body(rank, ls):
            ls.reduce([[1]], dist.ReduceOp.MIN)
            if rank == 0:
                ls.reduce([[1]], dist.ReduceOp.MIN)  # the extra call

        ls, errors, alive = run_ranks(2, body)
        self.assertIn(0, errors)
        self.assertIn("ranks [1] did not", errors[0])
        self.assertIn("ran fewer collectives", errors[0])

    def test_ranks_that_disagree_on_shape_are_reported(self):
        def body(rank, ls):
            ls.reduce([[1]] * (rank + 1), dist.ReduceOp.MIN)

        ls, errors, alive = run_ranks(2, body)
        self.assertTrue(errors)
        self.assertIn("disagree", next(iter(errors.values())))

    def test_results_are_the_real_reduction(self):
        seen = {}

        def body(rank, ls):
            seen[rank] = (
                ls.reduce([[rank, 10 - rank]], dist.ReduceOp.MIN),
                ls.reduce([[rank, 10 - rank]], dist.ReduceOp.MAX),
                ls.common([rank, 7, 8]),
            )

        ls, errors, alive = run_ranks(3, body)
        self.assert_lockstep(ls, errors, alive)
        self.assertEqual(seen[2], ([[0, 8]], [[2, 10]], [7, 8]))


def _pull_fixture(free_tokens=64):
    return _pp._offered(have_tokens=4, free_tokens=free_tokens)


class TestPrefixPullerLockstep(LockstepCase):
    """Regression for the prefill deadlock: READY replies, allocation failures,
    READY mismatches and read settlement differ per rank, yet every rank must
    run the same collectives and free the same device pages."""

    def test_ready_on_a_subset_and_allocation_failure_on_one_rank(self):
        fx = [_pull_fixture() for _ in range(WORLD)]

        def body(rank, ls):
            puller, tree, allocator, mgr = fx[rank]
            req = _pp._req()
            puller.advance([req])  # offer taken and asked on every rank
            if rank in (0, 2):
                mgr.ready[_pp.ROOM] = (_pp.PX, 99, _pp.DECODE_RAIDEN, 1, 3)
            puller.advance([req])  # READY on two ranks only: nobody allocates
            self.assertEqual(allocator.allocs, [])
            mgr.ready[_pp.ROOM] = (_pp.PX, 99, _pp.DECODE_RAIDEN, 1, 3)
            allocator.fail = rank == 1  # one rank has no room for the pull
            puller.advance([req])  # agreed: nobody keeps pages, the room resolves
            self.assertNotIn(_pp.ROOM, puller._plans)
            self.assertIn(_pp.ROOM, mgr.resolved)
            fx[rank] = (puller, tree, allocator, mgr)

        ls, errors, alive = run_ranks(WORLD, body)
        self.assert_lockstep(ls, errors, alive)
        # The ranks that allocated freed exactly what they took; nobody holds pages.
        for rank, (_, _, allocator, _) in enumerate(fx):
            taken = sum(allocator.allocs) if not allocator.fail else 0
            freed = sum(len(x) for x in allocator.freed)
            self.assertEqual((rank, taken), (rank, freed))

    def test_reads_settle_on_a_subset_of_ranks_first(self):
        fx = [_pull_fixture() for _ in range(WORLD)]

        def body(rank, ls):
            puller, tree, allocator, mgr = fx[rank]
            req = _pp._req()
            puller.advance([req])
            mgr.ready[_pp.ROOM] = (_pp.PX, 99, _pp.DECODE_RAIDEN, 1, 3)
            puller.advance([req])  # allocated and started on every rank
            self.assertEqual(puller._plans[_pp.ROOM].state, "pulling")
            mgr.read_states[_pp.PX] = "pending"
            # Reads settle on two ranks first; the plan completes only when all did.
            if rank in (0, 3):
                mgr.read_states[_pp.PX] = "done"
            puller.advance([req])
            self.assertIsNone(puller._plans[_pp.ROOM].agreed_done)
            mgr.read_states[_pp.PX] = "done"
            puller.advance([req])
            self.assertTrue(puller._plans[_pp.ROOM].agreed_done)
            puller.commit(req)
            self.assertNotIn(_pp.ROOM, puller._plans)
            fx[rank] = (puller, tree, allocator, mgr)

        ls, errors, alive = run_ranks(WORLD, body)
        self.assert_lockstep(ls, errors, alive)
        self.assertEqual({fx[r][0].stats["pulls"] for r in range(WORLD)}, {1})


if __name__ == "__main__":
    unittest.main()
