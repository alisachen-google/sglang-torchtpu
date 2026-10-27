#!/usr/bin/env python3
"""Does a burst of tpu-sync host-to-host pushes stall TPU device->host copies?

Thread A: TPU matmuls with a device->host copy per iteration (the batch-result copy
in SGLang), timing each copy. Thread B: back-to-back h2h_push of a 128k-victim
share (2176 pages x 18 layers) between two host-pool managers on this host.
A copy that takes longer than --stall-s is reported; the run ends after --duration.
"""

import argparse
import json
import random
import threading
import time

import torch
import torch_tpu  # noqa: F401
from tpu_sync.api.torch.kv_cache_manager import KVCacheManager


def manager(port, dev, layers, slice_bytes, pages):
    t = [
        [
            torch.zeros(
                (8, slice_bytes), dtype=torch.uint8, device=torch.device("tpu", dev)
            )
        ]
        for _ in range(layers)
    ]
    m = KVCacheManager(
        t,
        local_control_port=port,
        host_blocks_to_allocate=pages,
        parallelism=4,
        timeout_s=120.0,
        unsafe_skip_buffer_lock=True,
    )
    return m, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=2176)
    ap.add_argument("--layers", type=int, default=18)
    ap.add_argument("--slice-bytes", type=int, default=8192)
    ap.add_argument("--duration", type=float, default=90)
    ap.add_argument("--stall-s", type=float, default=20)
    ap.add_argument(
        "--no-push", action="store_true", help="baseline: compute + copies only"
    )
    a = ap.parse_args()
    src, _s = manager(48300, 0, a.layers, a.slice_bytes, a.pages)
    dst, _d = manager(48330, 1, a.layers, a.slice_bytes, 2 * a.pages)
    peer = str(dst.transfer_address)
    rnd = random.Random(1).randbytes(a.slice_bytes)
    for layer in range(a.layers):
        for b in range(a.pages):
            src.write_block_bytes(layer, b, rnd)
    stop = threading.Event()
    stats = {"pushes": 0, "push_fail": 0, "copies": 0, "max_copy_s": 0.0, "stalls": []}

    def pusher():
        uuid = 0
        while not stop.is_set():
            try:
                src.h2h_push(
                    peer=peer,
                    src_block_ids=list(range(a.pages)),
                    dst_block_ids=list(range(a.pages, 2 * a.pages)),
                    uuid=uuid,
                )
                stats["pushes"] += 1
            except Exception as e:  # noqa: BLE001
                stats["push_fail"] += 1
                print("push failed:", str(e)[:120], flush=True)
                time.sleep(1)

    dev = torch.device("tpu", 0)
    x = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    t_end = time.time() + a.duration
    if not a.no_push:
        threading.Thread(target=pusher, daemon=True).start()
    while time.time() < t_end:
        y = x @ x
        t0 = time.perf_counter()
        h = y[:64].to("cpu")  # device -> host copy, like the batch-result copy
        dt = time.perf_counter() - t0
        stats["copies"] += 1
        stats["max_copy_s"] = max(stats["max_copy_s"], dt)
        if dt > a.stall_s:
            stats["stalls"].append(round(dt, 1))
            print(
                f"STALL: copy took {dt:.1f}s after {stats['pushes']} pushes", flush=True
            )
        x = y / 100 + h.sum().item() * 0  # keep the chain live
    stop.set()
    print(
        "RESULT "
        + json.dumps(
            {
                "push": not a.no_push,
                **{k: (v if k != "stalls" else v[:5]) for k, v in stats.items()},
                "max_copy_ms": round(stats["max_copy_s"] * 1e3, 1),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
