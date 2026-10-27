#!/usr/bin/env python3
"""Does a tpu-sync host-pool manager serve a host->host read in both directions?

Two managers on this host (A, B), each with a control port and a host pool.
Direction 1 (push path): A registers host blocks, B start_read(destination=host).
Direction 2 (recall path): B registers host blocks, A start_read(destination=host).
Each read polls poll_stats_with_details on both sides until settled or --timeout.
"""

import argparse
import json
import random
import time

import torch
import torch_tpu  # noqa: F401
from tpu_sync.api.torch.kv_cache_manager import KVCacheManager


def manager(port, dev, layers, slice_bytes, pages, timeout_s):
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
        timeout_s=timeout_s,
        unsafe_skip_buffer_lock=True,
    )
    return m, t


def settle(src, dst, req_id, timeout_s):
    t0 = time.time()
    seen = {"src": None, "dst": None}
    while time.time() - t0 < timeout_s and (seen["src"] is None or seen["dst"] is None):
        for name, m in (("src", src), ("dst", dst)):
            r = m.poll_stats_with_details()
            for status, lst in (
                ("sent", r.sent),
                ("received", r.received),
                ("failed", r.failed),
            ):
                for tr in lst:
                    if tr.req_id == req_id:
                        seen[name] = (
                            status,
                            getattr(tr, "code", ""),
                            getattr(tr, "phase", ""),
                            (getattr(tr, "reason", "") or "")[:80],
                        )
        time.sleep(0.05)
    return seen, round(time.time() - t0, 2)


def run(direction, src, dst, dst_ep, pages, layers, slice_bytes, timeout_s, label):
    rnd = random.Random(7).randbytes(slice_bytes)
    for layer in range(layers):
        for b in range(pages):
            src.write_block_bytes(layer, b, rnd)
    req, uuid = f"{label}#px", 4242 + direction
    src_blocks = list(range(pages))
    dst_blocks = list(range(pages, 2 * pages))
    needed = src.register_read(req, uuid, src_blocks, source="host")
    t0 = time.time()
    dst.start_read(
        req,
        uuid,
        dst_ep,
        src_blocks,
        None,
        parallelism=4,
        local_host_block_ids=dst_blocks,
        destination="host",
    )
    seen, dt = settle(src, dst, req, timeout_s)
    ok = (
        seen["dst"] is not None
        and seen["dst"][0] == "received"
        and dst.read_block_bytes(0, dst_blocks[0]) == rnd
    )
    print(
        "RESULT "
        + json.dumps(
            {
                "direction": label,
                "registered": needed,
                "src": seen["src"],
                "dst": seen["dst"],
                "settle_s": dt,
                "data_ok": ok,
            }
        ),
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--slice-bytes", type=int, default=8192)
    ap.add_argument("--timeout", type=float, default=30)
    a = ap.parse_args()
    import socket

    ip = socket.gethostbyname(socket.gethostname())
    A, _a = manager(48400, 0, a.layers, a.slice_bytes, 2 * a.pages, a.timeout)
    B, _b = manager(48430, 1, a.layers, a.slice_bytes, 2 * a.pages, a.timeout)
    # Host reads are served on the control port (sglang: host_control_address),
    # not on the block data port (transfer_address).
    ep_a, ep_b = f"{ip}:48400", f"{ip}:48430"
    print(
        "endpoints",
        ep_a,
        ep_b,
        "data ports",
        A.transfer_address,
        B.transfer_address,
        flush=True,
    )
    run(
        1,
        A,
        B,
        ep_a,
        a.pages,
        a.layers,
        a.slice_bytes,
        a.timeout,
        "A_registers_B_reads",
    )
    run(
        2,
        B,
        A,
        ep_b,
        a.pages,
        a.layers,
        a.slice_bytes,
        a.timeout,
        "B_registers_A_reads",
    )
    # A manager that already served a read (A) now also serves as reader and vice versa.
    run(
        3,
        A,
        B,
        ep_a,
        a.pages,
        a.layers,
        a.slice_bytes,
        a.timeout,
        "A_registers_B_reads_again",
    )


if __name__ == "__main__":
    main()
