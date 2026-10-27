#!/usr/bin/env python3
"""Find the first collective where TP ranks diverge, from SGLANG_TPU_COLLECTIVE_TRACE files.

usage: collective_diff.py <dir-or-files...> [--context N]
Lines: "<seq> <t> <thread> <op> g=[ranks] <payload> @ <caller>". Ranks are compared
by sequence number on (op, group, shape, caller); the first mismatch is the
collective that paired wrongly, and the calls before it on each rank show why.
"""

import argparse
import glob
import os
import re

LINE = re.compile(r"^(\d+) (\S+) (\S+) (\S+) g=\[([^\]]*)\] (\S+) @ (.*)$")


NOTE = re.compile(r"^(\d+) (\S+) (\S+) note (.*)$")


def load(path):
    rows = {}
    for line in open(path):
        n = NOTE.match(line.rstrip("\n"))
        if n and n[3] == "MainThread":
            rows[int(n[1])] = {
                "t": float(n[2]),
                "op": "note",
                "group": "",
                "shape": "",
                "values": n[4],
                "caller": "",
            }
            continue
        m = LINE.match(line.rstrip("\n"))
        if (
            not m or m[3] != "MainThread"
        ):  # helper threads (consensus checker) interleave freely
            continue
        seq = int(m[1])
        payload = m[6]
        shape = payload.split("=")[0]
        rows[seq] = {
            "t": float(m[2]),
            "op": m[4],
            "group": m[5],
            "shape": shape,
            "values": payload.split("=", 1)[1] if "=" in payload else "",
            "caller": m[7],
        }
    return rows


def sig(x):
    """Call identity without the parts that legitimately differ between ranks:
    line numbers (a broadcast's source takes another branch) and object payloads."""
    funcs = "<".join(part.split(":")[-1] for part in x["caller"].split("<"))
    files = "<".join(part.split(":")[0] for part in x["caller"].split("<"))
    shape = "" if x["op"].startswith("broadcast") else x["shape"]
    return (x["op"], x["group"], shape, files, funcs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--context", type=int, default=6)
    ap.add_argument("--start", type=int, default=1, help="first seq to compare")
    a = ap.parse_args()
    files = []
    for p in a.paths:
        files += (
            sorted(glob.glob(os.path.join(p, "ctrace_r*.log")))
            if os.path.isdir(p)
            else [p]
        )
    ranks = {}
    for f in files:
        m = re.search(r"ctrace_r(-?\d+)_(\d+)", os.path.basename(f))
        ranks[(int(m[1]), int(m[2])) if m else f] = load(f)
    # newest process per rank wins (a restart leaves older files behind); rows are re-numbered
    # by main-thread order so ranks line up even when a helper thread's calls were skipped
    latest = {}
    for (r, pid), rows in sorted(ranks.items()):
        latest[r] = (pid, {i + 1: x for i, (_, x) in enumerate(sorted(rows.items()))})
    print(
        "ranks:",
        {r: f"pid {pid}, {len(rows)} calls" for r, (pid, rows) in latest.items()},
    )
    n = min(len(rows) for _, rows in latest.values())
    # first note whose rank-local state differs (notes sit at the same seq on every rank while in lockstep)
    for seq in range(a.start, n + 1):
        vals = {
            r: rows[seq]["values"]
            for r, (_, rows) in latest.items()
            if seq in rows and rows[seq]["op"] == "note"
        }
        if len(vals) == len(latest) and len(set(vals.values())) > 1:
            print(f"FIRST STATE DIVERGENCE at note seq {seq}:")
            for r, v in sorted(vals.items()):
                print(f"   rank {r}: {v}")
            print("-- calls before it on rank 0 / on a differing rank:")
            groups = {}
            for r, v in vals.items():
                groups.setdefault(v, r)
            for r in sorted(groups.values()):
                rows = latest[r][1]
                print(f"   == rank {r}")
                for s2 in range(max(1, seq - a.context * 2), seq):
                    x = rows.get(s2)
                    if x:
                        print(
                            f"      {s2} {x['t']:.3f} {x['op']} {x['shape']}{'=' + x['values'] if x['values'] else ''} @ {x['caller'][:150]}"
                        )
            break
    first = None
    for seq in range(a.start, n + 1):
        sigs = {r: sig(rows[seq]) for r, (_, rows) in latest.items() if seq in rows}
        if len(set(sigs.values())) > 1:
            first = seq
            break
    if first is None:
        print(
            f"no call-site divergence in the first {n} calls common to all ranks; per-rank totals differ by",
            {r: len(rows) - n for r, (_, rows) in latest.items()},
        )
        # the rank(s) with extra calls: show their tail
        for r, (_, rows) in latest.items():
            if len(rows) > n:
                print(f"-- rank {r} extra calls after seq {n}:")
                for seq in range(n + 1, min(len(rows), n + a.context) + 1):
                    x = rows.get(seq)
                    if x:
                        print(f"   {seq} {x['op']} {x['shape']} @ {x['caller']}")
        return
    print(f"FIRST DIVERGENCE at seq {first}:")
    for r, (_, rows) in sorted(latest.items()):
        print(f"-- rank {r}:")
        for seq in range(max(1, first - a.context), min(len(rows), first + 2) + 1):
            x = rows.get(seq)
            if x:
                mark = ">>" if seq == first else "  "
                print(
                    f" {mark} {seq} {x['t']:.3f} {x['op']} g=[{x['group']}] {x['shape']}{'=' + x['values'] if x['values'] and seq >= first - 2 else ''} @ {x['caller']}"
                )


if __name__ == "__main__":
    main()
