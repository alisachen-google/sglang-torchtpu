#!/usr/bin/env python3
"""Join [victim-crc] (pushes) and [recall-crc] (recalls) lines of a pair-1 run:
prefill log on host a (over ssh), decode log local. Prints equal/mismatched counts."""

import re
import subprocess
import sys

run = sys.argv[1]
me = "alisachen_google_com"
ssh = [
    "ssh",
    "-i",
    f"/home/{me}/.ssh/google_compute_engine",
    "-o",
    "BatchMode=yes",
    "sa_101411256518727260534@10.128.15.217",
]
P = f"/home/{me}/pd_disagg/results/runs/{run}/prefill/server.log"
D = f"/home/{me}/pd_disagg/results/runs/{run}/decode/server.log"


def _fetch_prefill():
    """Fetch the prefill's CRC lines; retry until the count matches the remote count."""
    for _ in range(4):
        want = subprocess.run(
            ssh + [f"sudo grep -c 'crc] role=prefill' {P}"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        text = subprocess.run(
            ssh + [f"sudo grep -h 'crc] role=prefill' {P}"],
            capture_output=True,
            text=True,
        ).stdout
        if want.isdigit() and text.count("\n") >= int(want) - 1:
            return text
    return text


pre = _fetch_prefill()
dec = open(D).read()


def load(text, role):
    out = {}
    for m in re.finditer(r"\[(victim|recall)-crc\] role=%s (.*)" % role, text):
        kind, rest = m.group(1), m.group(2)
        kv = dict(x.split("=", 1) for x in rest.split() if "=" in x)
        if not {"comp", "rank", "pages", "bytes", "crc"} <= kv.keys():
            continue  # a line garbled by concurrent rank writes
        key = (kind, kv.get("seq") or kv.get("room"), kv["comp"], kv["rank"])
        out[key] = (int(kv["pages"]), int(kv["bytes"]), kv["crc"])
    return out


p, d = load(pre, "prefill"), load(dec, "decode")
both = sorted(set(p) & set(d))
match = sum(1 for k in both if p[k] == d[k])
bad = [(k, p[k], d[k]) for k in both if p[k] != d[k]]
print(
    f"prefill records {len(p)}, decode records {len(d)}, joined {len(both)}, equal {match}, mismatched {len(bad)}"
)
for k, a, b in bad[:12]:
    print("MISMATCH", k, "prefill", a, "decode", b)
kinds = {}
for k in both:
    kinds.setdefault((k[0], k[2]), [0, 0])
    kinds[(k[0], k[2])][0] += 1
    kinds[(k[0], k[2])][1] += int(p[k] == d[k])
for k, (n, eq) in sorted(kinds.items()):
    print(f"  {k[0]} {k[1]}: {eq}/{n} equal")
