#!/usr/bin/env python3
"""Tables for the victim-cache benchmark from the phase logs and summary files."""

import glob
import json
import re
import sys
from pathlib import Path

R = Path.home() / "pd_disagg/spill/results"
L = Path.home() / "pd_disagg/spill/logs"
runs = sys.argv[1:] or ["victim_1", "victim_base_1"]


def summaries(log):
    out = []
    for line in open(log):
        if line.startswith("SUMMARY "):
            out.append(json.loads(line[8:]))
    return out


def w1(run):
    rows = []
    p = R / f"{run}.jsonl"
    if not p.exists():
        return rows
    for line in open(p):
        r = json.loads(line)
        if r.get("label", "").startswith("recall"):
            t1 = (r.get("turn1") or {}).get("ttft_s")
            t2 = (r.get("turn2") or {}).get("ttft_s")
            d = (r.get("turn2") or {}).get("cached_tokens_details") or {}
            rows.append((r.get("prefix"), r.get("new"), t1, t2, d))
    return rows


print("## W1 single turn (turn 2 after a prefill-only flush)")
print("| run | prefix | new | turn-1 cold TTFT | turn-2 TTFT | turn-2 cached tiers |")
print("|---|---:|---:|---:|---:|---|")
for run in runs:
    for pfx, new, t1, t2, d in w1(run):
        print(
            f"| {run} | {pfx} | {new} | {t1:.2f} s | {t2:.2f} s | {d} |"
            if t1 and t2
            else f"| {run} | {pfx} | {new} | {t1} | {t2} | {d} |"
        )

print("\n## Capacity ladder (128k sessions x 4 turns)")
print(
    "| run | sessions | wall s | output tok/s | later TTFT p50 | p95 | errors | cached by tier |"
)
print("|---|---:|---:|---:|---:|---:|---:|---|")
for run in runs:
    log = L / f"{run}_phase.log"
    if not log.exists():
        continue
    for s in summaries(log):
        if (
            "sessions" in s
            and s.get("label", "").endswith(f"n{s['sessions']}")
            and "turn_new" in s
        ):
            print(
                f"| {run} | {s['sessions']} | {s['wall_s']} | {s['aggregate_output_tok_s']} | {s['ttft_later_p50']:.1f} | {s['ttft_later_p95']:.1f} | {s['errors']} | {s['cached_tokens_by_tier']} |"
            )

print(
    "\n## Session sweep (4k unique prefix, 4 turns x 2 rounds, 64-token questions, 128-token answers; passes 2-8 measured)"
)
print(
    "| run | sessions | token hit | request hit | round-2 token hit | total tok/s | output tok/s | TTFT p50 | TTFT p95 | e2e p50 | measured wall s | cold pass s | failures |"
)
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for run in runs:
    for f in sorted(
        glob.glob(str(R / f"{run}_sweep.n*.summary.json")),
        key=lambda x: int(re.search(r"\.n(\d+)\.", x).group(1)),
    ):
        s = json.load(open(f))
        print(
            f"| {run} | {s['sessions']} | {s['token_hit_rate']:.1%} | {s['request_hit_rate']:.1%} | {s['round2_token_hit_rate']:.1%} | {s['total_tok_s']} | {s['output_tok_s']} | {s['ttft_p50']:.2f} | {s['ttft_p95']:.2f} | {s['e2e_p50']:.2f} | {s['wall_s_measured']} | {s['cold_pass_wall_s']} | {s['failures_total']} |"
        )

print("\n## Session sweep, per pass")
for run in runs:
    for f in sorted(
        glob.glob(str(R / f"{run}_sweep.n*.summary.json")),
        key=lambda x: int(re.search(r"\.n(\d+)\.", x).group(1)),
    ):
        s = json.load(open(f))
        print(
            f"\n{run}, {s['sessions']} sessions: pass | round | turn | wall s | token hit | req hit | total tok/s | TTFT p50 | failed"
        )
        for p in s["passes"]:
            th = (
                f"{p['token_hit_rate']:.1%}" if p["token_hit_rate"] is not None else "-"
            )
            rh = (
                f"{p['request_hit_rate']:.1%}"
                if p["request_hit_rate"] is not None
                else "-"
            )
            print(
                f"  {p['pass']} | {p['round']} | {p['turn']} | {p['wall_s']} | {th} | {rh} | {p['total_tok_s']} | {p['ttft_p50']:.2f} | {p['failed']}"
            )

print("\n## Feature counters after each step (victim_stats.sh lines)")
for run in runs:
    log = L / f"{run}_phase.log"
    if not log.exists():
        continue
    print(f"\n{run}:")
    for line in open(log):
        if (
            re.match(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d (W1|ladder|sweep|stress)", line)
            or line.startswith("  prefill push:")
            or line.startswith("  decode push:")
            or line.startswith("  prefix-pull")
        ):
            print("  " + line.rstrip()[:220])
