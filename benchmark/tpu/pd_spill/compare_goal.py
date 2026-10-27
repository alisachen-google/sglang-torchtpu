#!/usr/bin/env python3
"""Before vs after markdown tables from the goal runs (logs/goal_before_1.log, logs/goal_after_1.log).

Prefill hit = prefill /metrics delta over the measured phase (cross-checked against the prefill
log's per-batch #cached-token sum). Latency/throughput from the client summaries.
"""

import json
import re
import sys
from pathlib import Path

L = Path.home() / "pd_disagg/spill/logs"


def load(name):
    out = {}
    p = L / f"goal_{name}_1.log"
    if p.exists():
        for line in p.open():
            if line.startswith("POINT "):
                r = json.loads(line[6:])
                out[r["label"]] = r
    return out


def f(x, fmt):
    return "-" if x is None else fmt.format(x)


def pair(x, y, fmt):
    return f"{f(x, fmt)} → {f(y, fmt)}"


def ratio(x, y):
    return "-" if not x or not y else f"{x / y:.2f}x"


def counters(r):
    v = (r or {}).get("victim", "")
    g = lambda pat: int(re.search(pat, v)[1]) if re.search(pat, v) else 0  # noqa: E731
    return {
        "stored": g(r"stored=(\d+)"),
        "landed": g(r"landed_tokens=(\d+)"),
        "pulls": g(r"prefix-pull commits \(TP0\): (\d+)"),
        "pulled": g(r"prefix-pull tokens \(TP0\): (\d+)"),
    }


def per_point(side):
    """Cumulative victim counters -> per-point deltas (a pair restart resets them to zero)."""
    prev = {"stored": 0, "landed": 0, "pulls": 0, "pulled": 0}
    out = {}
    for lab, r in side.items():  # insertion order = run order
        c = counters(r)
        d = {k: c[k] - prev[k] if c[k] >= prev[k] else c[k] for k in c}
        prev = c
        out[lab] = d
    return out


def hit_gain(x, y):
    """Absolute gain in percentage points and the relative factor."""
    if x is None or y is None:
        return "-"
    pts = (y - x) * 100
    rel = "-" if x <= 0 else f"{y / x:.2f}x"
    return f"{pts:+.1f} pts ({rel})"


def pull_cols(r, d):
    if r is None or d is None:
        return "- | - | - | -"
    pp = r.get("prefill_prompt") or 0
    share = f"{d['pulled'] / pp:.1%}" if pp else "-"
    return f"{d['landed'] / 1e6:.2f}M ({d['stored']} pushes) | {d['pulled'] / 1e6:.2f}M | {share} | {d['pulls']}"


def main():
    # usage: compare_goal.py [before_name] [after_name]   (log names goal_<name>_1.log)
    names = sys.argv[1:3] if len(sys.argv) >= 3 else ("before", "after")
    b, a = load(names[0]), load(names[1])
    ad = per_point(a)
    labs = sorted(
        set(b) | set(a),
        key=lambda l: (
            l.split("_")[0],
            int(l.split("_p")[1].split("k")[0]),
            int(l.rsplit("_", 1)[1][1:]),
        ),
    )
    for kind, title, unit in (
        (
            "recall",
            "Recall workload (fill N sessions, recall oldest first; recall phase)",
            "Sessions",
        ),
        (
            "static",
            "Static prefix-reuse workload (K documents x 3 visits, shuffled)",
            "Docs",
        ),
    ):
        print(f"### {title}\n")
        print(
            f"| Prefix+new | {unit} | Prefill hit before → after | Hit gain | TTFT p50 before → after | TTFT speedup | E2E p50 before → after | Total tok/s before → after | Throughput gain | KV offloaded to decode DRAM (after) | KV pulled from decode DRAM (after) | Pulled share of prompt | Pulls (after) |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for lab in labs:
            if not lab.startswith(kind):
                continue
            x, y = b.get(lab), a.get(lab)
            m = re.match(r"\w+_p(\d+)k_n(\d+)k_[sk](\d+)", lab)
            gx, gy = (x or {}), (y or {})
            print(
                f"| {m[1]}k+{m[2]}k | {m[3]} | {pair(gx.get('prefill_hit'), gy.get('prefill_hit'), '{:.1%}')} | {hit_gain(gx.get('prefill_hit'), gy.get('prefill_hit'))} | "
                f"{pair(gx.get('ttft50'), gy.get('ttft50'), '{:.1f} s')} | {ratio(gx.get('ttft50'), gy.get('ttft50'))} | "
                f"{pair(gx.get('e2e50'), gy.get('e2e50'), '{:.1f} s')} | {pair(gx.get('tok_s'), gy.get('tok_s'), '{:,.0f}')} | "
                f"{ratio(gy.get('tok_s'), gx.get('tok_s'))} | {pull_cols(y, ad.get(lab))} |"
            )
        print()
    # cumulative victim counters are per run; note it
    print(
        f"_Before points: {len(b)}; after points: {len(a)}. Offloaded tokens = victim-push tokens that landed in the decode's host pool during the point (decode landed_tokens deltas); pulled tokens = prefix-pull tokens the prefill received from the decode during the point (TP0 counter deltas); pulled share = those tokens / the point's measured-phase prompt tokens (for recall points the pulls of the fill phase are included, so the share can exceed 100%)._"
    )


if __name__ == "__main__":
    main()
