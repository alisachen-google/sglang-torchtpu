#!/usr/bin/env python3
"""Victim-recall workload: sessions the prefill must evict but the pair can keep.

Static prompts (prompts/gpt-oss/<prefix>.jsonl, seeded per-session salt, fixed appended answer), so the
before/after runs send identical token ids.

  fill   : sessions 0..N-1 in order, prompt = salt + prefix + new block + question, OUT tokens.
           With N * (prefix + new) above the prefill's device+host capacity the oldest sessions are
           evicted from the prefill (pushed to decode RAM when the victim cache is on).
  recall : the same sessions again, oldest first (worst case for the prefill LRU),
           prompt = fill prompt + fixed answer + another new block + question, OUT tokens.
           A session whose prefix came back from the decode shows as cached tokens on the recall.

Writes one JSON record per request and a summary (per phase and per recall position bucket).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spill_bench import generate, load_ids  # noqa: E402


def pct(v, p):
    if not v:
        return None
    s = sorted(v)
    return s[max(0, min(len(s) - 1, int(round((len(s) - 1) * p))))]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--lb", default="http://10.128.15.217:8080")
    ap.add_argument(
        "--prompts", type=Path, default=Path.home() / "pd_disagg/prompts/gpt-oss"
    )
    ap.add_argument(
        "--tokenizer",
        default="/mnt/data/tt/hf-cache/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
    )
    ap.add_argument("--sessions", type=int, required=True)
    ap.add_argument("--prefix", type=int, default=131072)
    ap.add_argument("--new", type=int, default=8192)
    ap.add_argument("--out-tokens", type=int, default=1024)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--phase", choices=["fill", "recall", "both"], default="both")
    ap.add_argument(
        "--settle",
        type=float,
        default=0,
        help="seconds between fill and recall (phase=both)",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--order",
        choices=["oldest", "newest", "random"],
        default="oldest",
        help="recall order",
    )
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--run", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731
    sizes = sorted(
        int(f.stem) for f in args.prompts.glob("*.jsonl") if f.stem.isdigit()
    )
    src = (
        args.prompts
        / f"{next((n for n in sizes if n >= args.prefix), sizes[-1])}.jsonl"
    )
    prefix_ids = load_ids(tok, src, args.prefix)
    tool_rows = [json.loads(l) for l in (args.prompts / "32768.jsonl").open()]
    tool_pool = enc(tool_rows[1 % len(tool_rows)]["text"])
    while len(tool_pool) < 2 * args.new + 64:
        tool_pool += tool_pool
    answer_ids = load_ids(tok, args.prompts / "1024.jsonl", 64)
    q_ids = enc(
        "\n\nBased on everything above, what should the agent do next? Answer briefly."
    )

    def block(sid, k):  # deterministic per (session, turn) slice of the tool pool
        off = (sid * 7919 + k * 104729) % max(1, len(tool_pool) - args.new)
        return enc(f"\n\n[tool result {k}]\n") + tool_pool[off : off + args.new]

    ctx = {
        s: enc(f"Agent session s{args.seed}-{s:04d} user {s}\n\n") + prefix_ids
        for s in range(args.sessions)
    }
    prompts = {
        "fill": {s: ctx[s] + block(s, 0) + q_ids for s in range(args.sessions)},
    }
    prompts["recall"] = {
        s: prompts["fill"][s] + answer_ids + block(s, 1) + q_ids
        for s in range(args.sessions)
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    records = []

    def one(phase, pos, s):
        p = prompts[phase][s]
        t0 = time.time()
        try:
            res, err = generate(args.lb, p, args.out_tokens, args.timeout), None
        except Exception as exc:  # noqa: BLE001
            res, err = {}, repr(exc)[:200]
        rec = {
            "run": args.run,
            "label": args.label,
            "phase": phase,
            "pos": pos,
            "session": s,
            "prompt_tokens": len(p),
            "reusable_tokens": len(prompts["fill"][s]) if phase == "recall" else 0,
            "t_start": round(t0, 3),
            "error": err,
            **{
                k: res.get(k)
                for k in (
                    "ttft_s",
                    "e2e_s",
                    "cached_tokens",
                    "cached_tokens_details",
                    "completion_tokens",
                )
            },
        }
        with lock:
            records.append(rec)
            with args.out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        print(
            json.dumps(
                {
                    k: rec[k]
                    for k in (
                        "phase",
                        "pos",
                        "session",
                        "prompt_tokens",
                        "ttft_s",
                        "cached_tokens",
                        "error",
                    )
                }
            ),
            flush=True,
        )
        return rec

    def run_phase(phase):
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            seq = list(range(args.sessions))
            if phase == "recall" and args.order == "newest":
                seq.reverse()
            elif phase == "recall" and args.order == "random":
                import random as _r

                _r.Random(args.seed).shuffle(seq)
            list(ex.map(lambda ps: one(phase, ps[0], ps[1]), list(enumerate(seq))))
        return time.time() - t0

    walls = {}
    phases = ["fill", "recall"] if args.phase == "both" else [args.phase]
    for i, ph in enumerate(phases):
        if i and args.settle:
            time.sleep(args.settle)
        walls[ph] = round(run_phase(ph), 1)

    def summ(recs, wall):
        ok = [r for r in recs if not r["error"]]
        pt = sum(r["prompt_tokens"] for r in ok)
        ct = sum(r.get("cached_tokens") or 0 for r in ok)
        out = sum(r.get("completion_tokens") or 0 for r in ok)
        reuse = sum(r["reusable_tokens"] for r in ok)
        tiers = {"device": 0, "host": 0, "storage": 0}
        for r in ok:
            for k in tiers:
                tiers[k] += (r.get("cached_tokens_details") or {}).get(k) or 0
        full = [
            r
            for r in ok
            if r["reusable_tokens"]
            and (r.get("cached_tokens") or 0) >= 0.9 * r["reusable_tokens"]
        ]
        return {
            "requests": len(recs),
            "errors": len(recs) - len(ok),
            "wall_s": wall,
            "prompt_tokens": pt,
            "cached_tokens": ct,
            "token_hit_rate": round(ct / pt, 4) if pt else None,
            "reusable_tokens": reuse,
            "reuse_recovered": round(ct / reuse, 4) if reuse else None,
            "sessions_fully_reused": len(full),
            "ttft_p50": pct([r["ttft_s"] for r in ok], 0.5),
            "ttft_p95": pct([r["ttft_s"] for r in ok], 0.95),
            "e2e_p50": pct([r["e2e_s"] for r in ok], 0.5),
            "e2e_p95": pct([r["e2e_s"] for r in ok], 0.95),
            "total_tok_s": round((pt + out) / wall, 1) if wall else None,
            "output_tok_s": round(out / wall, 1) if wall else None,
            "cached_tokens_by_tier": tiers,
        }

    summary = {
        "run": args.run,
        "label": args.label,
        "sessions": args.sessions,
        "prefix": args.prefix,
        "new": args.new,
        "out_tokens": args.out_tokens,
        "concurrency": args.concurrency,
        "seed": args.seed,
        "order": args.order,
        "working_set_tokens": sum(
            len(prompts["fill"][s]) for s in range(args.sessions)
        ),
    }
    for ph in phases:
        recs = [r for r in records if r["phase"] == ph]
        summary[ph] = summ(recs, walls[ph])
        if (
            ph == "recall"
        ):  # oldest-first quarters: Q1 = sessions the prefill evicted first
            q = max(1, args.sessions // 4)
            summary["recall_by_quarter"] = [
                {
                    "sessions": f"{i}-{min(args.sessions, i + q) - 1}",
                    **{
                        k: v
                        for k, v in summ(
                            [r for r in recs if i <= r["pos"] < i + q], walls[ph]
                        ).items()
                        if k not in ("wall_s", "total_tok_s", "output_tok_s")
                    },
                }
                for i in range(0, args.sessions, q)
            ]
    with args.out.with_suffix(".summary.json").open("w") as f:
        json.dump(summary, f, indent=1)
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
