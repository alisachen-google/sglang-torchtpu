#!/usr/bin/env python3
"""Static prefix-reuse workload: a fixed request list, no sessions and no growing history.

K distinct prefixes (prompts/gpt-oss/<prefix>.jsonl with a seeded salt). Every request is
prefix_i + a unique new block + a question and generates OUT tokens (greedy, ignore_eos).
Each prefix appears VISITS times; the whole list is shuffled once with the seed, so the before
and after runs send the identical requests in the identical order. A prefix's later visits hit
only if some tier still holds it: the prefill (device/host) or, with the victim cache, the decode.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spill_bench import generate, load_ids, scrape  # noqa: E402


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
    ap.add_argument("--prefill-url", default="http://10.128.15.217:30000")
    ap.add_argument("--decode-url", default="http://10.128.0.90:30000")
    ap.add_argument(
        "--prompts", type=Path, default=Path.home() / "pd_disagg/prompts/gpt-oss"
    )
    ap.add_argument(
        "--tokenizer",
        default="/mnt/data/tt/hf-cache/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
    )
    ap.add_argument("--prefixes", type=int, required=True)
    ap.add_argument("--visits", type=int, default=3)
    ap.add_argument("--prefix", type=int, default=131072)
    ap.add_argument("--new", type=int, default=8192)
    ap.add_argument("--out-tokens", type=int, default=1024)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--run", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731
    sizes = sorted(int(f.stem) for f in a.prompts.glob("*.jsonl") if f.stem.isdigit())
    prefix_ids = load_ids(
        tok,
        a.prompts / f"{next((n for n in sizes if n >= a.prefix), sizes[-1])}.jsonl",
        a.prefix,
    )
    rows = [json.loads(l) for l in (a.prompts / "32768.jsonl").open()]
    pool = enc(rows[1 % len(rows)]["text"])
    while len(pool) < 2 * a.new + 64:
        pool += pool
    q = enc(
        "\n\nBased on everything above, what should the agent do next? Answer briefly."
    )
    prefixes = [
        enc(f"Static document s{a.seed}-{i:04d}\n\n") + prefix_ids
        for i in range(a.prefixes)
    ]

    def new_block(i, v):
        off = (i * 7919 + v * 104729) % max(1, len(pool) - a.new)
        return enc(f"\n\n[request {i}.{v}]\n") + pool[off : off + a.new]

    reqs = [(i, v) for i in range(a.prefixes) for v in range(a.visits)]
    random.Random(a.seed).shuffle(reqs)
    # Each prefix's first appearance in the shuffled list is its cold visit.
    seen, order = set(), []
    for i, v in reqs:
        order.append((i, v, i not in seen))
        seen.add(i)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    recs = []

    def one(pos_item):
        pos, (i, v, first) = pos_item
        prompt = prefixes[i] + new_block(i, v) + q
        try:
            res, err = generate(a.lb, prompt, a.out_tokens, a.timeout), None
        except Exception as exc:  # noqa: BLE001
            res, err = {}, repr(exc)[:200]
        rec = {
            "run": a.run,
            "label": a.label,
            "pos": pos,
            "prefix_id": i,
            "visit": v,
            "first_visit": first,
            "prompt_tokens": len(prompt),
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
            recs.append(rec)
            with a.out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    pre_p, pre_d = scrape(a.prefill_url), scrape(a.decode_url)
    t0 = time.time()
    with ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(one, list(enumerate(order))))
    wall = time.time() - t0
    post_p, post_d = scrape(a.prefill_url), scrape(a.decode_url)

    def d(post, pre, k):
        return (post.get(k) or 0) - (pre.get(k) or 0)

    ok = [r for r in recs if not r["error"]]
    rep = [r for r in ok if not r["first_visit"]]
    pt = sum(r["prompt_tokens"] for r in ok)
    out = sum(r.get("completion_tokens") or 0 for r in ok)
    p_prompt, p_cached = (
        d(post_p, pre_p, "sglang:prompt_tokens_total"),
        d(post_p, pre_p, "sglang:cached_tokens_total"),
    )
    summary = {
        "run": a.run,
        "label": a.label,
        "prefixes": a.prefixes,
        "visits": a.visits,
        "prefix": a.prefix,
        "new": a.new,
        "out_tokens": a.out_tokens,
        "concurrency": a.concurrency,
        "seed": a.seed,
        "unique_prefix_tokens": sum(len(p) for p in prefixes),
        "requests": len(recs),
        "errors": len(recs) - len(ok),
        "wall_s": round(wall, 1),
        "prefill_hit_rate": round(p_cached / p_prompt, 4) if p_prompt else None,
        "prefill_cached_tokens": p_cached,
        "prefill_prompt_tokens": p_prompt,
        "decode_reuse_rate": round(sum(r.get("cached_tokens") or 0 for r in ok) / pt, 4)
        if pt
        else None,
        "ttft_p50": pct([r["ttft_s"] for r in ok], 0.5),
        "ttft_p95": pct([r["ttft_s"] for r in ok], 0.95),
        "repeat_ttft_p50": pct([r["ttft_s"] for r in rep], 0.5),
        "repeat_ttft_p95": pct([r["ttft_s"] for r in rep], 0.95),
        "first_ttft_p50": pct([r["ttft_s"] for r in ok if r["first_visit"]], 0.5),
        "e2e_p50": pct([r["e2e_s"] for r in ok], 0.5),
        "e2e_p95": pct([r["e2e_s"] for r in ok], 0.95),
        "total_tok_s": round((pt + out) / wall, 1) if wall else None,
        "output_tok_s": round(out / wall, 1) if wall else None,
    }
    with a.out.with_suffix(".summary.json").open("w") as f:
        json.dump(summary, f, indent=1)
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
