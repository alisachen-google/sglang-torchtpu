#!/usr/bin/env python3
"""Data-integrity check of the P/D victim cache under concurrent D2D and H2H transfers.

Greedy decoding, static prompts. Phases:
  ref     : probe prompts (fresh, unique) sent one at a time on a quiet pair -> reference token ids
  fill    : N long sessions (prefix + new block) -> prefill overflows -> victim pushes (H2H)
  settle  : wait, probes continue
  recall  : the sessions oldest first -> pulls from decode (D2D reverse); outputs recorded
  verify  : flush both engines, recompute the recall prompts one at a time -> must equal recall outputs
A background thread re-sends the probes (normal P/D, D2D) during fill/settle/recall and compares each
output with its reference. Output: JSON summary with match statistics; records in --out.
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

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spill_bench import load_ids  # noqa: E402


def gen(lb, ids, n, timeout=1800):
    body = {
        "input_ids": ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": n, "ignore_eos": True},
        "return_text_in_logprobs": False,
        "stream": False,
    }
    t0 = time.time()
    r = requests.post(lb.rstrip("/") + "/generate", json=body, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    j = j[0] if isinstance(j, list) else j
    meta = j.get("meta_info", {})
    out = j.get("output_ids") or meta.get("output_ids")
    return {
        "out": list(out) if out is not None else None,
        "text": j.get("text", ""),
        "e2e_s": time.time() - t0,
        "cached_tokens": meta.get("cached_tokens"),
        "details": meta.get("cached_tokens_details"),
    }


def agree(a, b):
    """Leading tokens in common (or text characters when ids are missing)."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lb", required=True)
    ap.add_argument(
        "--flush",
        nargs="+",
        required=True,
        help="engine base urls to flush before verify",
    )
    ap.add_argument(
        "--prompts", type=Path, default=Path.home() / "pd_disagg/prompts/gpt-oss"
    )
    ap.add_argument(
        "--tokenizer",
        default="/mnt/data/tt/hf-cache/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
    )
    ap.add_argument("--sessions", type=int, default=20)
    ap.add_argument("--prefix", type=int, default=131072)
    ap.add_argument("--new", type=int, default=8192)
    ap.add_argument("--out-tokens", type=int, default=64)
    ap.add_argument("--probes", type=int, default=6)
    ap.add_argument("--probe-len", type=int, default=8192)
    ap.add_argument("--probe-out", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--settle", type=float, default=90)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]  # noqa: E731
    sizes = sorted(int(f.stem) for f in a.prompts.glob("*.jsonl") if f.stem.isdigit())
    prefix = load_ids(
        tok, a.prompts / f"{next(n for n in sizes if n >= a.prefix)}.jsonl", a.prefix
    )
    rows = [json.loads(l) for l in (a.prompts / "32768.jsonl").open()]
    pool = []
    for r in rows:
        pool += enc(r["text"])
        if len(pool) > 4 * (a.new + a.probe_len) + 4096:
            break
    q = enc(
        "\n\nBased on everything above, what should the agent do next? Answer briefly."
    )

    def block(i, k, n):
        off = (i * 7919 + k * 104729) % max(1, len(pool) - n)
        return enc(f"\n\n[tool result {k}]\n") + pool[off : off + n]

    probes = [
        enc(f"Probe {a.seed}-{i}: summarize.\n") + block(1000 + i, 9, a.probe_len) + q
        for i in range(a.probes)
    ]
    sess = {
        s: enc(f"Integrity session s{a.seed}-{s:04d}\n\n")
        + prefix
        + block(s, 0, a.new)
        + q
        for s in range(a.sessions)
    }
    recall = {
        s: sess[s] + enc(" Noted. Continuing.") + block(s, 1, a.new) + q
        for s in range(a.sessions)
    }
    recs = []
    lock = threading.Lock()

    def log(rec):
        with lock:
            recs.append(rec)
            with a.out.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"ref: {a.probes} probes", flush=True)
    ref = [gen(a.lb, p, a.probe_out)["out"] for p in probes]
    if any(r is None for r in ref):
        sys.exit("server did not return output_ids; cannot compare")
    stop = threading.Event()
    probe_stats = {
        "sent": 0,
        "exact": 0,
        "min_agree": a.probe_out,
        "errors": 0,
        "mismatch_positions": [],
    }

    def prober():
        i = 0
        while not stop.is_set():
            p = i % a.probes
            try:
                r = gen(a.lb, probes[p], a.probe_out)
                n = agree(r["out"], ref[p])
                with lock:
                    probe_stats["sent"] += 1
                    probe_stats["exact"] += int(n == len(ref[p]))
                    probe_stats["min_agree"] = min(probe_stats["min_agree"], n)
                    if n < len(ref[p]):
                        probe_stats["mismatch_positions"].append(n)
                log(
                    {
                        "phase": "probe",
                        "probe": p,
                        "agree": n,
                        "e2e_s": round(r["e2e_s"], 2),
                        "t": time.time(),
                    }
                )
            except Exception as e:  # noqa: BLE001
                with lock:
                    probe_stats["errors"] += 1
                log(
                    {
                        "phase": "probe",
                        "probe": p,
                        "error": repr(e)[:200],
                        "t": time.time(),
                    }
                )
            i += 1

    th = threading.Thread(target=prober, daemon=True)
    th.start()

    def run(phase, prompts):
        res = {}

        def one(s):
            try:
                r = gen(a.lb, prompts[s], a.out_tokens)
                res[s] = r
                log(
                    {
                        "phase": phase,
                        "session": s,
                        "cached": r["cached_tokens"],
                        "details": r["details"],
                        "e2e_s": round(r["e2e_s"], 2),
                        "out": r["out"],
                        "t": time.time(),
                    }
                )
            except Exception as e:  # noqa: BLE001
                log(
                    {
                        "phase": phase,
                        "session": s,
                        "error": repr(e)[:200],
                        "t": time.time(),
                    }
                )

        with ThreadPoolExecutor(a.concurrency) as ex:
            list(ex.map(one, range(a.sessions)))
        return res

    t0 = time.time()
    print("fill", flush=True)
    run("fill", sess)
    print("settle", flush=True)
    time.sleep(a.settle)
    print("recall", flush=True)
    rec = run("recall", recall)
    stop.set()
    th.join(timeout=600)
    t_conc = time.time() - t0
    print("verify: flush + recompute", flush=True)
    for u in a.flush:
        for _ in range(30):
            if (
                requests.post(u.rstrip("/") + "/flush_cache", timeout=300).status_code
                == 200
            ):
                break
            time.sleep(10)
    ver = {}
    for s in range(a.sessions):
        r = gen(a.lb, recall[s], a.out_tokens)
        ver[s] = r
        log(
            {
                "phase": "verify",
                "session": s,
                "cached": r["cached_tokens"],
                "out": r["out"],
                "t": time.time(),
            }
        )
    cmp = []
    for s in range(a.sessions):
        if s in rec and rec[s]["out"] is not None and ver[s]["out"] is not None:
            n = agree(rec[s]["out"], ver[s]["out"])
            cmp.append(
                {
                    "session": s,
                    "agree": n,
                    "recall_cached": rec[s]["cached_tokens"],
                    "recall_details": rec[s]["details"],
                }
            )
    summary = {
        "sessions": a.sessions,
        "prefix": a.prefix,
        "new": a.new,
        "out_tokens": a.out_tokens,
        "concurrent_phase_s": round(t_conc, 1),
        "probes": {
            **probe_stats,
            "mismatch_positions": sorted(probe_stats["mismatch_positions"])[:50],
        },
        "recall_vs_recompute": {
            "compared": len(cmp),
            "exact": sum(1 for c in cmp if c["agree"] == a.out_tokens),
            "min_agree": min((c["agree"] for c in cmp), default=None),
            "pulled_sessions": sum(
                1 for c in cmp if (c["recall_cached"] or 0) >= a.prefix
            ),
            "per_session": cmp,
        },
        "errors": sum(1 for r in recs if r.get("error")),
    }
    with a.out.with_suffix(".summary.json").open("w") as f:
        json.dump(summary, f, indent=1)
    print(
        "SUMMARY "
        + json.dumps(
            {k: v for k, v in summary.items() if k != "recall_vs_recompute"}
            | {
                "recall": {
                    k: v
                    for k, v in summary["recall_vs_recompute"].items()
                    if k != "per_session"
                }
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
