#!/usr/bin/env python3
"""Semantic check for outputs produced from build_prompts.py prompt sets.

Each prompt embeds needle code words at 10/50/95 % depth and asks the model to list them first.
This checks, per output, which needle words appear in the generated text, plus the divergence point
between two output files (e.g. golden vs PD) measured only over the needle-answer region.

Usage: needle_check.py --prompts prompts/<model>/<isl>.jsonl --out outputs.jsonl [--other golden.jsonl] [--report r.json]
Exit 1 if any output misses a needle (within native context, caller decides) or has an error.
"""

import argparse
import json
import sys


def load(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p) if l.strip()}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--prompts", required=True)
    ap.add_argument(
        "--out", required=True, help="outputs jsonl from golden_diff.py generate"
    )
    ap.add_argument(
        "--other", help="second outputs file to compare needle answers with"
    )
    ap.add_argument("--report")
    ap.add_argument(
        "--no-fail", action="store_true", help="report only (beyond native context)"
    )
    a = ap.parse_args()
    prompts, outs = load(a.prompts), load(a.out)
    other = load(a.other) if a.other else {}
    rows, bad = [], 0
    for pid, p in prompts.items():
        o = outs.get(pid)
        needles = list(p.get("needles", {}).values())
        if not o or o.get("error"):
            rows.append(
                {
                    "id": pid,
                    "status": "ERROR",
                    "error": (o or {}).get("error", "missing"),
                }
            )
            bad += 1
            continue
        text = o.get("text") or ""
        found = [w for w in needles if w in text]
        first_pos = [text.find(w) for w in needles if w in text]
        row = {
            "id": pid,
            "isl": p.get("isl"),
            "needles": len(needles),
            "found": len(found),
            "missing": [w for w in needles if w not in text],
            "answer_region_chars": max(first_pos) + 16 if first_pos else None,
            "completion_tokens": o.get("completion_tokens"),
            "finish_reason": o.get("finish_reason"),
        }
        if pid in other:
            t2 = other[pid].get("text") or ""
            found2 = [w for w in needles if w in t2]
            row["other_found"] = len(found2)
            row["needle_order_identical"] = [
                w for w in needles if w in text and text.find(w) >= 0
            ] == [w for w in needles if w in t2]
            # common prefix in characters
            n = 0
            for x, y in zip(text, t2):
                if x != y:
                    break
                n += 1
            row["common_prefix_chars"] = n
        rows.append(row)
        if len(found) < len(needles):
            bad += 1
    for r in rows:
        print(json.dumps(r))
    print(
        f"needle_check: {len(rows)} outputs, {bad} with missing needles/errors",
        file=sys.stderr,
    )
    if a.report:
        json.dump(rows, open(a.report, "w"), indent=1)
    sys.exit(0 if (bad == 0 or a.no_fail) else 1)


if __name__ == "__main__":
    main()
