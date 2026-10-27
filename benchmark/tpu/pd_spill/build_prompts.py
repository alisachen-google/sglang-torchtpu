#!/usr/bin/env python3
"""Build token-exact prompt sets for golden vs PD differential tests.

Each output line: {"id", "text", "isl" (exact token count under the model tokenizer),
"needles": {depth: word}, "question", "max_new_tokens"}.
The text is deterministic filler prose (seeded) with needle sentences inserted at the
requested depths and a final retrieval question, trimmed/padded to the exact token count.

Usage:
  build_prompts.py --model-path /mnt/data/.../snapshots/<sha> --out prompts/gpt-oss \
      --lengths 1024,8192,32768,131072,262144,524288,1047552 --unaligned 127999,255123,511777,999999 \
      --per-length 2
"""

import argparse
import hashlib
import json
import os
import random
import sys

FILLER = [
    "The quarterly report summarised regional sales, operating costs, and staffing changes across all divisions.",
    "A light rain fell over the harbour while the fishing boats returned with their nets and crates.",
    "The committee postponed the vote until every member had reviewed the amended proposal in detail.",
    "Engineers replaced the worn bearings, recalibrated the sensors, and logged each step for the audit.",
    "The library extended its opening hours during the examination period to accommodate students.",
    "Migrating birds followed the river valley south, resting briefly on the flooded fields near the town.",
    "The recipe calls for two cups of flour, a pinch of salt, and butter softened at room temperature.",
    "Historians disagree about the exact date, but most place the treaty in the late autumn of that year.",
    "The orchestra rehearsed the second movement twice before the conductor was satisfied with the tempo.",
    "New signage along the trail marks the distance to the summit and warns hikers about loose rock.",
    "The museum's east wing houses ceramics, textiles, and a small collection of early navigational charts.",
    "Volunteers planted native grasses along the embankment to slow erosion after the spring floods.",
]
WORDS = [
    "lantern",
    "copper",
    "meadow",
    "falcon",
    "harbor",
    "velvet",
    "quartz",
    "saffron",
    "thistle",
    "ember",
    "glacier",
    "marble",
    "orchid",
    "pepper",
    "raven",
    "sable",
    "tundra",
    "walnut",
    "zephyr",
    "cobalt",
]


def needle_word(seed, depth):
    h = hashlib.sha256(f"{seed}-{depth}".encode()).hexdigest()
    return WORDS[int(h[:8], 16) % len(WORDS)] + "-" + h[8:12]


def build_text(tok, target, depths, seed, question):
    rng = random.Random(seed)
    needles = {d: needle_word(seed, d) for d in depths}
    needle_sents = {
        d: f" The secret code word for checkpoint {i + 1} is {w}."
        for i, (d, w) in enumerate(sorted(needles.items()))
    }
    q = "\n\n" + question + "\n"
    q_toks = len(tok.encode(q, add_special_tokens=False))
    budget = (
        target
        - q_toks
        - sum(
            len(tok.encode(s, add_special_tokens=False)) for s in needle_sents.values()
        )
    )
    if budget < 32:
        raise ValueError(f"target {target} too small")
    # Generate filler by sentences until over budget, then trim at token level.
    sents = []
    est = 0
    while est < budget * 1.05 + 64:
        s = rng.choice(FILLER)
        sents.append(s)
        est += len(tok.encode(" " + s, add_special_tokens=False))
    filler_ids = tok.encode(" ".join(sents), add_special_tokens=False)[:budget]
    filler = tok.decode(filler_ids)
    # Insert needles at character positions proportional to depth.
    parts, last = [], 0
    for d in sorted(needle_sents):
        pos = int(len(filler) * d)
        pos = (
            filler.rfind(". ", 0, max(pos, 1)) + 1
            if filler.rfind(". ", 0, max(pos, 1)) > 0
            else pos
        )
        parts.append(filler[last:pos])
        parts.append(needle_sents[d])
        last = pos
    parts.append(filler[last:])
    text = "".join(parts) + q
    # Exact-length adjust (tokenisation of joined text can differ by a few tokens).
    ids = tok.encode(text, add_special_tokens=False)
    tries = 0
    while len(ids) != target and tries < 40:
        diff = len(ids) - target
        if diff > 0:  # too long: remove tokens from filler tail (before the question)
            cut = (
                tok.decode(tok.encode(parts[-1], add_special_tokens=False)[:-diff])
                if len(parts[-1]) > 0
                else ""
            )
            parts[-1] = cut
        else:  # too short: append filler words before the question
            parts[-1] = parts[-1] + " " + " ".join(["lorem"] * (-diff))
        text = "".join(parts) + q
        ids = tok.encode(text, add_special_tokens=False)
        tries += 1
    return text, len(ids), needles


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model-path", required=True, help="HF snapshot dir (tokenizer)")
    ap.add_argument(
        "--out", required=True, help="output directory; one <isl>.jsonl per length"
    )
    ap.add_argument("--lengths", default="1024,8192,32768,131072")
    ap.add_argument(
        "--unaligned", default="", help="extra exact lengths, e.g. 127999,255123"
    )
    ap.add_argument(
        "--per-length",
        type=int,
        default=2,
        help="prompts per length (different seeds/needles)",
    )
    ap.add_argument("--depths", default="0.1,0.5,0.95")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument(
        "--question",
        default="List the secret code words for every checkpoint mentioned above, in order, then continue writing a detailed story about a lighthouse keeper.",
    )
    ap.add_argument("--seed", type=int, default=20261007)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True)
    os.makedirs(a.out, exist_ok=True)
    depths = [float(x) for x in a.depths.split(",") if x]
    lengths = [int(x) for x in a.lengths.split(",") if x] + [
        int(x) for x in a.unaligned.split(",") if x
    ]
    for L in lengths:
        path = os.path.join(a.out, f"{L}.jsonl")
        with open(path, "w") as f:
            for k in range(a.per_length):
                seed = a.seed * 1000 + L % 997 + k
                text, n, needles = build_text(tok, L, depths, seed, a.question)
                rec = {
                    "id": f"isl{L}_k{k}",
                    "isl": n,
                    "target": L,
                    "needles": {str(d): w for d, w in needles.items()},
                    "question": a.question,
                    "max_new_tokens": a.max_new_tokens,
                    "text": text,
                }
                f.write(json.dumps(rec) + "\n")
                print(
                    f"{path}: id={rec['id']} tokens={n} (target {L}) exact={'yes' if n == L else 'NO'}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()
