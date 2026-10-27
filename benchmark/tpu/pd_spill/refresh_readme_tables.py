#!/usr/bin/env python3
"""Re-splice the generated comparison tables into README.md (run from the benchmark dir)."""


def rows(path, kind):
    for s in open(path).read().split("### "):
        if s.lower().startswith(kind):
            return [
                l
                for l in s.splitlines()
                if l.startswith("| ") and "→" in l and "→ -" not in l
            ]
    return []


hdr = (
    "| Prefix+new | N | Prefill hit before → after | Hit gain (pts, relative) | TTFT p50 | TTFT speedup | E2E p50 | Total tok/s | Throughput | KV offloaded to decode RAM | KV pulled from decode RAM | Pulled share of prompt | Pulls |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)
readme = open("README.md").read()
blocks = [
    ("checker-on run", "results/compare_table_after_window.md"),
    ("checker-off run", "results/compare_table_after_window_nocheck.md"),
]
for marker, path in blocks:
    start = readme.index(f"## Before vs after: {marker}")
    end = readme.index("\n## ", start + 5)
    section = readme[start:end]
    for kind, label in (("static", "Static prefix reuse"), ("recall", "Recall")):
        new = (
            hdr + "\n" + "\n".join(rows(path, kind))
            if rows(path, kind)
            else "_no points yet_"
        )
        i = section.index(label)
        j = section.index("\n\n", i) + 2  # after the intro line
        k = section.find("\n\n", j)
        if k == -1:
            k = len(section)
        section = section[:j] + new + section[k:]
    readme = readme[:start] + section + readme[end:]
open("README.md", "w").write(readme)
print("tables refreshed")
