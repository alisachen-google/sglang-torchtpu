#!/usr/bin/env python3
"""Live race / deadlock / hang probes for the decode-host DRAM spill feature on a running 1P/1D pair.

Phases (each bounded by --phase-timeout; a phase that does not finish is reported as HANG):
  concurrent : N two-turn sessions at once, flush between turns -> N recalls in flight together
  abort      : start a recall (turn 2 after flush), drop the client after --abort-after s, re-issue it
  store_pause: SIGSTOP one host-store process on this host while a cold prefill spills, hold it past
               the backend's rpc_timeout, SIGCONT, then flush + recall the same prefix
  registry   : stop the registry during a recall, then restart it (stores must re-register)
Throughout, /health of both engines is polled; any probe slower than --health-slow s is recorded.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spill_bench import flush, generate, load_ids  # noqa: E402

ME = os.environ.get("USER", "alisachen_google_com")


class HealthWatch:
    def __init__(self, urls, slow):
        self.urls, self.slow, self.stop = urls, slow, threading.Event()
        self.samples, self.slow_events, self.errors = 0, [], []
        self.t = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            for u in self.urls:
                t0 = time.perf_counter()
                try:
                    r = requests.get(u.rstrip("/") + "/health", timeout=30)
                    dt = time.perf_counter() - t0
                    if r.status_code != 200:
                        self.errors.append((time.time(), u, r.status_code))
                except Exception as exc:  # noqa: BLE001
                    dt = time.perf_counter() - t0
                    self.errors.append((time.time(), u, repr(exc)[:80]))
                if dt > self.slow:
                    self.slow_events.append((time.time(), u, round(dt, 2)))
                self.samples += 1
            self.stop.wait(5)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.t.join(35)


def pids_of(pattern):
    # Anchor at the start of the command line so the launcher (which names the binary in
    # its --binary argument) is never picked as the victim.
    out = subprocess.run(
        ["pgrep", "-u", ME, "-f", "^[^ ]*" + pattern], capture_output=True, text=True
    ).stdout.split()
    return [int(p) for p in out if int(p) != os.getpid()]


def bounded(fn, timeout, label):
    """Run fn in a thread; return (result|None, status) where status in ok|error|HANG."""
    box = {}

    def run():
        try:
            box["r"] = fn()
        except Exception as exc:  # noqa: BLE001
            box["e"] = repr(exc)[:200]

    t = threading.Thread(target=run, daemon=True)
    t0 = time.perf_counter()
    t.start()
    t.join(timeout)
    dt = time.perf_counter() - t0
    if t.is_alive():
        return None, "HANG", dt
    if "e" in box:
        return box["e"], "error", dt
    return box["r"], "ok", dt


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
    ap.add_argument("--prefix", type=int, default=32768)
    ap.add_argument("--new", type=int, default=8192)
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--phases", default="concurrent,abort,store_pause,registry")
    ap.add_argument(
        "--flush-targets",
        choices=["both", "prefill"],
        default="both",
        help="Plan C: flush only the prefill so the decode keeps the prefix",
    )
    ap.add_argument("--spill-wait", type=float, default=60)
    ap.add_argument("--abort-after", type=float, default=1.0)
    ap.add_argument("--pause-hold", type=float, default=90)
    ap.add_argument("--phase-timeout", type=float, default=1800)
    ap.add_argument("--health-slow", type=float, default=2.0)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument(
        "--stores-script", default=str(Path.home() / "pd_disagg/spill/spill_stores.sh")
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    prefix_src = (
        args.prompts / f"{args.prefix}.jsonl"
        if (args.prompts / f"{args.prefix}.jsonl").exists()
        else args.prompts / "32768.jsonl"
    )
    prefix_ids = load_ids(tok, prefix_src, args.prefix)
    new_ids = load_ids(tok, args.prompts / "8192.jsonl", args.new, row=1)
    q = tok("\n\nContinue the story.", add_special_tokens=False)["input_ids"]
    engines = [args.prefill_url, args.decode_url]
    flush_set = [args.prefill_url] if args.flush_targets == "prefill" else engines
    report = {
        "args": vars(args) | {"prompts": str(args.prompts), "out": str(args.out)},
        "phases": {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def session_ids(i):
        salt = tok(
            f"Stress session {uuid.uuid4().hex} {i}\n\n", add_special_tokens=False
        )["input_ids"]
        t1 = salt + prefix_ids + q
        return t1, t1 + new_ids + q

    def gen(ids):
        return generate(args.lb, ids, args.max_new_tokens, args.phase_timeout)

    def summarize(res):
        return {
            k: res.get(k)
            for k in (
                "ttft_s",
                "e2e_s",
                "cached_tokens",
                "cached_tokens_details",
                "completion_tokens",
                "finish_reason",
            )
        }

    with HealthWatch(engines, args.health_slow) as hw:
        phases = [p.strip() for p in args.phases.split(",") if p.strip()]

        if "concurrent" in phases:

            def phase():
                sessions = [session_ids(i) for i in range(args.sessions)]
                with ThreadPoolExecutor(args.sessions) as ex:
                    first = list(ex.map(lambda s: gen(s[0]), sessions))
                time.sleep(args.spill_wait)
                flush(flush_set, 600)
                with ThreadPoolExecutor(args.sessions) as ex:
                    second = list(ex.map(lambda s: gen(s[1]), sessions))
                return {
                    "turn1": [summarize(r) for r in first],
                    "turn2": [summarize(r) for r in second],
                }

            r, st, dt = bounded(phase, args.phase_timeout, "concurrent")
            report["phases"]["concurrent"] = {
                "status": st,
                "seconds": round(dt, 1),
                "result": r,
            }
            print(
                json.dumps(
                    {"phase": "concurrent", "status": st, "seconds": round(dt, 1)}
                ),
                flush=True,
            )

        if "abort" in phases:

            def phase():
                t1, t2 = session_ids(99)
                a = gen(t1)
                time.sleep(args.spill_wait)
                flush(flush_set, 600)
                body = {
                    "input_ids": t2,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": args.max_new_tokens,
                        "ignore_eos": True,
                    },
                    "stream": True,
                }
                t0 = time.perf_counter()
                try:
                    with requests.post(
                        args.lb.rstrip("/") + "/generate",
                        json=body,
                        stream=True,
                        timeout=(10, args.abort_after),
                    ) as r:
                        for _ in r.iter_lines():
                            break
                except (
                    requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError,
                ):
                    pass  # dropped the client while the recall / prefill was in flight
                aborted_after = time.perf_counter() - t0
                time.sleep(2)
                b = gen(t2)  # re-issue the same request
                return {
                    "turn1": summarize(a),
                    "aborted_after_s": round(aborted_after, 2),
                    "reissued": summarize(b),
                }

            r, st, dt = bounded(phase, args.phase_timeout, "abort")
            report["phases"]["abort"] = {
                "status": st,
                "seconds": round(dt, 1),
                "result": r,
            }
            print(
                json.dumps({"phase": "abort", "status": st, "seconds": round(dt, 1)}),
                flush=True,
            )

        if "store_pause" in phases:

            def phase():
                stores = pids_of("kv_cache_host_store_node_main")
                if not stores:
                    return {"skipped": "no host-store process on this host"}
                victim = stores[0]
                t1, t2 = session_ids(77)
                os.kill(victim, signal.SIGSTOP)
                t_stop = time.time()
                try:
                    a = gen(
                        t1
                    )  # cold prefill; its spill to the paused store must time out, not hang
                    time.sleep(max(0, args.pause_hold - (time.time() - t_stop)))
                finally:
                    os.kill(victim, signal.SIGCONT)
                time.sleep(args.spill_wait)
                flush(engines, 600)
                b = gen(t2)
                c = gen(session_ids(78)[0])  # a fresh cold request afterwards
                return {
                    "paused_pid": victim,
                    "turn1_during_pause": summarize(a),
                    "turn2_after_resume": summarize(b),
                    "fresh_after": summarize(c),
                }

            r, st, dt = bounded(phase, args.phase_timeout, "store_pause")
            report["phases"]["store_pause"] = {
                "status": st,
                "seconds": round(dt, 1),
                "result": r,
            }
            print(
                json.dumps(
                    {"phase": "store_pause", "status": st, "seconds": round(dt, 1)}
                ),
                flush=True,
            )

        if "decode_flush" in phases:

            def phase():
                t1, t2 = session_ids(66)
                a = gen(t1)
                time.sleep(args.spill_wait)
                flush(engines, 600)  # both: the decode no longer has the prefix either
                b = gen(
                    t2
                )  # the pull must miss and the request recompute, with no hang
                t3, t4 = session_ids(67)
                c = gen(t3)
                time.sleep(args.spill_wait)
                flush([args.prefill_url], 600)
                d = gen(t4)  # and a recall right after still works
                return {
                    "turn1": summarize(a),
                    "turn2_after_both_flushed": summarize(b),
                    "turn1_next": summarize(c),
                    "turn2_recall": summarize(d),
                }

            r, st, dt = bounded(phase, args.phase_timeout, "decode_flush")
            report["phases"]["decode_flush"] = {
                "status": st,
                "seconds": round(dt, 1),
                "result": r,
            }
            print(
                json.dumps(
                    {"phase": "decode_flush", "status": st, "seconds": round(dt, 1)}
                ),
                flush=True,
            )

        if "registry" in phases:

            def phase():
                t1, t2 = session_ids(55)
                a = gen(t1)
                time.sleep(args.spill_wait)
                flush(engines, 600)
                subprocess.run([args.stores_script, "registry-stop"], check=False)
                b = gen(t2)  # lookup fails -> miss -> recompute; must not hang
                subprocess.run([args.stores_script, "registry-start"], check=False)
                time.sleep(20)
                t3, t4 = session_ids(56)
                c = gen(t3)
                time.sleep(args.spill_wait)
                flush(engines, 600)
                d = gen(
                    t4
                )  # after restart: recall works again only if the stores re-registered
                return {
                    "turn1": summarize(a),
                    "turn2_registry_down": summarize(b),
                    "turn1_after_restart": summarize(c),
                    "turn2_after_restart": summarize(d),
                }

            r, st, dt = bounded(phase, args.phase_timeout, "registry")
            report["phases"]["registry"] = {
                "status": st,
                "seconds": round(dt, 1),
                "result": r,
            }
            print(
                json.dumps(
                    {"phase": "registry", "status": st, "seconds": round(dt, 1)}
                ),
                flush=True,
            )

    report["health"] = {
        "samples": hw.samples,
        "slow": hw.slow_events[:50],
        "errors": hw.errors[:50],
    }
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(
        json.dumps(
            {
                "health_samples": hw.samples,
                "slow_probes": len(hw.slow_events),
                "probe_errors": len(hw.errors),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
