# P/D decode-DRAM offload: benchmark and verification harness

Scripts used to validate and benchmark the KV offload from the prefill's RAM
into the decode's RAM on a 1P/1D TPU pair (see `sglang_tpu/README.md`,
"Decode DRAM as a KV offload extension for prefill"). They assume two hosts
reachable over ssh, the venv at `~/venv/sglang`, tpu-sync on `PYTHONPATH`,
this tree at `~/sglang-e18f` on both hosts, and the layout under
`~/pd_disagg` (`prompts/`, `recipes/`, `results/`). Host IPs in the scripts are
defaults for one environment; override them with the variables noted below.

## Workloads (clients, run on the decode host)

| Script | Workload | Key metrics |
|---|---|---|
| `static_reuse.py` | K distinct prefixes, each visited 3 times, interleaved so the oldest spills out of HBM before its next visit | prefill hit rate, TTFT, output tok/s |
| `victim_recall.py` | fill N sessions, then recall them oldest first (`--phase fill|recall`) | recall hit rate, TTFT |
| `victim_integrity.py` | greedy decoding; cold references, fill, recall from decode RAM, then flush and recompute: outputs must match token for token, with background probes compared throughout | match counts |
| `build_prompts.py` / `needle_check.py` | build the needle-in-haystack prompt files `prompts/<model>/<tokens>.jsonl`, check needles in outputs | |

## Drivers

| Script | Purpose |
|---|---|
| `pd_e18f_vw.sh` | start/stop a 1P/1D pair from this tree on any two hosts: `PREFILL_IP=... DECODE_IP=... bash pd_e18f_vw.sh start <recipe> <run>` (logs under `~/pd_disagg/results/runs/<run>/`) |
| `pd_fork.sh` | same for a fixed local pair |
| `chain_pair.sh` | one sweep: start the pair, wipe check (both caches must be empty after a flush), then static and recall points with a wipe before each; `A_IP B_IP CFG NAME LAUNCHER RECIPE TOK KINDS STATIC RECALL FLUSH_TRIES` |
| `chain_goal_v2.sh` | the full before/after sweep (128k and 256k static and recall ladders) |
| `make_spill_recipes.py` | derive recipe JSON files (engine args and env) from a base |
| `recipes/` | the recipes used: gpt-oss-120b with the offload on (`gpt-oss_C_victim_tip.json` is the switch-only recipe), local tiers only, the consensus-checker variant, and a dense Qwen3 recipe |

## Reports

| Script | Purpose |
|---|---|
| `compare_goal.py` | before vs after markdown tables (hit, hit gain, TTFT, tok/s, KV pushed/pulled) from two sweeps |
| `refresh_readme_tables.py` | rebuild the tables in a results README |
| `victim_report.py`, `victim_stats2.sh` | per-run push/offer/pull counters from the engine logs |

## Debugging

| Script | Purpose |
|---|---|
| `hang_capture.sh`, `trace_cap.sh` | py-spy dumps and bounded collective traces of all ranks on a hang |
| `collective_trace.py`, `collective_diff.py` | per-rank collective tracer (`SGLANG_TPU_COLLECTIVE_TRACE=<dir>`) and the diff that names the first divergent collective |
| `native/ram2ram_check.py` | tpu-sync host-pool manager: a host-to-host read in both directions on one host, bytes verified (run with the pair stopped) |
| `native/h2h_vs_d2h_stress.py` | do back-to-back host pushes stall device-to-host copies |

## Unit tests

`test/registered/unit/hardware_backend/tpu/test_pd_victim_push.py` covers the
push, receive, flush and recall-landing paths; `test_pd_victim_lockstep.py`
runs the puller on several ranks with a lockstep harness that reports any
rank skipping or disagreeing on a collective.

## Robustness cases (added 2026-10-27)

| Script / recipe | Purpose |
|---|---|
| `decode_kill_case.sh` | kill the decode while recalls are in flight, restart it alone (`pd_e18f_vw.sh start-decode`), recall again; the prefill must survive |
| `prefill_kill_case.sh` | kill the prefill while the decode holds offloaded KV, restart it alone (`start-prefill`), recall again; the decode must survive and offers to the new prefill must be clean |
| `crc_compare_p1.py <run>` | join the `[victim-crc]` / `[recall-crc]` lines of a run made with `recipes/gpt-oss/gpt-oss_C_victim_crc.json` (`SGLANG_TPU_DEBUG_VICTIM_CRC=1`, a diagnostic build) into per-rank byte-level equality counts |
| `recipes/gpt-oss/gpt-oss_C_victim_tipcheck.json` | the switch-only recipe with `SGLANG_ENABLE_RANK_CONSENSUS_CHECKER=1` (rank divergence ends the process) |
| `recipes/gpt-oss/gpt-oss_C_victim_tiphostpress.json` | prefill `--hicache-ratio 0.5`: host pool of about three 128k documents |
| `recipes/gpt-oss/gpt-oss_C_victim_unified.json` | `--enable-unified-memory` on both roles (for the `pd-victim-unified-pool` branch) |
