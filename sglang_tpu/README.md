# sglang_tpu: SGLang's TPU backend as an out-of-tree plugin

All TPU-only code of `sglang-torchtpu` lives in this package (dist `sglang-tpu`).
The sglang fork in `python/` shrinks toward a pinned upstream commit as its TPU
edits move here or go upstream as seams. This file is the operational summary.

## Install

Both distributions install editable, side by side:

```bash
pip install -e python/            # the sglang fork (dist "sglang")
pip install --no-deps -e .        # this plugin (dist "sglang-tpu")
```

`--no-deps` keeps pip from re-resolving the fork's dependency tree; the editable
fork already satisfies the plugin's `sglang` requirement.

## Two platform identities, one code base

| Mode | Select with | Platform class | `is_out_of_tree()` | Status |
|---|---|---|---|---|
| A (default) | `SGLANG_PLATFORM=tpu` or unset | `sglang_tpu.tpu_platform.TpuSRTPlatform` (`PlatformEnum.TPU`) | False | the shipped path |
| B | `SGLANG_PLATFORM=tpu_oot` | `sglang_tpu.platform.TpuOOTPlatform` (`PlatformEnum.OOT`, `is_tpu()` True) | True | what the PR gate and the single-host nightly run; opt-in for users until the divergences below close |

`tpu_oot` only activates when named: with `SGLANG_PLATFORM` unset the loader
calls every installed `activate()` and errors when two answer.

## Mode-B divergences (gates that take a different arm under OOT)

Measured against the fork at the wave-0 move; re-audit after every sync.

Closed: hybrid-SWA MHA models (the OOT pool and allocator arms mirror the TPU
arms); the import-time `init_backend()` call (`TpuOOTPlatform.init_backend`
defers to the worker bootstrap); MLA pool construction (the OOT builder's
`index_head_dim` kwarg); `get_device_memory_capacity`, which reports nothing
for TPU in both modes.

Open: sparse attention (DSA) models run in mode A only. Mode B refuses them at
pool construction (`TpuOOTPlatform.get_dsa_kv_pool_cls`), because the OOT DSA
builder does not say which layers run their own indexer.

Different arm, same outcome for every configuration that runs:

- `--prefill-only-disable-kv-cache`: the pool-family check refuses OOT pools,
  but argument resolution already rejects the flag on TPU in both modes (it
  needs the fa3/fa4 prefill backend).
- DFLASH draft backend: mode B falls back to `pallas_tpu_rpa_v3` for a backend
  `speculative/draft_utils.py` cannot build on TPU; mode A fails at construction.
- `SGLANG_PROFILE_V2=1` with the `"TPU"` activity: mode B also records a torch
  `PrivateUse1` trace, mode A records none; neither reaches xprof on that path.

Unreachable today: MLA with hybrid SWA or an FP4 KV cache takes the plain OOT
MLA builder in mode B and the generic builders in mode A; no TPU model uses
either.

## Layout

```
sglang_tpu/
  __init__.py            activate() for the tpu_oot entry point
  platform.py            TpuOOTPlatform (identity + the gate-required methods)
  tpu_platform.py        TpuSRTPlatform, the in-tree identity (moved here as-is)
  envs.py  hooks/  compat/
  attention/pallas/      Pallas attention kernels (was layers/attention/pallas)
  moe/pallas/            Pallas MoE kernels (was layers/moe/pallas)
  disaggregation/{tpu,raiden}/
  graph_runner/          decode graph runner + ViT compile runners
  <everything else>      was python/sglang/srt/hardware_backend/tpu/
```

The old paths under `python/sglang/srt/` are one-file re-export shims so
in-flight branches keep importing; they are deleted once the open-PR backlog
drains. Do not add code to them.

`tools/plugin/inventory.py` lists every hook and runtime patch with the core
attributes it replaces; `tools/plugin/scoreboard.py` prints every debt counter
of the migration. The TPU Plugin Scoreboard workflow publishes both daily.

## Rules for new TPU code

1. New TPU-only modules go here, never under `python/sglang/srt/`.
2. Core never imports this package at module scope. A function-scope import
   inside an `is_tpu()` guard is tolerated during the migration and counted by
   the ratchet lint (`tools/plugin/check_tpu_branches.py`); the baseline may
   only go down.
3. Prefer, in order: an existing platform method or capability flag; a registry
   populated from `init_backend()` (`register_attention_backend`,
   `BaseFusedOp.register_oot_forward`); a resolution hook; subclass-and-override
   through a factory; a seam PR upstream carried as a `hooks/` entry until it
   merges. A new `is_tpu()` branch in core needs a linked upstream seam issue.
4. Env vars: declare new TPU-only variables in `sglang_tpu/envs.py`. The
   existing `SGLANG_TPU_*` entries stay in `sglang.srt.environ` until their
   core readers move; never declare one name in both registries.

## Hooks

`hooks.register` is declared by both distributions (`tpu_hooks` in `python/`,
`tpu_oot_hooks` here): `SGLANG_PLATFORM=tpu` skips every general plugin of
`sglang-tpu`, so mode A needs its own entry point. `register()` runs once per
process and only on TPU, since every host installs the plugin.

## Tests

`test/registered/tpu/**` and `test/registered/unit/hardware_backend/tpu/**`
stay where they are (`test/run_suite.py` only collects `test/registered/**`).
The e2e smoke suites honour an inherited `SGLANG_PLATFORM`, so mode B is
`SGLANG_PLATFORM=tpu_oot python test/registered/tpu/e2e/<file>.py`.
CI picks the mode per lane (`render_jobset.py --platform`, the `platform`
input of `_tpu-test.yml`), so each test runs in one mode only: the PR gate
(stage-a, stage-b) and the single-host nightly (`nightly-tpu-4-v7`) run mode B;
the multi-host, P/D, accuracy, perf, combo and external lanes stay on mode A.

## Muse Spark multimodal

`models/muse_spark.py` is the text decoder; `models/muse_spark_omni.py` adds the
NaViT vision tower (adapted from upstream SGLang PR
[#44666](https://github.com/sgl-project/sglang/pull/44666), commit
`26305be7f410`) and is the registered `MuseSparkForConditionalGeneration`.
Image and video serving needs `--enable-multimodal`,
`SGLANG_EXTERNAL_MM_MODEL_ARCH=MuseSparkForConditionalGeneration` and
`SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=sglang_tpu.multimodal`.

The vision encoder is replicated and complete images are distributed across
attention-TP ranks; CPU features stay on the host until their encoder rank
prepares patches. Live check:
[`test_tpu_muse_spark_mm_pd_multihost.py`](../test/registered/tpu/e2e/test_tpu_muse_spark_mm_pd_multihost.py)
(images and video over P/D), with `SGLANG_TPU_USE_LIVE_SERVER=1` and
`BASE_URL` pointing at the router.

## Experimental P/D victim prefix cache

The v3 policy keeps prefill as the primary prefix cache. Before host pressure
would discard an old cached branch, prefill exports its full KV and its ending
SWA window to the decode that served that branch. Decode retains pushed victims;
ordinary generation requests do not create another persistent radix-cache copy.
A later prefill miss can pull the victim back through the existing Raiden prefix
exchange. Different branches of a shared system prompt, cache salts, and adapter
namespaces retain independent ownership.

For an experiment, enable HiCache on both roles with
`--enable-hierarchical-cache --hicache-io-backend raiden` and enable decode radix
caching with `--disaggregation-decode-enable-radix-cache`. Set on both roles:

```bash
SGLANG_DISAGG_PREFILL_PULL_DECODE_PREFIX=1
SGLANG_DISAGG_PREFIX_VICTIM_PUSH=1
```

For hybrid SWA models, also set
`SGLANG_TPU_ENABLE_PD_PREFILL_SWA_PATH_END_BACKUP=1` on prefill and size the decode
window tier with `SGLANG_TPU_PD_DECODE_SWA_HOST_WINDOWS=128` on decode. The latter
is a count of windows; choose it together with the decode full-KV host budget.
The default is zero, preserving the existing full-only host tier when this
experiment is disabled. Both platform modes use the same implementation.

The current path uses the Python unified radix tree, matching P/D attention-TP
and context-parallel geometry, equal page sizes/layouts, and no pipeline or
GDN/Mamba state transfer. Multiple decodes need session affinity so the next
turn reaches its holder. This is an experimental implementation, not a newly
validated support-matrix entry.

Exports are independent of later requests and move RAM to RAM through the
existing HiCache host-pool managers. Prefill registers pinned host pages with
`register_read(source="host")` before offering them. Decode reserves host pages
and pulls into them with `start_read(destination="host", local_host_block_ids=...)`.
The received victim enters decode's radix cache as a host-only node; this
transfer uses no HBM or transient staging slot on either side. At most four
victims per engine are in flight. Every rank agrees on the selected branch,
namespace, length and outcome before allocation or insertion; request labels
carry a per-prefill incarnation so separate prefills and restarts cannot alias.

Source pins and destination allocations remain live until native
`poll_stats_with_details()` reports terminal completion, including after an
offer is refused or expires. Native transfer deadlines use
`SGLANG_RAIDEN_CHUNK_TRANSFER_TIMEOUT_S`. A wall-clock timeout alone never frees
pages that a remote writer may still use. Cache reset/memory release waits for
accepted reads to drain; an early attempt raises and can be retried. The wrapper
owns completion polling for each host manager and retains reports for other
in-flight victims.

At native revision `759530c7`, a session deadline starts draining but does not
interrupt a blocked TCP data operation. The socket I/O timeouts below default
to disabled, so a paused peer can keep source pins and destination allocations
live past the session deadline. Configure positive native I/O timeouts before
starting both roles as well as setting the session timeout. These native names
describe send/receive operations; set all nine on both roles because either
role can send KV during spill or recall. For example, this Bash snippet keeps
existing values and otherwise supplies a 60-second per-I/O budget:

```bash
for raiden_io_timeout in \
  TPU_RAIDEN_PREFILL_HANDSHAKE_ACK_READ_TIMEOUT_S \
  TPU_RAIDEN_PREFILL_FINAL_ACK_READ_TIMEOUT_S \
  TPU_RAIDEN_PREFILL_PAYLOAD_READ_TIMEOUT_S \
  TPU_RAIDEN_PREFILL_HANDSHAKE_WRITE_TIMEOUT_S \
  TPU_RAIDEN_PREFILL_PAYLOAD_WRITE_TIMEOUT_S \
  TPU_RAIDEN_DECODE_HANDSHAKE_READ_TIMEOUT_S \
  TPU_RAIDEN_DECODE_PAYLOAD_READ_TIMEOUT_S \
  TPU_RAIDEN_DECODE_ACK_WRITE_TIMEOUT_S \
  TPU_RAIDEN_DECODE_PAYLOAD_WRITE_TIMEOUT_S
do
  export "${raiden_io_timeout}=${!raiden_io_timeout:-60}"
done
```

Choose the I/O budget for the expected payload and network. These settings bound
individual stalled I/O calls, not the total duration of a transfer that keeps
making slow progress. Memory remains reserved until native terminal completion;
neither the session clock nor the I/O clock alone permits reusing live pages.
Pausing generation does not revoke native reads; invoke memory-release APIs
only after all native ownership has drained and the scheduler is fully idle.

The transport requires matching Python and native tpu-sync artifacts from the
`ram2ram` API in
[`759530c7fb0a5f7c38114187e1bcde745f709f24`](https://github.com/alisachen-google/h2h/commit/759530c7fb0a5f7c38114187e1bcde745f709f24).
Both SGLang roles must use the same victim-protocol revision. An existing
tpu-sync build of the pinned revision needs no patch. For a source build, follow
the pinned fork's
[build instructions](https://github.com/alisachen-google/h2h/blob/759530c7fb0a5f7c38114187e1bcde745f709f24/README.md#option-2-building-from-source)
with `./build.sh torch` in an environment matching the installed `torch_tpu`
ABI. Select that checkout on both roles:

```bash
export TPU_SYNC_SRC=/path/to/tpu-sync
export PYTHONPATH="$TPU_SYNC_SRC${PYTHONPATH:+:$PYTHONPATH}"
python - <<'PY'
import inspect
from tpu_sync.api.torch import kv_cache_manager as k
print(k.__file__)
assert "source" in inspect.signature(k.KVCacheManager.register_read).parameters
assert "destination" in inspect.signature(k.KVCacheManager.start_read).parameters
assert "local_host_block_ids" in inspect.signature(k.KVCacheManager.start_read).parameters
PY
```

The signature check confirms the selected Python API; the native extension must
also come from that build. Each victim-enabled host-pool manager opens a
positive control port: this revision's Python constructor treats port zero as
cache-only. Offers publish the producer's reachable control endpoint, rather
than its block-data endpoint.

FULL pages and the page-aligned trailing SWA window move through their respective
host pools under one producer-owned UUID. A session is inserted only when every
rank has received both copies and has the producer's successful completion
notice. Failed transfers release both landing allocations after accepted native
reads settle; shared-prefix insertions release duplicate FULL and SWA pages
independently. This path has CPU unit coverage but has not been validated with
model inference on TPU.
