#!/usr/bin/env python3
"""Derive the spill-test recipes from the validated gpt-oss C (prefill HiCache) 256k recipe.

gpt-oss_C_spill_base.json : context 280k (256k prefix + 16k new + 1k out + slack), no storage backend
gpt-oss_C_spill_dram.json : same + prefill RaidenDramStorage backend (decode-host DRAM stores)
"""

import copy
import json
from pathlib import Path

RECIPES = Path.home() / "pd_disagg/recipes"
CONTEXT = 286720  # 280k
MANIFEST_DIR = (
    "/home/alisachen_google_com/pd_disagg/spill/manifests"  # on the prefill host
)
REGISTRY = "10.128.0.90:50051"  # global_registry_server on the decode host

EXTRA = {
    "backend_name": "raiden_dram",
    "module_path": "sglang_tpu.raiden_dram.storage",
    "class_name": "RaidenDramStorage",
    "interface_v1": True,
    "global_registry_address": REGISTRY,
    "kv_pool_group": "sglang-prefill-cache",
    "cache_namespace": "gpt-oss-120b-b5c939d-fp8e4m3-page64-v1",
    "decode_store_job": "sglang-decode-dram",
    "decode_store_replicas": ["0"],
    "manifest_dir": MANIFEST_DIR,
    "rpc_timeout_s": 60,
    "prefetch_threshold": 64,
}

base = json.load((RECIPES / "gpt-oss_C_256k.json").open())
for role in ("prefill", "decode"):
    base[role]["server_args"]["context_length"] = CONTEXT
(RECIPES / "gpt-oss_C_spill_base.json").write_text(json.dumps(base, indent=1) + "\n")

dram = copy.deepcopy(base)
p = dram["prefill"]["server_args"]
p.update(
    {
        "hicache_host_memory_mode": "cache",
        "hicache_write_policy": "write_through",
        "hicache_storage_backend": "dynamic",
        "hicache_storage_prefetch_policy": "wait_complete",
        "hicache_storage_backend_extra_config": json.dumps(EXTRA),
    }
)
(RECIPES / "gpt-oss_C_spill_dram.json").write_text(json.dumps(dram, indent=1) + "\n")
print(
    "wrote",
    RECIPES / "gpt-oss_C_spill_base.json",
    RECIPES / "gpt-oss_C_spill_dram.json",
)
