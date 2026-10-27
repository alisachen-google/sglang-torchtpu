#!/usr/bin/env bash
# Goal benchmark on a+b (gpt-oss-120b 1P1D), one config per invocation:
#   CFG=before : upstream main 49ddede7, installed tpu-sync wheel, features off
#   CFG=after  : fork pd-victim-prefix-cache c0e561665c + tpu-sync 759530c7 with the h2h_push patch (~/h2h_fork)
# Device and host KV caches are wiped (flush on both engines) before every benchmark point; the wipe is
# verified on hardware once per config (a host-resident prompt must come back with 0 cached tokens).
# Workloads: victim recall (fill N, settle, recall oldest first) and static prefix reuse (K docs x 3 visits).
set -uo pipefail
CFG=${CFG:?before|after}; RUN=${RUN:-goal_${CFG}_1}
D=$HOME/pd_disagg/spill; R=$D/results_goal; L=$D/logs; PY=$HOME/venv/sglang/bin/python; mkdir -p $R
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
A_IP=10.128.15.217; B_IP=10.128.0.90
log() { echo "$(date -u '+%F %T') $*"; }
if [ $CFG = before ]; then launcher=$D/pd_base49.sh; recipe=$HOME/pd_disagg/recipes/v6c_long/base/gpt-oss_C_victim.json
else launcher=$D/pd_fork.sh; recipe=$HOME/pd_disagg/recipes/v6c_long/victim/gpt-oss_C_victim.json; fi
RECALL=${RECALL:-"131072:8192:8 131072:8192:10 131072:8192:12 131072:8192:16 131072:8192:20 131072:8192:24 131072:8192:28 131072:8192:32 262144:16384:4 262144:16384:5 262144:16384:6 262144:16384:8 262144:16384:10 262144:16384:12 262144:16384:14 262144:16384:16"}
STATIC=${STATIC:-"131072:8192:4 131072:8192:8 131072:8192:12 131072:8192:16 131072:8192:20 131072:8192:24 131072:8192:28 262144:16384:2 262144:16384:4 262144:16384:6 262144:16384:8 262144:16384:10 262144:16384:12 262144:16384:14"}
SSH="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@$A_IP"
PLOG=/home/alisachen_google_com/pd_disagg/results/runs/$RUN/prefill/server.log
RESTARTS=0
warm() { # wait until both engines finished their torch.compile warmup (traffic during warmup skews timing)
  for i in $(seq 1 60); do
    a=$($SSH "sudo grep -c 'Warmup complete' /home/alisachen_google_com/pd_disagg/results/runs/$RUN/prefill/server.log" 2>/dev/null || echo 0)
    b=$(grep -c 'Warmup complete' $HOME/pd_disagg/results/runs/$RUN/decode/server.log 2>/dev/null || echo 0)
    [ "${a:-0}" -ge 8 ] && [ "${b:-0}" -ge 8 ] && { log "warmup complete on both roles"; return 0; }; sleep 10
  done; log "WARN: warmup not confirmed after 10 min (prefill=$a decode=$b)"; }
start_pair() { bash $launcher stop all > /dev/null 2>&1; sleep 20
  bash $launcher start $recipe $RUN >> $L/${RUN}_start.log 2>&1
  curl -sf -m 5 http://$A_IP:8080/health >/dev/null || return 1
  warm; }
flush_one() { for i in $(seq 1 6); do c=$(curl -s -m 300 -o /dev/null -w '%{http_code}' -X POST $1/flush_cache); [ "$c" = 200 ] && return 0; sleep 10; done; return 1; }
wipe() { # flush both tiers on both engines; if the prefill refuses (stuck inflight send), restart the pair
  local ok=1
  flush_one http://$B_IP:30000 || { log "WIPE: decode flush failed"; ok=0; }
  flush_one http://$A_IP:30000 || { log "WIPE: prefill flush refused after 60 s (inflight send never settled); restarting the pair"; RESTARTS=$((RESTARTS+1)); start_pair || { log "WIPE: pair restart failed"; return 1; }; log "WIPE: pair restarted (restart #$RESTARTS)"; }
  sleep 10; [ $ok = 1 ]; }
pm() { curl -s -m 10 http://$1:30000/metrics | awk -v k="sglang:$2" '$1 ~ "^"k"(\\{|$)" {s+=$2} END {printf "%d", s}'; }
plines() { $SSH "sudo sh -c 'wc -l < $PLOG'"; }
logcached() { $SSH "sudo sed -n '$1,$2p' $PLOG | grep 'TP0' | grep 'Prefill batch' | grep -oE '#cached-token: [0-9]+' | awk '{s+=\$2} END {printf \"%d\", s}'"; }
snap() { echo "$(pm $A_IP cached_tokens_total) $(pm $A_IP prompt_tokens_total) $(pm $B_IP cached_tokens_total) $(pm $B_IP prompt_tokens_total) $(plines)"; }
point() { # kind label snap0 snap1 summary_json
  read pc0 pp0 dc0 dp0 l0 <<<"$3"; read pc1 pp1 dc1 dp1 l1 <<<"$4"
  lc=$(logcached $((l0+1)) $l1); lc=${lc:-0}; vs=$(bash $D/victim_stats.sh $RUN 2>/dev/null | tr '\n' ' ' | tr -s ' ')
  python3 - "$1" "$2" "$5" "$((pc1-pc0))" "$((pp1-pp0))" "$((dc1-dc0))" "$((dp1-dp0))" "$lc" "$vs" <<'PY'
import json, sys
kind, lab, f = sys.argv[1:4]; pc, pp, dc, dp, lc = map(int, sys.argv[4:9]); vs = sys.argv[9]
try:
    s = json.load(open(f))
except Exception as e:
    print(f"POINT {lab} no summary ({e})"); sys.exit()
ph = s["recall"] if kind == "recall" else s
print("POINT " + json.dumps({"cfg": s["run"], "kind": kind, "label": lab, "sessions": s.get("sessions", s.get("prefixes")),
    "prefill_hit": round(pc / pp, 4) if pp else None, "prefill_cached": pc, "prefill_cached_log": lc, "prefill_prompt": pp,
    "decode_reuse": round(dc / dp, 4) if dp else None,
    "ttft50": ph.get("ttft_p50"), "ttft95": ph.get("ttft_p95"), "e2e50": ph.get("e2e_p50"), "e2e95": ph.get("e2e_p95"),
    "tok_s": ph.get("total_tok_s"), "out_tok_s": ph.get("output_tok_s"), "wall": ph.get("wall_s"), "errors": ph.get("errors"),
    "victim": vs}))
PY
}
log "starting $RUN ($CFG)"; : > $L/${RUN}_start.log
start_pair || { log "$RUN pair did not come up"; exit 1; }
$SSH "sudo -n -iu alisachen_google_com bash -c 'd=/home/alisachen_google_com/pd_disagg/results/runs/$RUN/prefill/maps; mkdir -p \$d; for p in \$(pgrep -u alisachen_google_com -f sglang::scheduler); do cat /proc/\$p/maps > \$d/\$p.maps; done'"

# Wipe check: 12 x 128k sessions push the oldest out of HBM; after a wipe, the oldest must come back cold.
wipe || exit 1
$PY $D/victim_recall.py --sessions 12 --prefix 131072 --new 8192 --out-tokens 16 --concurrency 4 --seed 99 --run $RUN --phase fill --label wipecheck_fill --out $R/${RUN}_wipecheck_fill.jsonl > /dev/null 2>&1
wipe || exit 1
s0=$(snap)
$PY $D/victim_recall.py --sessions 1 --prefix 131072 --new 8192 --out-tokens 16 --concurrency 1 --seed 99 --run $RUN --phase fill --label wipecheck_probe --out $R/${RUN}_wipecheck_probe.jsonl > /dev/null 2>&1
s1=$(snap); read pc0 pp0 dc0 dp0 _ <<<"$s0"; read pc1 pp1 dc1 dp1 _ <<<"$s1"
log "WIPE CHECK: after flush, prefill cached $((pc1-pc0))/$((pp1-pp0)), decode cached $((dc1-dc0))/$((dp1-dp0)) (both must be 0)"
[ $((pc1-pc0)) = 0 ] && [ $((dc1-dc0)) = 0 ] || { log "WIPE CHECK FAILED: cache survived the flush"; exit 1; }

for kind in ${KINDS:-recall static}; do
if [ $kind = recall ]; then for pt in $RECALL; do
  IFS=: read P N S <<<"$pt"; lab=recall_p$((P/1024))k_n$((N/1024))k_s$S
  curl -sf -m 10 http://$A_IP:8080/health >/dev/null || { log "LB unhealthy before $lab"; exit 1; }
  wipe || exit 1
  A=(--sessions $S --prefix $P --new $N --out-tokens 1024 --concurrency 4 --seed 7 --timeout 1800 --run $RUN --order oldest)
  log "$lab fill"; $PY $D/victim_recall.py "${A[@]}" --phase fill --label ${lab}_fill --out $R/${RUN}_${lab}_fill.jsonl 2>&1 | grep -E 'Traceback|Error' | head -3
  prev=""; for i in $(seq 1 20); do cur=$(bash $D/victim_stats.sh $RUN 2>/dev/null | grep 'prefill push' | md5sum); [ "$cur" = "$prev" ] && break; prev=$cur; sleep 15; done
  s0=$(snap); log "$lab recall"
  $PY $D/victim_recall.py "${A[@]}" --phase recall --label ${lab}_recall --out $R/${RUN}_${lab}_recall.jsonl 2>&1 | grep -E 'Traceback|Error' | head -3
  sleep 5; s1=$(snap); point recall $lab "$s0" "$s1" $R/${RUN}_${lab}_recall.summary.json
done; fi
if [ $kind = static ]; then for pt in $STATIC; do
  IFS=: read P N K <<<"$pt"; lab=static_p$((P/1024))k_n$((N/1024))k_k$K
  curl -sf -m 10 http://$A_IP:8080/health >/dev/null || { log "LB unhealthy before $lab"; exit 1; }
  wipe || exit 1
  s0=$(snap); log "$lab"
  $PY $D/static_reuse.py --prefixes $K --visits 3 --prefix $P --new $N --out-tokens 1024 --concurrency 4 --seed 13 --run $RUN --label $lab --out $R/${RUN}_$lab.jsonl 2>&1 | grep -E 'Traceback|Error' | head -3
  sleep 5; s1=$(snap); point static $lab "$s0" "$s1" $R/${RUN}_$lab.summary.json
done; fi
done
log "GOAL $CFG DONE (pair restarts for refused wipes: $RESTARTS)"
