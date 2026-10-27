#!/usr/bin/env bash
# Driver (run on the decode host -b as alisachen_google_com). Starts/stops/inspects the 1P1D pair or a golden server.
#   pd.sh start  <recipe.json> <RUN_NAME>        # prefill on -a (via ssh), decode here, then waits for health + LB
#   pd.sh golden <recipe.json> <RUN_NAME> [a|b]  # monolithic golden on host a (default) or b
#   pd.sh stop   [a|b|all]                        # stops OUR processes only
#   pd.sh status
#   pd.sh info                                    # /get_server_info snapshot of both engines -> results/runs/<RUN>/server_info_*.json
set -uo pipefail
HERE=/home/alisachen_google_com/pd_disagg/scripts
PREFILL_IP=10.128.15.217; DECODE_IP=10.128.0.90; SERVER_PORT=30000; LB_PORT=${LB_PORT:-8080}
ME=alisachen_google_com
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 sa_101411256518727260534@$PREFILL_IP"
log() { echo "$(date -u '+%F %T') $*"; }
# Run a command on host a as our own user (the SA account has sudo; our files live under /home/$ME).
on_a() { $SSH_A "sudo -n -iu $ME bash -lc $(printf %q "$*")"; }
spawn_a() { # recipe role run
  $SSH_A "sudo -n -iu $ME bash -lc $(printf %q "cd /tmp && exec >/dev/null 2>&1 </dev/null; PATH=/home/alisachen_google_com/venv/sglang/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin SGLANG_ROOT=/home/alisachen_google_com/sglang-victim PYTHONPATH=/home/alisachen_google_com/h2h_fork TPU_SYNC_SRC=/home/alisachen_google_com/h2h_fork ENABLE_COMPILE_CACHE=${ENABLE_COMPILE_CACHE:-1} setsid nohup bash $HERE/launch_role.sh $1 $2 $3 >> /home/$ME/pd_disagg/results/runs/$3/launcher_$2.out 2>&1 </dev/null & echo \$!")"
}
spawn_b() { # recipe role run
  mkdir -p "$HOME/pd_disagg/results/runs/$3"
  PATH=/home/alisachen_google_com/venv/sglang/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin SGLANG_ROOT=/home/alisachen_google_com/sglang-victim PYTHONPATH=/home/alisachen_google_com/h2h_fork TPU_SYNC_SRC=/home/alisachen_google_com/h2h_fork ENABLE_COMPILE_CACHE=${ENABLE_COMPILE_CACHE:-1} setsid nohup bash "$HERE/launch_role.sh" "$1" "$2" "$3" >> "$HOME/pd_disagg/results/runs/$3/launcher_$2.out" 2>&1 < /dev/null &
  echo $!
}
wait_url() { # url deadline_s label
  local deadline=$(( $(date +%s) + $2 )); local t0=$(date +%s)
  while (( $(date +%s) < deadline )); do
    if curl -sf -m 5 "$1" >/dev/null 2>&1; then log "$3 ready after $(( $(date +%s) - t0 )) s"; return 0; fi
    sleep 15
  done; log "$3 NOT ready after $2 s"; return 1
}
case "${1:-}" in
start)
  recipe=${2:?recipe}; run=${3:?RUN}
  bash "$HOME/pd_disagg/monitoring/start.sh" 2>/dev/null || log "monitoring stack not started (see monitoring/start.sh)"
  mkdir -p "$HOME/pd_disagg/results/runs/$run"; on_a "mkdir -p /home/$ME/pd_disagg/results/runs/$run"
  cp "$recipe" "$HOME/pd_disagg/results/runs/$run/recipe.json"
  log "starting decode on b ($DECODE_IP)"; spawn_b "$recipe" decode "$run"
  log "starting prefill on a ($PREFILL_IP)"; spawn_a "$recipe" prefill "$run"
  wait_url "http://$PREFILL_IP:$SERVER_PORT/health" 5400 "prefill /health"
  wait_url "http://$DECODE_IP:$SERVER_PORT/health" 5400 "decode /health"
  wait_url "http://$PREFILL_IP:$LB_PORT/health_generate" 900 "LB /health_generate"
  "$0" info "$run"
  ;;
golden)
  recipe=${2:?recipe}; run=${3:?RUN}; host=${4:-a}
  mkdir -p "$HOME/pd_disagg/results/runs/$run"; cp "$recipe" "$HOME/pd_disagg/results/runs/$run/recipe.json"
  if [[ $host == a ]]; then on_a "mkdir -p /home/$ME/pd_disagg/results/runs/$run"; spawn_a "$recipe" golden "$run"; wait_url "http://$PREFILL_IP:$SERVER_PORT/health" 5400 "golden(a) /health"
  else spawn_b "$recipe" golden "$run"; wait_url "http://$DECODE_IP:$SERVER_PORT/health" 5400 "golden(b) /health"; fi
  ;;
stop)
  which=${2:-all}
  [[ $which == all || $which == a ]] && { log "stopping ours on a"; on_a "bash $HERE/cleanup.sh"; }
  [[ $which == all || $which == b ]] && { log "stopping ours on b"; bash "$HERE/cleanup.sh"; }
  ;;
status)
  echo "== a ($PREFILL_IP)"; on_a "pgrep -u $ME -af 'launch_server|launch_router' | cut -c1-160; echo sched=\$(pgrep -u $ME -fc 'sglang::scheduler'); free -g | sed -n 2p"
  echo "== b ($DECODE_IP)"; pgrep -u $ME -af 'launch_server|launch_router' | cut -c1-160; echo "sched=$(pgrep -u $ME -fc 'sglang::scheduler')"; free -g | sed -n 2p
  for u in "http://$PREFILL_IP:$SERVER_PORT" "http://$DECODE_IP:$SERVER_PORT" "http://$PREFILL_IP:$LB_PORT"; do printf '%s health=%s\n' "$u" "$(curl -s -m 5 -o /dev/null -w '%{http_code}' $u/health)"; done
  echo "TPU holders a: $(on_a "sudo -n fuser -v /dev/vfio/0 2>&1 | awk 'NR>1{print \$(NF-3)}' | sort -u | tr '\n' ' '")"
  echo "TPU holders b: $(sudo -n fuser -v /dev/vfio/0 2>&1 | awk 'NR>1{print $(NF-3)}' | sort -u | tr '\n' ' ')"
  ;;
info)
  run=${2:-adhoc}; d="$HOME/pd_disagg/results/runs/$run"; mkdir -p "$d"
  for h in a:$PREFILL_IP b:$DECODE_IP; do
    curl -s -m 10 "http://${h#*:}:$SERVER_PORT/get_server_info" > "$d/server_info_${h%%:*}.json"
    python3 -c "
import json,sys
try: d=json.load(open('$d/server_info_${h%%:*}.json'))
except Exception as e: print('${h%%:*}: no server info', e); sys.exit()
keys=['version','disaggregation_mode','disaggregation_transfer_backend','model_path','kv_cache_dtype','page_size','context_length','max_total_num_tokens','max_running_requests','chunked_prefill_size','disable_radix_cache','enable_hierarchical_cache','hicache_io_backend','hicache_ratio','disaggregation_decode_enable_radix_cache','speculative_algorithm','enable_torch_compile','tpu_compile_bs','tpu_compile_tokens','mem_fraction_static']
print('${h%%:*}:', {k:d.get(k) for k in keys})
st=d.get('internal_states') or [{}]; print('   state:', {k:st[0].get(k) for k in ['num_running_reqs','num_queue_reqs','token_usage','gen_throughput','avg_spec_accept_length']})"
  done
  curl -s -m 5 "http://$PREFILL_IP:$LB_PORT/workers" | head -c 600; echo
  ;;
*) echo "usage: pd.sh start|golden|stop|status|info ..."; exit 2 ;;
esac
