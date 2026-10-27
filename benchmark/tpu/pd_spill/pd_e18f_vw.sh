#!/usr/bin/env bash
# 1P1D driver for the 6c60fd77 worktree (~/sglang-6c60) + tpu-sync 759530c7 (~/h2h), venv ~/venv/sglang on any two hosts.
#   PREFILL_IP=.. DECODE_IP=.. pd_main.sh start <recipe.json> <RUN>   # recipe path must exist on both hosts
#   PREFILL_IP=.. DECODE_IP=.. pd_main.sh stop|status
# Runs from host b as alisachen_google_com; remote hosts are reached as the SA user and switched to our account.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ME=alisachen_google_com; SERVER_PORT=30000; LB_PORT=${LB_PORT:-8080}
PREFILL_IP=${PREFILL_IP:?}; DECODE_IP=${DECODE_IP:?}
MYIP=$(hostname -I | awk '{print $1}')
SSH="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20"
log() { echo "$(date -u '+%F %T') $*"; }
run_on() { # ip cmd
  if [[ $1 == "$MYIP" ]]; then bash -lc "$2"; else $SSH sa_101411256518727260534@$1 "sudo -n -iu $ME bash -lc $(printf %q "$2")"; fi
}
spawn() { # ip role recipe run
  local ip=$1 role=$2 recipe=$3 run=$4
  run_on "$ip" "mkdir -p /home/$ME/pd_disagg/results/runs/$run && cd /tmp && exec >/dev/null 2>&1 </dev/null; PATH=/home/$ME/venv/sglang/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin SGLANG_ROOT=/home/$ME/sglang-e18f PYTHON=/home/$ME/venv/sglang/bin/python PYTHONPATH=/home/$ME/h2h TPU_SYNC_SRC=/home/$ME/h2h SGLANG_TPU_DEBUG_VICTIM_CRC=0 PREFILL_IP=$PREFILL_IP DECODE_IP=$DECODE_IP ENABLE_COMPILE_CACHE=1 setsid nohup bash /home/$ME/pd_disagg/scripts/launch_role.sh $recipe $role $run >> /home/$ME/pd_disagg/results/runs/$run/launcher_$role.out 2>&1 </dev/null & echo \$!"
}
wait_url() { local deadline=$(( $(date +%s) + $2 )); local t0=$(date +%s)
  while (( $(date +%s) < deadline )); do if curl -sf -m 5 "$1" >/dev/null 2>&1; then log "$3 ready after $(( $(date +%s) - t0 )) s"; return 0; fi; sleep 20; done
  log "$3 NOT ready after $2 s"; return 1; }
case "${1:-}" in
start) recipe=${2:?}; run=${3:?}
  log "decode on $DECODE_IP"; spawn "$DECODE_IP" decode "$recipe" "$run"
  log "prefill on $PREFILL_IP"; spawn "$PREFILL_IP" prefill "$recipe" "$run"
  wait_url "http://$PREFILL_IP:$SERVER_PORT/health" 5400 "prefill /health"
  wait_url "http://$DECODE_IP:$SERVER_PORT/health" 5400 "decode /health"
  wait_url "http://$PREFILL_IP:$LB_PORT/health" 900 "LB /health" ;;
start-prefill) recipe=${2:?}; run=${3:?}
  log "prefill only on $PREFILL_IP"; spawn "$PREFILL_IP" prefill "$recipe" "$run"
  wait_url "http://$PREFILL_IP:$SERVER_PORT/health" 5400 "prefill /health"
  wait_url "http://$PREFILL_IP:$LB_PORT/health" 900 "LB /health" ;;
stop-prefill) log "stopping ours on $PREFILL_IP"; run_on "$PREFILL_IP" "bash /home/$ME/pd_disagg/scripts/cleanup.sh" ;;
start-decode) recipe=${2:?}; run=${3:?}
  log "decode only on $DECODE_IP"; spawn "$DECODE_IP" decode "$recipe" "$run"
  wait_url "http://$DECODE_IP:$SERVER_PORT/health" 5400 "decode /health" ;;
stop-decode) log "stopping ours on $DECODE_IP"; run_on "$DECODE_IP" "bash /home/$ME/pd_disagg/scripts/cleanup.sh" ;;
stop) for ip in $PREFILL_IP $DECODE_IP; do log "stopping ours on $ip"; run_on "$ip" "bash /home/$ME/pd_disagg/scripts/cleanup.sh"; done ;;
status) for ip in $PREFILL_IP $DECODE_IP; do echo "== $ip"; run_on "$ip" "pgrep -u $ME -af 'launch_server|launch_router' | cut -c1-120 | head -3; echo sched=\$(pgrep -u $ME -fc 'sglang::scheduler'); free -g | sed -n 2p"; done
  for u in "http://$PREFILL_IP:$SERVER_PORT" "http://$DECODE_IP:$SERVER_PORT" "http://$PREFILL_IP:$LB_PORT"; do printf '%s health=%s\n' "$u" "$(curl -s -m 5 -o /dev/null -w '%{http_code}' $u/health)"; done ;;
*) echo "usage: start <recipe> <run> | stop | status" >&2; exit 2 ;;
esac
