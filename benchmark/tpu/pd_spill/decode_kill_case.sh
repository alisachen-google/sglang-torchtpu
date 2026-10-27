#!/usr/bin/env bash
# Robustness case: kill the decode engine while recalls are in flight, then restart the decode alone.
# The prefill must survive both (no exception, stays healthy) and serve again once the decode is back.
cd ~/pd_disagg/spill
echo $$ > logs/decode_kill_case.pid
L=logs/decode_kill_case.log
RUN=tipkill; A=10.128.15.239; B=10.128.0.20; TOK=/mnt/data/tt/hf-cache/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a
SSH="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no"
log() { echo "$(date -u '+%F %T') $*" >> $L; }
PREFILL_IP=$A DECODE_IP=$B bash pd_e18f_vw.sh stop > /dev/null 2>&1; sleep 8
PREFILL_IP=$A DECODE_IP=$B bash pd_e18f_vw.sh start $HOME/pd_disagg/recipes/v6c_long/victim/gpt-oss_C_victim_tip.json $RUN > logs/${RUN}_start.log 2>&1
for i in $(seq 1 60); do a=$($SSH sa_101411256518727260534@$A "sudo grep -c 'after warmup' /home/alisachen_google_com/pd_disagg/results/runs/$RUN/prefill/server.log 2>/dev/null" 2>/dev/null | head -1); b=$($SSH sa_101411256518727260534@$B "sudo grep -c 'after warmup' /home/alisachen_google_com/pd_disagg/results/runs/$RUN/decode/server.log 2>/dev/null" 2>/dev/null | head -1); [ "${a:-0}" -ge 8 ] && [ "${b:-0}" -ge 8 ] && break; sleep 15; done
log "pair warm (prefill=$a decode=$b)"
curl -s -m 60 -X POST http://$B:30000/flush_cache -o /dev/null; curl -s -m 60 -X POST http://$A:30000/flush_cache -o /dev/null
~/venv/sglang/bin/python victim_recall.py --sessions 12 --prefix 131072 --new 8192 --out-tokens 16 --concurrency 4 --seed 5 --run $RUN --phase fill --label kill_fill --out results_goal/${RUN}_fill.jsonl --lb http://$A:8080 > logs/${RUN}_fill.log 2>&1
log "fill done; starting the recall phase and killing the decode 20 s in"
~/venv/sglang/bin/python victim_recall.py --sessions 12 --prefix 131072 --new 8192 --out-tokens 16 --concurrency 4 --seed 5 --run $RUN --phase recall --label kill_recall --out results_goal/${RUN}_recall.jsonl --lb http://$A:8080 > logs/${RUN}_recall.log 2>&1 &
sleep 20
PREFILL_IP=$A DECODE_IP=$B bash pd_e18f_vw.sh stop-decode > /dev/null 2>&1
log "decode killed; waiting 60 s"
sleep 60
pa=$($SSH sa_101411256518727260534@$A 'pgrep -u alisachen_google_com -fc scheduler_T[P]')
ex=$($SSH sa_101411256518727260534@$A "sudo grep -c 'Scheduler hit an exception' /home/alisachen_google_com/pd_disagg/results/runs/$RUN/prefill/server.log")
h=$(curl -s -m 10 -o /dev/null -w '%{http_code}' http://$A:30000/health)
log "after decode death: prefill schedulers=$pa exceptions=$ex health=$h"
wait
log "recall client ended: $(grep -oE '\"errors\": [0-9]+|\"requests\": [0-9]+' logs/${RUN}_recall.log | tr '\n' ' ')"
PREFILL_IP=$A DECODE_IP=$B bash pd_e18f_vw.sh start-decode $HOME/pd_disagg/recipes/v6c_long/victim/gpt-oss_C_victim_tip.json ${RUN}_decode2 >> logs/${RUN}_start.log 2>&1
for i in $(seq 1 60); do b=$($SSH sa_101411256518727260534@$B "sudo grep -c 'after warmup' /home/alisachen_google_com/pd_disagg/results/runs/${RUN}_decode2/decode/server.log 2>/dev/null" 2>/dev/null | head -1); [ "${b:-0}" -ge 8 ] && break; sleep 15; done
log "decode restarted (warm=$b); LB health $(curl -s -m 10 -o /dev/null -w '%{http_code}' http://$A:8080/health)"
~/venv/sglang/bin/python victim_recall.py --sessions 12 --prefix 131072 --new 8192 --out-tokens 16 --concurrency 4 --seed 5 --run $RUN --phase recall --label kill_recall2 --out results_goal/${RUN}_recall2.jsonl --lb http://$A:8080 > logs/${RUN}_recall2.log 2>&1
log "recall after restart: $(grep -oE '\"errors\": [0-9]+|\"requests\": [0-9]+|\"prefill_hit_rate\": [0-9.]+' logs/${RUN}_recall2.log | tr '\n' ' ')"
pa=$($SSH sa_101411256518727260534@$A 'pgrep -u alisachen_google_com -fc scheduler_T[P]')
ex=$($SSH sa_101411256518727260534@$A "sudo grep -c 'Scheduler hit an exception' /home/alisachen_google_com/pd_disagg/results/runs/$RUN/prefill/server.log")
log "end: prefill schedulers=$pa exceptions=$ex"
echo "DONE" >> $L
