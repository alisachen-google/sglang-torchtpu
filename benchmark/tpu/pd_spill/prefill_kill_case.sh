#!/usr/bin/env bash
# Mirror robustness case: kill the prefill while the decode holds offloaded KV (after a fill), restart the
# prefill alone, then recall. The decode must survive the prefill's death and the recall must complete
# through the new prefill (which has no record of the decode's offers); no engine error on either role.
cd ~/pd_disagg/spill
echo $$ > logs/prefill_kill_case.pid
L=logs/prefill_kill_case.log
RUN=tipkillp; A=10.128.15.239; B=10.128.0.20
SSH="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no"
log() { echo "$(date -u '+%F %T') $*" >> $L; }
PREFILL_IP=$A DECODE_IP=$B bash pd_e18f_vw.sh stop > /dev/null 2>&1; sleep 8
PREFILL_IP=$A DECODE_IP=$B bash pd_e18f_vw.sh start $HOME/pd_disagg/recipes/v6c_long/victim/gpt-oss_C_victim_tip.json $RUN > logs/${RUN}_start.log 2>&1
for i in $(seq 1 60); do a=$($SSH sa_101411256518727260534@$A "sudo grep -c 'after warmup' /home/alisachen_google_com/pd_disagg/results/runs/$RUN/prefill/server.log 2>/dev/null" 2>/dev/null | head -1); b=$($SSH sa_101411256518727260534@$B "sudo grep -c 'after warmup' /home/alisachen_google_com/pd_disagg/results/runs/$RUN/decode/server.log 2>/dev/null" 2>/dev/null | head -1); [ "${a:-0}" -ge 8 ] && [ "${b:-0}" -ge 8 ] && break; sleep 15; done
log "pair warm (prefill=$a decode=$b)"
curl -s -m 60 -X POST http://$B:30000/flush_cache -o /dev/null; curl -s -m 60 -X POST http://$A:30000/flush_cache -o /dev/null
~/venv/sglang/bin/python victim_recall.py --sessions 12 --prefix 131072 --new 8192 --out-tokens 16 --concurrency 4 --seed 6 --run $RUN --phase fill --label killp_fill --out results_goal/${RUN}_fill.jsonl --lb http://$A:8080 > logs/${RUN}_fill.log 2>&1
landed=$($SSH sa_101411256518727260534@$B "sudo grep -h 'TP0.*victim-push\] role=decode received=' /home/alisachen_google_com/pd_disagg/results/runs/$RUN/decode/server.log | tail -1 | grep -oE 'landed=[0-9]+'")
log "fill done; decode holds offloaded KV ($landed); starting the recall phase and killing the prefill 20 s in"
~/venv/sglang/bin/python victim_recall.py --sessions 12 --prefix 131072 --new 8192 --out-tokens 16 --concurrency 4 --seed 6 --run $RUN --phase recall --label killp_recall --out results_goal/${RUN}_recall.jsonl --lb http://$A:8080 > logs/${RUN}_recall.log 2>&1 &
sleep 20
PREFILL_IP=$A DECODE_IP=$B bash pd_e18f_vw.sh stop-prefill > /dev/null 2>&1
log "prefill killed; waiting 60 s"
sleep 60
pb=$($SSH sa_101411256518727260534@$B 'pgrep -u alisachen_google_com -fc scheduler_T[P]')
ex=$($SSH sa_101411256518727260534@$B "sudo grep -c 'Scheduler hit an exception' /home/alisachen_google_com/pd_disagg/results/runs/$RUN/decode/server.log")
h=$(curl -s -m 10 -o /dev/null -w '%{http_code}' http://$B:30000/health)
log "after prefill death: decode schedulers=$pb exceptions=$ex health=$h"
wait
log "recall client ended: $(grep -oE '\"errors\": [0-9]+|\"requests\": [0-9]+' logs/${RUN}_recall.log | head -2 | tr '\n' ' ')"
PREFILL_IP=$A DECODE_IP=$B bash pd_e18f_vw.sh start-prefill $HOME/pd_disagg/recipes/v6c_long/victim/gpt-oss_C_victim_tip.json ${RUN}_prefill2 >> logs/${RUN}_start.log 2>&1
for i in $(seq 1 60); do a=$($SSH sa_101411256518727260534@$A "sudo grep -c 'after warmup' /home/alisachen_google_com/pd_disagg/results/runs/${RUN}_prefill2/prefill/server.log 2>/dev/null" 2>/dev/null | head -1); [ "${a:-0}" -ge 8 ] && break; sleep 15; done
log "prefill restarted (warm=$a); LB health $(curl -s -m 10 -o /dev/null -w '%{http_code}' http://$A:8080/health)"
~/venv/sglang/bin/python victim_recall.py --sessions 12 --prefix 131072 --new 8192 --out-tokens 16 --concurrency 4 --seed 6 --run $RUN --phase recall --label killp_recall2 --out results_goal/${RUN}_recall2.jsonl --lb http://$A:8080 > logs/${RUN}_recall2.log 2>&1
log "recall after prefill restart: $(grep -oE '\"errors\": [0-9]+|\"requests\": [0-9]+|\"prefill_hit_rate\": [0-9.]+' logs/${RUN}_recall2.log | head -3 | tr '\n' ' ')"
pb=$($SSH sa_101411256518727260534@$B 'pgrep -u alisachen_google_com -fc scheduler_T[P]')
ex=$($SSH sa_101411256518727260534@$B "sudo grep -c 'Scheduler hit an exception' /home/alisachen_google_com/pd_disagg/results/runs/$RUN/decode/server.log")
pa=$($SSH sa_101411256518727260534@$A 'pgrep -u alisachen_google_com -fc scheduler_T[P]')
exa=$($SSH sa_101411256518727260534@$A "sudo grep -c 'Scheduler hit an exception' /home/alisachen_google_com/pd_disagg/results/runs/${RUN}_prefill2/prefill/server.log")
pulls=$($SSH sa_101411256518727260534@$A "sudo grep -h 'TP0.*prefix-pull\]' /home/alisachen_google_com/pd_disagg/results/runs/${RUN}_prefill2/prefill/server.log | tail -1 | grep -oE 'pulls.: [0-9]+'")
log "end: decode schedulers=$pb exceptions=$ex; new prefill schedulers=$pa exceptions=$exa recalls($pulls)"
echo "DONE" >> $L
