#!/usr/bin/env bash
# On a prefill stall: full py-spy stacks of every prefill rank on host a, the collective traces of both roles,
# and the first divergent collective. Output under results_goal/hang_<ts>/.
RUN=${1:-goal_after_1}; TS=$(date -u +%Y%m%dT%H%M%S); OUT=$HOME/pd_disagg/spill/results_goal/hang_$TS; mkdir -p $OUT/ctrace_prefill $OUT/ctrace_decode
SSH="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
$SSH 'PY=/home/alisachen_google_com/venv/sglang/bin/py-spy; for p in $(pgrep -u alisachen_google_com -f "sglang::scheduler_TP"); do r=$(tr -d "\0" < /proc/$p/cmdline | grep -oE "TP[0-9]+" | head -1); echo "=== $r pid=$p"; sudo timeout 60 $PY dump --pid $p 2>&1 | sed "s#(.*/\(sglang_tpu/\|srt/\|site-packages/\)#(#"; done' > $OUT/prefill_stacks.txt 2>&1
sync_from_a() { rsync -a -e "ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no" --rsync-path='sudo rsync' sa_101411256518727260534@10.128.15.217:/home/alisachen_google_com/pd_disagg/results/ctrace/ $OUT/ctrace_prefill/; }
sync_from_a; cp $HOME/pd_disagg/results/ctrace/* $OUT/ctrace_decode/ 2>/dev/null
echo "== prefill ranks, main-thread top frames:"; awk '/^=== /{r=$2} /"MainThread"/{f=1;n=0;next} /^Thread /{f=0} f&&/\(/&&n<3{n++; printf "%s %s\n", r, $0}' $OUT/prefill_stacks.txt
echo "== prefill collective divergence:"; python3 $HOME/pd_disagg/spill/collective_diff.py $OUT/ctrace_prefill --context 8 | tee $OUT/prefill_diff.txt | head -60
echo "== decode collective divergence:"; python3 $HOME/pd_disagg/spill/collective_diff.py $OUT/ctrace_decode --context 4 | tee $OUT/decode_diff.txt | head -30
echo "saved under $OUT"
