#!/usr/bin/env bash
# Victim-cache counters for a run on an arbitrary pair: PIP (prefill host) and DIP (decode host) via ssh.
RUN=${1:?run}; PIP=${PIP:-10.128.15.217}; DIP=${DIP:-10.128.0.90}
SSHP="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@$PIP"
SSHD="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@$DIP"
P=/home/alisachen_google_com/pd_disagg/results/runs/$RUN/prefill/server.log
D=/home/alisachen_google_com/pd_disagg/results/runs/$RUN/decode/server.log
$SSHP "sudo -n grep -h '\[victim-push\] role=prefill' $P 2>/dev/null | grep TP0 | tail -1 | sed 's/.*\[victim-push\]/  prefill push:/'"
$SSHP "sudo -n grep -h '\[prefix-pull\] room=' $P 2>/dev/null | grep -c TP0 | sed 's/^/  prefix-pull commits (TP0): /'"
$SSHP "sudo -n grep -h '\[prefix-pull\] room=' $P 2>/dev/null | grep TP0 | sed 's/.* pulled_tokens=\([0-9]*\).*/\1/' | awk '{s+=\$1} END {printf \"  prefix-pull tokens (TP0): %d\n\", s}'"
$SSHP "sudo -n grep -h '\[prefix-pull\].*\(not completed\|no device memory\|giving up\|READY covers\)' $P 2>/dev/null | grep TP0 | sed 's/.*\[prefix-pull\]//' | sed 's/room=[0-9]*//' | cut -c1-60 | sort | uniq -c | sort -rn | head -4 | sed 's/^/  prefix-pull not done: /'"
$SSHD "sudo -n grep -h '\[victim-push\] role=decode' $D 2>/dev/null | grep TP0 | tail -1 | sed 's/.*\[victim-push\]/  decode push:/'"
$SSHD "sudo -n grep -h '\[prefix-offer\] room=' $D 2>/dev/null | grep -c TP0 | sed 's/^/  decode offers (TP0): /'"
$SSHD "sudo -n grep -h '\[prefix-offer\] room=' $D 2>/dev/null | grep TP0 | grep -vc 'pages=0 ' | sed 's/^/  decode non-empty offers (TP0): /'"
for side in prefill:$PIP decode:$DIP; do
  n=${side%%:*}; ip=${side##*:}
  curl -s -m 10 "http://$ip:30000/metrics" | grep -E '^sglang:(prompt_tokens_total|cached_tokens_total|num_requests_total)' | sed 's/{[^}]*}//' | awk -v n=$n '{a[$1]+=$2} END {for (k in a) printf "  %s %s %.0f\n", n, k, a[k]}' | sort
done
