#!/usr/bin/env bash
# Keep collective-trace files from filling the disk: truncate any over 400 MB every 5 min.
while true; do
  for f in /home/alisachen_google_com/pd_disagg/results/ctrace/ctrace_r*.log; do
    [ -f "$f" ] && [ $(stat -c %s "$f") -gt 419430400 ] && : > "$f"
  done
  sleep 300
done
