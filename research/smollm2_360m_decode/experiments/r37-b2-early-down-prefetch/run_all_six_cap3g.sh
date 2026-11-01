#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: run_all_six_cap3g.sh <unit> <guard-dir>" >&2
  exit 2
fi

unit=$1
guard_dir=$2
repo=/home/pradheep/tiny-prl/prime-rl
runner=$repo/outputs/megakernel-20261025/runtime/run_gpu.sh
bench=$repo/outputs/megakernel-20261025/r37-b2-early-down-prefetch/bench_r30_r37.py
output=$repo/outputs/megakernel-20261025/r37-b2-early-down-prefetch/results/all-six-r1.json
mkdir -p "$guard_dir" "$(dirname "$output")"

bash "$runner" "$unit" "$guard_dir" "$bench" --output "$output" &
runner_pid=$!
(
  peak=0
  tripped=0
  while kill -0 "$runner_pid" 2>/dev/null; do
    state=$(systemctl --user show --property=ActiveState --value "$unit.service" 2>/dev/null || true)
    if [ "$state" = active ]; then
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1 {print $1}')
      if [[ "$used" =~ ^[0-9]+$ ]]; then
        if [ "$used" -gt "$peak" ]; then peak=$used; fi
        if [ "$used" -gt 3072 ]; then
          tripped=1
          echo "GPU memory cap exceeded: ${used} MiB > 3072 MiB" >> "$guard_dir/gpu-cap.log"
          systemctl --user stop "$unit.service" >/dev/null 2>&1 || true
          break
        fi
      fi
    fi
    sleep 0.2
  done
  echo "peak_observed_mib=$peak cap_mib=3072 tripped=$tripped" >> "$guard_dir/gpu-cap.log"
  if [ "$tripped" -eq 1 ]; then touch "$guard_dir/GPU_CAP_EXCEEDED"; fi
) &
monitor_pid=$!
set +e
wait "$runner_pid"
runner_rc=$?
wait "$monitor_pid"
monitor_rc=$?
set -e
if [ -e "$guard_dir/GPU_CAP_EXCEEDED" ]; then exit 124; fi
if [ "$monitor_rc" -ne 0 ]; then exit "$monitor_rc"; fi
exit "$runner_rc"
