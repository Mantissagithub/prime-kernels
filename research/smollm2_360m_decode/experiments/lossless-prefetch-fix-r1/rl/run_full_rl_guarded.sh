#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: run_full_rl_guarded.sh <baseline|candidate> <fresh-run-id>" >&2
  exit 2
fi
arm=$1
run_id=$2
if [ "$arm" != baseline ] && [ "$arm" != candidate ]; then
  echo "invalid arm: $arm" >&2
  exit 2
fi
if [[ ! "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "invalid run-id: $run_id" >&2
  exit 2
fi

repo=/home/pradheep/tiny-prl/prime-rl
work=$repo/outputs/megakernel-20261025/lossless-full-kernel-integration-r1
runner=$repo/outputs/megakernel-20261025/runtime/run_gpu.sh
driver=$work/run_full_rl_case.py
guard=$work/results/guard-$run_id-$arm
if [ -e "$guard" ] || [ -e "$work/runs/$run_id-$arm" ] || [ -e "$work/results/$run_id-$arm" ]; then
  echo "refusing to reuse run paths" >&2
  exit 1
fi
bash "$runner" "lossless-fullrl-$run_id-$arm" "$guard" \
  "$driver" --arm "$arm" --run-id "$run_id"
