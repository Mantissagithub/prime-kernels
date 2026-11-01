#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: run_micro.sh <unit> <probe-dir> <tag> <benchmark-script> [args...]" >&2
  exit 2
fi

unit=$1
probe_dir=$2
tag=$3
benchmark_script=$4
shift 4

outer_guard="$probe_dir/guard-outer-$tag"
inner_guard="$probe_dir/guard-inner-$tag"
if [ -e "$outer_guard" ] || [ -e "$inner_guard" ]; then
  echo "Guard output already exists; refusing to reuse: $outer_guard or $inner_guard" >&2
  exit 2
fi

runtime_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=/home/pradheep/tiny-prl/prime-rl
exec bash "$runtime_dir/run_gpu.sh" "$unit" "$outer_guard" \
  "$repo/outputs/megakernel-20261025/r28-fast-stage-profile-r1/guard_3gib.py" \
  "$inner_guard" /home/pradheep/.venv/bin/python "$benchmark_script" "$@"
