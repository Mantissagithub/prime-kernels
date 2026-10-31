#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PYTHON=/home/pradheep/.venv/bin/python
UV=/home/pradheep/.local/bin/uv
RUN_GPU="$REPO/outputs/megakernel-20261025/runtime/run_gpu.sh"
"$UV" run --offline --no-project --python "$PYTHON" python "$HERE/preflight.py"
test -s "$HERE/build/libdecode360_r28_stage_profile.so"

cases=(seed17-b1-ctx32 seed17-b1-ctx128 seed17-b2-ctx32 seed17-b2-ctx128 synthetic-b1-ctx1024 synthetic-b2-ctx1024)
for case_id in "${cases[@]}"; do
  bash "$RUN_GPU" "r28-stage-profile-${case_id}" \
    "$HERE/results/runtime-guard-${case_id}" \
    "$HERE/guard_3gib.py" "$HERE/results/gpu-guard-${case_id}" \
    "$PYTHON" "$HERE/profile_bench_r28.py" \
    --output "$HERE/results/${case_id}.json" --warmups 2 --repeats 5 --blocks 3 --cases "$case_id"
done
"$UV" run --offline --no-project --python "$PYTHON" python "$HERE/analyze_stage_profile.py" \
  "$HERE/results/seed17-b1-ctx32.json" "$HERE/results/seed17-b1-ctx128.json" \
  "$HERE/results/seed17-b2-ctx32.json" "$HERE/results/seed17-b2-ctx128.json" \
  "$HERE/results/synthetic-b1-ctx1024.json" "$HERE/results/synthetic-b2-ctx1024.json" \
  --output "$HERE/results/stage-profile-summary.json"
