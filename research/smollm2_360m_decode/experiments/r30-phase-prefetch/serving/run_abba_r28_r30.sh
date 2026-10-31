#!/usr/bin/env bash
# Usage: run_abba_r28_r30.sh [tag]
set -u
D=/home/pradheep/tiny-prl/prime-rl/outputs/megakernel-20261025/r30-phase-prefetch/serving
RUN=/home/pradheep/tiny-prl/prime-rl/outputs/megakernel-20261025/runtime/run_gpu.sh
PY=/home/pradheep/.venv/bin/python
H=$D/bench_serving_r28_r30.py
TAG=${1:-$(date +%Y%m%d-%H%M%S)}

run_arm() {
  local name=$1 arm=$2 out outer_guard inner_guard rc
  out="$D/$name-$TAG.json"
  if [ -e "$out" ]; then echo "refusing to overwrite $out" >&2; return 1; fi
  outer_guard="$D/guard-$name-$TAG"
  inner_guard="$D/guard-3gb-$name-$TAG"
  bash "$RUN" "r30-serving-$name-$TAG-$(date +%s)" "$outer_guard" \
    "$D/guard_3gb.py" "$inner_guard" "$PY" "$H" --arm "$arm" --out "$out" \
    > "$D/driver-$name-$TAG.log" 2>&1
  rc=$?
  if [ ! -d "$outer_guard" ]; then
    echo "$name arm=$arm refused before guard start (rc=$rc)" >&2
    return "$rc"
  fi
  echo "$(date +%T) $name arm=$arm rc=$rc"
  return "$rc"
}

run_arm A1-r28 r28 || exit 1
run_arm B1-r30 r30 || exit 1
run_arm B2-r30 r30 || exit 1
run_arm A2-r28 r28 || exit 1
echo "$(date +%T) sequence done; tag=$TAG"
