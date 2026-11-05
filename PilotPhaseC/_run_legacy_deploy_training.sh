#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
LOG="PilotPhaseC/_legacy_deploy_training.log"
: > "$LOG"

run() {
  echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) $* =====" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
}

run $PY -m PilotPhaseC.src.train_classifier --participant ring04_pid101 --lineage all_continuous --label-mode four_class --force-reload --no-legacy-fallback
run $PY -m PilotPhaseC.src.train_classifier --participant ring04_pid101 --lineage all_continuous --label-mode binary_good_bad --force-reload --no-legacy-fallback
run $PY -m PilotPhaseC.src.train_classifier --participant ring03_pid102 --lineage all_continuous --label-mode four_class --force-reload --no-legacy-fallback
run $PY -m PilotPhaseC.src.train_classifier --participant ring03_pid102 --lineage all_continuous --label-mode binary_good_bad --force-reload --no-legacy-fallback
run $PY -m PilotPhaseC.src.train_classifier --generalized --lineage all_continuous --label-mode four_class --force-reload --no-legacy-fallback
run $PY -m PilotPhaseC.src.train_classifier --generalized --lineage all_continuous --label-mode binary_good_bad --force-reload --no-legacy-fallback
run $PY -m PilotPhaseC.src.summarize_results

echo "ALL_DONE" | tee -a "$LOG"
