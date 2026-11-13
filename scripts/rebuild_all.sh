#!/usr/bin/env bash
# Rebuild everything from the raw meter data: notebooks 10-60, the caches and customer_status, then every
# analysis script in dependency order. Stops at the first failure.
#
# Run from anywhere:
#     scripts/rebuild_all.sh            # all steps
#     scripts/rebuild_all.sh 9          # resume from step 9 (see the step list below)
#
# Notebook 10 writes the raw meter data into avangrid.db only when the file doesn't exist yet, so to
# pick up new or changed meter data, move the old onedrive-export/Data/avangrid.db aside first.
# Stop anything that has avangrid.db open (notebooks, the battery web app) before running.
# Each step's output goes to onedrive-export/Data/results/rebuild_logs/<start time>/.

set -euo pipefail
cd "$(dirname "$0")/.."

STEPS=(
  "nb   10_exploratory-data-analysis.ipynb"
  "nb   20_exploratory_data_analysis_p2.ipynb"
  "nb   30_define-variables.ipynb"
  "nb   40_identify-technology-customers.ipynb"
  "nb   50_usage-stabilization.ipynb"
  "nb   60_mid-year-changes.ipynb"
  "cli  cache-build"
  "cli  status-build"
  "py   scripts/predict_technologies.py"
  "py   scripts/predict_with_signatures.py"
  "py   scripts/predict_with_peaks.py"
  "py   scripts/improve_step1.py"
  "py   scripts/improve_step2.py"
  "py   scripts/refresh_predictions.py"
  "py   scripts/appliance_calibration.py"
  "py   scripts/appliance_inventory.py"
  "py   scripts/segmentation.py"
  "py   scripts/segmentation_scree.py"
  "py   scripts/battery_simulation.py"
)
# Not run: scripts/review_sample.py draws a new blind hand-labelling sample and overwrites the key of the
# existing one (results/review_key.parquet), so run it only when starting a new review.

START=${1:-1}
LOG_DIR="onedrive-export/Data/results/rebuild_logs/$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "$LOG_DIR"
T0=$(date +%s)

for i in "${!STEPS[@]}"; do
  n=$((i + 1))
  (( n < START )) && continue
  read -r kind target <<< "${STEPS[$i]}"
  log="$LOG_DIR/$(printf '%02d' "$n")_$(basename "$target" | sed 's/\.[a-z]*$//').log"
  echo "[$(date +%H:%M:%S)] step $n/${#STEPS[@]}: $target"
  s=$(date +%s)
  case $kind in
    nb)  cmd=(python3 -m avangrid run "$target") ;;
    cli) cmd=(python3 -m avangrid "$target") ;;
    py)  cmd=(python3 "$target") ;;
  esac
  if ! "${cmd[@]}" > "$log" 2>&1; then
    echo "FAILED at step $n ($target). Last lines of $log:"
    tail -20 "$log"
    echo "Fix the problem, then resume with: scripts/rebuild_all.sh $n"
    exit 1
  fi
  echo "           done in $(( ($(date +%s) - s) / 60 )) min $(( ($(date +%s) - s) % 60 )) s"
done
echo "All steps done in $(( ($(date +%s) - T0) / 60 )) min. Logs: $LOG_DIR"
