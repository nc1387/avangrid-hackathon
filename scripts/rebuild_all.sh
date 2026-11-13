#!/usr/bin/env bash
# Rebuild everything from the raw meter data by running notebooks 10-90 in order. Stops at the first failure.
#
# Run from anywhere:
#     scripts/rebuild_all.sh            # all notebooks
#     scripts/rebuild_all.sh 6          # resume from step 6 (see the step list below)
#
# Notebook 10 writes the raw meter data into avangrid.db only when the file doesn't exist yet, so to
# pick up new or changed meter data, move the old onedrive-export/Data/avangrid.db aside first.
# Stop anything that has avangrid.db open (notebooks, the battery web app) before running.
# Notebook 50 holds every long-running model and takes about 2 hours; the rest take about 15 minutes together.
# Each notebook is executed in place (outputs saved); its log goes to
# onedrive-export/Data/results/rebuild_logs/<start time>/.

set -euo pipefail
cd "$(dirname "$0")/.."

STEPS=(
  10_load-data.ipynb
  20_clean-data.ipynb
  30_time-of-day-and-weather.ipynb
  40_customer-features.ipynb
  45_change-detection.ipynb
  50_technology-models.ipynb
  60_appliances.ipynb
  70_segmentation.ipynb
  80_battery-opportunity.ipynb
  90_usage-stabilization.ipynb
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
  target=${STEPS[$i]}
  log="$LOG_DIR/$(printf '%02d' "$n")_${target%.ipynb}.log"
  echo "[$(date +%H:%M:%S)] step $n/${#STEPS[@]}: $target"
  s=$(date +%s)
  # Up to 2 hours per cell: notebook 50's longest stages run for about half an hour each.
  if ! python3 -m avangrid run "$target" --timeout 7200 > "$log" 2>&1; then
    echo "FAILED at step $n ($target). Last lines of $log:"
    tail -20 "$log"
    echo "Fix the problem, then resume with: scripts/rebuild_all.sh $n"
    exit 1
  fi
  echo "           done in $(( ($(date +%s) - s) / 60 )) min $(( ($(date +%s) - s) % 60 )) s"
done
echo "All steps done in $(( ($(date +%s) - T0) / 60 )) min. Logs: $LOG_DIR"
