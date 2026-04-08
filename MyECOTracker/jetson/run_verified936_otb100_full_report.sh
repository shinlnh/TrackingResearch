#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/HELIOS/MyECOTracker}"
REPORT_ROOT="${REPORT_ROOT:-$PROJECT_ROOT/jetson/reports}"
REPORT_TAG="${REPORT_TAG:-verified_otb936_main_otb100_full}"
REPORT_DIR="${REPORT_DIR:-$REPORT_ROOT/$REPORT_TAG}"
TRACKER_NAME="${TRACKER_NAME:-eco}"
PARAMETER_NAME="${PARAMETER_NAME:-verified_otb936_main}"
RUN_ID="${RUN_ID:-953}"
DATASET_NAME="${DATASET_NAME:-otb}"
RESULTS_DIR="$PROJECT_ROOT/pytracking/pytracking/tracking_results/$TRACKER_NAME/${PARAMETER_NAME}_$(printf '%03d' "$RUN_ID")"
RESULT_PLOT_DIR="$PROJECT_ROOT/pytracking/pytracking/result_plots/${DATASET_NAME}_${TRACKER_NAME}_${PARAMETER_NAME}_${RUN_ID}"

# shellcheck disable=SC1090
source "$SCRIPT_DIR/activate_verified936_env.sh"

mkdir -p "$REPORT_DIR"

clean_otb_results() {
  export RESULTS_DIR
  python3 - <<'PY'
import os
from pathlib import Path

from pytracking.evaluation.otbdataset import OTBDataset

results_dir = Path(os.environ["RESULTS_DIR"])
results_dir.mkdir(parents=True, exist_ok=True)

removed = 0
dataset = OTBDataset().get_sequence_list()
for seq in dataset:
    for suffix in (".txt", "_time.txt", "_object_presence_scores.txt"):
        path = results_dir / (seq.name + suffix)
        if path.exists():
            path.unlink()
            removed += 1

print("removed_otb_result_files={}".format(removed))
PY
}

export_reports() {
  python3 "$PROJECT_ROOT/pytracking/pytracking/util_scripts/export_otb_sequence_metrics_csv.py" \
    --tracker-name "$TRACKER_NAME" \
    --parameter-name "$PARAMETER_NAME" \
    --run-id "$RUN_ID" \
    --dataset-name "$DATASET_NAME" \
    --report-name "${DATASET_NAME}_${TRACKER_NAME}_${PARAMETER_NAME}_${RUN_ID}" \
    --rows-csv "$REPORT_DIR/per_sequence_metrics.csv" \
    --summary-csv "$REPORT_DIR/summary.csv"

  if [ -f "$RESULT_PLOT_DIR/eval_data.pkl" ]; then
    cp "$RESULT_PLOT_DIR/eval_data.pkl" "$REPORT_DIR/"
  fi
}

clean_otb_results

{
  echo "[full-otb] report_dir=$REPORT_DIR"
  echo "[full-otb] results_dir=$RESULTS_DIR"
  echo "[full-otb] tracker_name=$TRACKER_NAME"
  echo "[full-otb] parameter_name=$PARAMETER_NAME"
  echo "[full-otb] run_id=$RUN_ID"
  echo "[full-otb] dataset_name=$DATASET_NAME"
  echo "[full-otb] started_at=$(date '+%Y-%m-%d %H:%M:%S')"
} | tee "$REPORT_DIR/job_meta.txt"

export PROJECT_ROOT TRACKER_NAME PARAMETER_NAME RUN_ID DATASET_NAME
python3 - <<'PY' | tee "$REPORT_DIR/run_experiment.log"
import os
import sys

project_root = os.environ["PROJECT_ROOT"]
tracker_name = os.environ["TRACKER_NAME"]
parameter_name = os.environ["PARAMETER_NAME"]
run_id = int(os.environ["RUN_ID"])
dataset_name = os.environ["DATASET_NAME"]

pytracking_root = os.path.join(project_root, "pytracking")
if pytracking_root not in sys.path:
    sys.path.append(pytracking_root)

from pytracking.evaluation import Tracker, get_dataset
from pytracking.evaluation.running import run_dataset

dataset = get_dataset(dataset_name)
trackers = [Tracker(tracker_name, parameter_name, run_id)]
run_dataset(dataset, trackers, debug=0, threads=0, visdom_info={"use_visdom": False})
PY

export_reports | tee "$REPORT_DIR/export.log"

{
  echo "[full-otb] finished_at=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[full-otb] rows_csv=$REPORT_DIR/per_sequence_metrics.csv"
  echo "[full-otb] summary_csv=$REPORT_DIR/summary.csv"
} | tee -a "$REPORT_DIR/job_meta.txt"
