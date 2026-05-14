#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/HELIOS/MyECOTracker}"
REPORT_ROOT="${REPORT_ROOT:-$PROJECT_ROOT/jetson/reports}"
TRACKER_NAME="${TRACKER_NAME:-eco}"
SEQUENCE_FILE="${SEQUENCE_FILE:-$SCRIPT_DIR/lasot_headtail40_sequences.txt}"
DATASET_NAME="${DATASET_NAME:-lasot}"
DATASET_LABEL="${DATASET_LABEL:-lasot_headtail40}"
PROFILE="${1:-main}"

case "$PROFILE" in
  main)
    PARAMETER_NAME="${PARAMETER_NAME:-verified_otb936_main}"
    RUN_ID="${RUN_ID:-953}"
    RUN_MODE="${RUN_MODE:-lasot_headtail40}"
    DEFAULT_REPORT_TAG="verified_otb936_main_lasot_headtail40"
    ;;
  run_update)
    PARAMETER_NAME="${PARAMETER_NAME:-verified_otb936_run_update}"
    RUN_ID="${RUN_ID:-954}"
    RUN_MODE="${RUN_MODE:-lasot_headtail40_run_update}"
    DEFAULT_REPORT_TAG="verified_otb936_run_update_lasot_headtail40"
    ;;
  *)
    echo "Usage: $0 [main|run_update]" >&2
    exit 1
    ;;
esac

REPORT_TAG="${REPORT_TAG:-$DEFAULT_REPORT_TAG}"
REPORT_DIR="${REPORT_DIR:-$REPORT_ROOT/$REPORT_TAG}"
RESULTS_DIR="$PROJECT_ROOT/pytracking/pytracking/tracking_results/$TRACKER_NAME/${PARAMETER_NAME}_$(printf '%03d' "$RUN_ID")"
REPORT_NAME="${REPORT_NAME:-${DATASET_LABEL}_${TRACKER_NAME}_${PARAMETER_NAME}_${RUN_ID}}"
RESULT_PLOT_DIR="$PROJECT_ROOT/pytracking/pytracking/result_plots/$REPORT_NAME"

# shellcheck disable=SC1090
source "$SCRIPT_DIR/activate_verified936_env.sh"

mkdir -p "$REPORT_DIR"

clean_subset_results() {
  export RESULTS_DIR SEQUENCE_FILE
  python3 - <<'PY'
import os
from pathlib import Path

results_dir = Path(os.environ["RESULTS_DIR"])
sequence_file = Path(os.environ["SEQUENCE_FILE"])
results_dir.mkdir(parents=True, exist_ok=True)

sequences = [line.strip() for line in sequence_file.read_text(encoding="utf-8").splitlines() if line.strip()]
removed = 0
for seq_name in sequences:
    for suffix in (".txt", "_time.txt", "_object_presence_scores.txt"):
        path = results_dir / f"{seq_name}{suffix}"
        if path.exists():
            path.unlink()
            removed += 1

print(f"removed_subset_result_files={removed}")
print(f"subset_sequences={len(sequences)}")
PY
}

export_reports() {
  python3 "$PROJECT_ROOT/pytracking/pytracking/util_scripts/export_otb_sequence_metrics_csv.py" \
    --tracker-name "$TRACKER_NAME" \
    --parameter-name "$PARAMETER_NAME" \
    --run-id "$RUN_ID" \
    --dataset-name "$DATASET_NAME" \
    --sequence-file "$SEQUENCE_FILE" \
    --dataset-label "$DATASET_LABEL" \
    --report-name "$REPORT_NAME" \
    --rows-csv "$REPORT_DIR/per_sequence_metrics.csv" \
    --summary-csv "$REPORT_DIR/summary.csv"

  if [ -f "$RESULT_PLOT_DIR/eval_data.pkl" ]; then
    cp "$RESULT_PLOT_DIR/eval_data.pkl" "$REPORT_DIR/"
  fi
}

clean_subset_results | tee "$REPORT_DIR/cleanup.log"

{
  echo "[lasot-headtail40] report_dir=$REPORT_DIR"
  echo "[lasot-headtail40] results_dir=$RESULTS_DIR"
  echo "[lasot-headtail40] tracker_name=$TRACKER_NAME"
  echo "[lasot-headtail40] parameter_name=$PARAMETER_NAME"
  echo "[lasot-headtail40] run_id=$RUN_ID"
  echo "[lasot-headtail40] dataset_name=$DATASET_NAME"
  echo "[lasot-headtail40] dataset_label=$DATASET_LABEL"
  echo "[lasot-headtail40] sequence_file=$SEQUENCE_FILE"
  echo "[lasot-headtail40] run_mode=$RUN_MODE"
  echo "[lasot-headtail40] started_at=$(date '+%Y-%m-%d %H:%M:%S')"
} | tee "$REPORT_DIR/job_meta.txt"

bash "$SCRIPT_DIR/run_verified936.sh" "$RUN_MODE" | tee "$REPORT_DIR/run_experiment.log"
export_reports | tee "$REPORT_DIR/export.log"

{
  echo "[lasot-headtail40] finished_at=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[lasot-headtail40] rows_csv=$REPORT_DIR/per_sequence_metrics.csv"
  echo "[lasot-headtail40] summary_csv=$REPORT_DIR/summary.csv"
} | tee -a "$REPORT_DIR/job_meta.txt"
