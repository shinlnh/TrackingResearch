#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REPORT_TAG="${REPORT_TAG:-verified_otb936_run_update_otb100_full}"
export PARAMETER_NAME="${PARAMETER_NAME:-verified_otb936_run_update}"
export RUN_ID="${RUN_ID:-954}"

"$SCRIPT_DIR/run_verified936_otb100_full_report.sh"
