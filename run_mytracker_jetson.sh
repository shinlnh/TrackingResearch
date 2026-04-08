#!/bin/bash

# Run MyTracker on Jetson Nano
# This script runs ECO Tracker (MyTrackerECO) on LaSOT dataset
# Usage: ./run_mytracker_jetson.sh [host] [port] [profile]

set -e

# Configuration
JETSON_HOST="${1:-helios@192.168.1.162}"
JETSON_PORT="${2:-22}"
PROFILE="${3:-main}"
REMOTE_PROJECT_DIR="/home/helios/TransTResearch"
LOCAL_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$PROFILE" in
    main)
        EXPERIMENT="eco_verified_otb936_lasot_headtail40"
        PARAM_DESC="verified_otb936_main (run 953)"
        ;;
    run_update)
        EXPERIMENT="eco_verified_otb936_run_update_lasot_headtail40"
        PARAM_DESC="verified_otb936_run_update (run 954)"
        ;;
    *)
        echo "Unknown profile: $PROFILE" >&2
        echo "Usage: ./run_mytracker_jetson.sh [host] [port] [main|run_update]" >&2
        exit 1
        ;;
esac

echo "==========================================="
echo "Running MyTracker on Jetson Nano"
echo "==========================================="
echo "Host: $JETSON_HOST"
echo "Port: $JETSON_PORT"
echo "Profile: $PROFILE"
echo "Remote Project Dir: $REMOTE_PROJECT_DIR"
echo ""

# Create remote command
REMOTE_CMD="
set -e
cd $REMOTE_PROJECT_DIR
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS='ignore::FutureWarning'

# Activate virtual environment if exists
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
elif [ -f venv312/bin/activate ]; then
    source venv312/bin/activate
fi

# Run MyTracker
echo 'Starting MyTracker (ECO) on LaSOT dataset...'
echo 'Parameters: $PARAM_DESC, Dataset: LaSOT (head 20 + tail 20)'
echo ''

python MyECOTracker/pytracking/pytracking/run_experiment.py \\
    myexperiments $EXPERIMENT \\
    --debug 0 \\
    --threads 0

echo 'MyTracker run completed!'
"

# Execute via SSH
echo "Connecting to Jetson Nano and executing MyTracker..."
echo ""

ssh -p "$JETSON_PORT" "$JETSON_HOST" "$REMOTE_CMD"

echo ""
echo "==========================================="
echo "MyTracker execution finished!"
echo "==========================================="
