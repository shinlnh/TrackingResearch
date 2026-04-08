#!/usr/bin/env python3
"""Run MyTracker on Jetson Nano via SSH with optional profile selection."""

import argparse
import subprocess
import sys


EXPERIMENTS = {
    "main": {
        "lasot_headtail40": ("eco_verified_otb936_lasot_headtail40", "verified_otb936_main (run 953)"),
        "lasot": ("eco_verified_otb936_lasot", "verified_otb936_main (run 953)"),
        "lasot_first20": ("eco_verified_otb936_lasot_first20", "verified_otb936_main (run 953)"),
        "otb": ("eco_verified_otb936_otb", "verified_otb936_main (run 953)"),
    },
    "run_update": {
        "lasot_headtail40": ("eco_verified_otb936_run_update_lasot_headtail40", "verified_otb936_run_update (run 954)"),
        "lasot": ("eco_verified_otb936_run_update_lasot", "verified_otb936_run_update (run 954)"),
        "lasot_first20": ("eco_verified_otb936_run_update_lasot_first20", "verified_otb936_run_update (run 954)"),
        "otb": ("eco_verified_otb936_run_update_otb", "verified_otb936_run_update (run 954)"),
    },
}


def run_mytracker_on_jetson(host, port, password, dataset="lasot_headtail40", debug=0, threads=0, profile="main"):
    """Run MyTracker on Jetson Nano."""
    experiment, param_desc = EXPERIMENTS[profile][dataset]

    remote_cmd = f"""
set -e
cd ~/HELIOS/TransTResearch 2>/dev/null || cd ~/TransTResearch 2>/dev/null || {{ echo "Project directory not found"; exit 1; }}

export PYTHONUNBUFFERED=1
export PYTHONWARNINGS='ignore::FutureWarning'

if [ -f venv312/bin/activate ]; then
    source venv312/bin/activate
elif [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

echo "Starting MyTracker (ECO) on Jetson Nano..."
echo "Dataset: {dataset}"
echo "Profile: {profile}"
echo "Parameters: {param_desc}"
echo "Debug: {debug}"
echo "Threads: {threads}"
echo ""

python MyECOTracker/pytracking/pytracking/run_experiment.py \\
    myexperiments {experiment} \\
    --debug {debug} \\
    --threads {threads}

echo ""
echo "MyTracker run completed!"
"""

    print("=" * 60)
    print("Running MyTracker on Jetson Nano")
    print("=" * 60)
    print(f"Host: {host}:{port}")
    print(f"Dataset: {dataset}")
    print(f"Profile: {profile}")
    print(f"Debug: {debug}")
    print(f"Threads: {threads}")
    print("=" * 60)
    print()

    try:
        try:
            cmd = f'sshpass -p "{password}" ssh -p {port} helios@{host} bash -s'
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception:
            cmd = f'ssh -p {port} helios@{host} bash -s'
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

        stdout, _ = process.communicate(input=remote_cmd, timeout=3600)

        for line in stdout.split("\n"):
            if line.strip():
                print(line)

        if process.returncode == 0:
            print()
            print("=" * 60)
            print("MyTracker execution completed successfully!")
            print("=" * 60)
            return 0

        print()
        print("=" * 60)
        print("MyTracker execution failed!")
        print("=" * 60)
        return process.returncode
    except subprocess.TimeoutExpired:
        process.kill()
        print("Timeout: Tracker execution took too long")
        return 1
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MyTracker on Jetson Nano")
    parser.add_argument("--host", default="192.168.1.162", help="Jetson Nano IP address")
    parser.add_argument("--port", type=int, default=22, help="SSH port")
    parser.add_argument("--password", default="041209", help="SSH password")
    parser.add_argument(
        "--dataset",
        default="lasot_headtail40",
        choices=sorted(EXPERIMENTS["main"].keys()),
        help="Dataset to run on",
    )
    parser.add_argument(
        "--profile",
        default="main",
        choices=sorted(EXPERIMENTS.keys()),
        help="Tracker profile to run",
    )
    parser.add_argument("--debug", type=int, default=0, help="Debug level")
    parser.add_argument("--threads", type=int, default=0, help="Number of threads")

    args = parser.parse_args()

    sys.exit(
        run_mytracker_on_jetson(
            args.host,
            args.port,
            args.password,
            args.dataset,
            args.debug,
            args.threads,
            args.profile,
        )
    )
