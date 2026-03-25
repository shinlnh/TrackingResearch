from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OtherTracker.tools.fps_benchmark import main as benchmark_main


if __name__ == "__main__":
    raise SystemExit(
        benchmark_main(
            [
                "--backend",
                "pytracking",
                "--tracker-name",
                "tomp",
                "--parameter",
                "tomp50",
                "--display-name",
                "ToMP-50",
                *sys.argv[1:],
            ]
        )
    )
