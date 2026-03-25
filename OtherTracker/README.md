# OtherTracker

This directory stores external tracker sources and local benchmark helpers for OTB FPS comparison.

Current setup:

- `CCOT`, `Staple`, `LCT`, `SAMF`, `CF2`, `MDNet`: cloned from upstream repositories.
- `DSST`, `SRDCF`, `SRDCFdecon`: downloaded from original project pages.
- `tools/fps_benchmark.py`: shared Python-only OTB FPS harness for OpenCV and pytracking trackers.
- `CSRT`, `KCF`: local OpenCV-based OTB runners built on the shared harness.
- `tools/extract_embedded_otb_fps.py`: extracts embedded FPS from existing OTB `.mat` files.
- `tools/otb_sequences.py`: reproduces OTB sequence frame ranges used by the MATLAB toolkit.

Notes:

- The benchmark layer is now intended to stay in Python. Each run writes one `*_summary.csv` whose `fps_global` is the mean FPS over sequences, i.e. the single number to move into MATLAB later.
- The frame-weighted alternative `total_frames / total_time_sec` is kept separately as `fps_weighted_by_frames`.
- Some legacy trackers such as `MDNet` require extra build steps before they can be executed locally.
- Several OTB result `.mat` files already contain per-sequence `fps`. These values can be summarized without rerunning the tracker code.
