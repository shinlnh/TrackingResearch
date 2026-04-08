# Verified Runs

## 2026-03-27

- Saved alias: `eco/verified_otb936`
- Source parameter file: `eco/embed_v2`
- Saved run id: `936`
- Verified environment: `venv312`
- Verified OTB100 summary: `AUC=67.4048`, `Precision=91.5277`, `Success50=84.7481`, `FPS_weighted=68.9708`
- Raw OTB results source: `MyECOTracker/pytracking/pytracking/tracking_results/eco/embed_v2_936`
- Saved run folder: `MyECOTracker/pytracking/pytracking/tracking_results/eco/verified_otb936_936`
- Summary CSV: `MyECOTracker/otb100result/mytrackereco_verified_otb936_run936_summary.csv`

Run again on OTB:

```powershell
venv312\Scripts\python.exe MyECOTracker\pytracking\pytracking\run_experiment.py myexperiments eco_verified_otb936_otb --debug 0 --threads 0
```

Run later on LaSOT:

```powershell
venv312\Scripts\python.exe MyECOTracker\pytracking\pytracking\run_experiment.py myexperiments eco_verified_otb936_lasot --debug 0 --threads 0
```

## 2026-04-03

- Saved Jetson main alias: `eco/verified_otb936_main`
- Source parameter file: `eco/jetson_fast_trt_rgb`
- Canonical Jetson run id: `953`
- Canonical Jetson display name: `MyTrackerECO-Main`
- Selected benchmark on `Girl + Walking2 + Woman`: `AUC=59.2330`, `FPS_avg_seq=27.5965`, `FPS_weighted=27.5805`
- Saved summary CSV on Jetson: `~/HELIOS/MyECOTracker/jetson/sweeps/verified_otb936_jetson_fast_trt_rgb.csv`
- Note: Jetson helper script `jetson/run_verified936.sh` and `myexperiments` now use this alias by default for future runs.

## 2026-04-04

- Added Jetson balanced alias: `eco/verified_otb936_run_update`
- Source parameter file: `eco/jetson_fast_trt_dual_acc` via `eco/jetson_fast_trt_rgb_run_update`
- Planned Jetson run id: `954`
- Goal: recover AUC over `verified_otb936_main` while staying in the realtime range on Jetson.
- Full OTB100 result for current best realtime point: `AUC=54.0044`, `Precision=74.4743`, `Success50=63.7959`, `FPS_weighted=22.7594`
- Saved summary CSV on Jetson: `~/HELIOS/MyECOTracker/jetson/reports/verified_otb936_dual_acc_otb100_full/summary.csv`
- Comparison:
  - `verified_otb936_main`: `AUC=49.5614`, `FPS_weighted=27.2438`
  - `verified_otb936_run_update` current best: `AUC=54.0044`, `FPS_weighted=22.7594`
  - `verified_otb936_trt`: `AUC=66.9997`, `FPS_weighted=5.7353`
