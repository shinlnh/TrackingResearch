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
