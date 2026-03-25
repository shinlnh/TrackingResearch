# OTB MATLAB Export

Tracker exported: `MyTrackerECO`
Source folder: `E:\Programming\Python\Project\AI\CV\Tracking\TransTResearch\MyECOTracker\pytracking\pytracking\tracking_results\eco\embed_v2_932`
Sequences exported: `100`

## Exported folders
- `txt_results/MyTrackerECO`: One bbox text file per sequence (`x,y,w,h`).
- `classic_otb/results/OPE/MyTrackerECO`: `.mat` files for classic OTB toolkit (`res` variable).
- `unified_otb/results/OPE_OTB`: `.mat` files for unified toolkit (`results` variable).
- `manifest.csv`: Per-sequence mapping and FPS.

## Use with classic OTB toolkit (PAMI13 style)
1. Copy folder `classic_otb/results/OPE/MyTrackerECO` into `<OTB_TOOLKIT_ROOT>/results/OPE/`.
2. Open `<OTB_TOOLKIT_ROOT>/configTrackers.m`.
3. Duplicate one existing tracker block and change:
   - tracker name -> `MyTrackerECO`
   - result path -> `./results/OPE/MyTrackerECO/`
4. Add `MyTrackerECO` to the tracker list used by `perfPlot`.

## Use with unified OTB toolkit
1. Copy all `.mat` files from `unified_otb/results/OPE_OTB/` into `<UNIFIED_TOOLKIT_ROOT>/results/OPE_OTB/`.
2. Open `<UNIFIED_TOOLKIT_ROOT>/seqs/config_trackers.m`.
3. Add a tracker entry named `MyTrackerECO` pointing to `results/OPE_OTB`.
4. Run evaluation/plot scripts with tracker list containing `MyTrackerECO` and baselines (e.g. MDNet, CCOT, DeepSRDCF).
