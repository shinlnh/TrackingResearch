# OTB MATLAB Export

Tracker exported: `ToMP_plus3`
Source folder: `E:\Programming\C\C2P\Project\TransTResearch\pytracking\pytracking\tracking_results\tomp\tomp50_auc60_plus3_705`
Sequences exported: `100`

## Exported folders
- `txt_results/ToMP_plus3`: One bbox text file per sequence (`x,y,w,h`).
- `classic_otb/results/OPE/ToMP_plus3`: `.mat` files for classic OTB toolkit (`res` variable).
- `unified_otb/results/OPE_OTB`: `.mat` files for unified toolkit (`results` variable).
- `manifest.csv`: Per-sequence mapping and FPS.

## Use with classic OTB toolkit (PAMI13 style)
1. Copy folder `classic_otb/results/OPE/ToMP_plus3` into `<OTB_TOOLKIT_ROOT>/results/OPE/`.
2. Open `<OTB_TOOLKIT_ROOT>/configTrackers.m`.
3. Duplicate one existing tracker block and change:
   - tracker name -> `ToMP_plus3`
   - result path -> `./results/OPE/ToMP_plus3/`
4. Add `ToMP_plus3` to the tracker list used by `perfPlot`.

## Use with unified OTB toolkit
1. Copy all `.mat` files from `unified_otb/results/OPE_OTB/` into `<UNIFIED_TOOLKIT_ROOT>/results/OPE_OTB/`.
2. Open `<UNIFIED_TOOLKIT_ROOT>/seqs/config_trackers.m`.
3. Add a tracker entry named `ToMP_plus3` pointing to `results/OPE_OTB`.
4. Run evaluation/plot scripts with tracker list containing `ToMP_plus3` and baselines (e.g. MDNet, CCOT, DeepSRDCF).
