MEEM local benchmark notes.

Current status:

- Local source is mirrored under `upstream/` from `JHvisionchen/MEEM-matlab` on GitHub.
- The original BU project page is no longer live, but an archived copy still lists `MEEM_v1.1_release.zip`.
- Local OTB runner entry point is `run_meem_otb.m`.
- Compatibility shims were added for modern MATLAB:
  - `svmtrain.m` wraps legacy calls onto `fitcsvm`
  - `resize_safe.m` falls back to `imresize`
  - `calcIIF_safe.m` falls back to MATLAB code when the bundled MEX is missing DLLs

Latest local full OTB100 run:

- Output dir: `otb100_fps_full_20260325`
- Summary: `valid_sequences = 100`, `fps_global = 12.8467525079035`, `fps_weighted_by_frames = 11.9411746790807`
