# Video Validation Commands

Use one file only:

- `run_video_validation_plot.ps1`

Recommended usage: pass one preset via `-Target`.

```powershell
powershell -ExecutionPolicy Bypass -File .\run_video_validation_plot.ps1 -Target otb_only_mytracker_success_plot
powershell -ExecutionPolicy Bypass -File .\run_video_validation_plot.ps1 -Target otb_only_mytracker_fps_avg
powershell -ExecutionPolicy Bypass -File .\run_video_validation_plot.ps1 -Target otb_full_tracker_success_plot
powershell -ExecutionPolicy Bypass -File .\run_video_validation_plot.ps1 -Target otb_full_tracker_fps_avg

powershell -ExecutionPolicy Bypass -File .\run_video_validation_plot.ps1 -Target lasot_only_mytracker_success_plot
powershell -ExecutionPolicy Bypass -File .\run_video_validation_plot.ps1 -Target lasot_only_mytracker_fps_avg
powershell -ExecutionPolicy Bypass -File .\run_video_validation_plot.ps1 -Target lasot_full_tracker_success_plot
powershell -ExecutionPolicy Bypass -File .\run_video_validation_plot.ps1 -Target lasot_full_tracker_fps_avg
```

Legacy usage is still supported:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_video_validation_plot.ps1 -Dataset otb -Scope only_mytracker -Metric success_plot
```

Launcher config:

- `overall_result/video/video_plot_config.json`

Launcher script:

- `run_video_validation_plot.ps1`
