# MDNet LaSOT Benchmark

Nguon ket qua:
- `tracking_results/*.txt` duoc copy tu `OtherTracker/pytracking/tracking_results_external_mdnet/MDNet/default`.
- `mdnet_lasot_testset_summary.csv` va `mdnet_lasot_testset_summary.txt` duoc sinh boi `ls/LaSOT_Evaluation_Toolkit/run_mdnet_testset_evaluation.m`.
- `benchmark/` chua plot va bang diem benchmark voi `MDNet` va `MyTracker`.

LaSOT test_set 280 sequence:
- `success_auc = 0.396980`
- `success50 = 0.434582`
- `precision20 = 0.372538`

Ghi chu:
- Day la ket qua benchmark chinh thuc cua MDNet duoc toolkit LaSOT danh gia lai.
- Full local `PyMDNet` run de lay `FPS_avg_seq` va `FPS_weighted_by_frames` dang chay tai `OtherTracker/MDNet/lasot_pymdnet_vototb_testset280_local`.
