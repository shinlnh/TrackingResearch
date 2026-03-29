# ECO LaSOT

Trang thai hien tai:
- Benchmark `ECO/default_deep` tai bang utility co san trong `OtherTracker/pytracking` khong phai LaSOT result, nen khong the dung truc tiep cho `LaSOT_Evaluation_Toolkit`.
- Da chuyen sang chay local `OtherTracker/pytracking` tracker `eco/default` tren `LaSOT test_set` voi GPU.

Smoke test:
- Sequence `guitar-16`
- `1000` frame
- `fps ~ 49.8332`

Tracking command:
- `venv312\Scripts\python.exe -u OtherTracker\pytracking\pytracking\run_experiment.py myexperiments eco_default_lasot --debug 0 --threads 0`

Log files:
- `tracking.stdout.log`
- `tracking.stderr.log`

Sau khi run xong se sync bbox vao `LaSOT_Evaluation_Toolkit`, tinh `success_auc/success50/precision20`, va copy plot + summary vao folder nay.
