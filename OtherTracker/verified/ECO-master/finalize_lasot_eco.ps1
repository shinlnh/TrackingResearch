$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $repoRoot

$pythonExe = Join-Path $repoRoot "venv312\Scripts\python.exe"
$postprocessScript = Join-Path $repoRoot "OtherTracker\ECO\postprocess_lasot_eco.py"
$resultsDir = Join-Path $repoRoot "OtherTracker\pytracking\pytracking\tracking_results\eco\default"
$toolkitDir = Join-Path $repoRoot "ls\LaSOT_Evaluation_Toolkit"
$outDir = Join-Path $repoRoot "OtherTracker\lasot\lasot936\ECO"
$benchmarkDir = Join-Path $outDir "benchmark"

New-Item -ItemType Directory -Force $outDir, $benchmarkDir | Out-Null

& $pythonExe $postprocessScript `
    --results-dir $resultsDir `
    --toolkit-dir $toolkitDir `
    --out-dir $outDir `
    --display-name ECO

Push-Location $toolkitDir
try {
    matlab.exe -batch "run_eco_testset_evaluation"
    matlab.exe -batch "run_mytracker_benchmark_comparison"
}
finally {
    Pop-Location
}

Copy-Item (Join-Path $toolkitDir "res_fig\eco_lasot_testset_summary.csv") $outDir -Force
Copy-Item (Join-Path $toolkitDir "res_fig\eco_lasot_testset_summary.txt") $outDir -Force
Copy-Item (Join-Path $toolkitDir "res_fig\eco_testset_overlap_auc.png") $outDir -Force
Copy-Item (Join-Path $toolkitDir "res_fig\eco_testset_error_threshold.png") $outDir -Force

Copy-Item (Join-Path $toolkitDir "res_fig\mytracker_benchmark_scores.csv") (Join-Path $benchmarkDir "benchmark_scores.csv") -Force
Copy-Item (Join-Path $toolkitDir "res_fig\mytracker_benchmark_success_auc.png") (Join-Path $benchmarkDir "success_plot_benchmark_auc.png") -Force
Copy-Item (Join-Path $toolkitDir "res_fig\mytracker_benchmark_precision20.png") (Join-Path $benchmarkDir "precision_plot_benchmark20.png") -Force

Write-Output "Finalized ECO LaSOT outputs under $outDir"
