$py = "C:\Users\patri\.venvs\bluebeam\Scripts\python.exe"
$aux = "C:\Users\patri\Documents\Codex\rmp_v0_aux"
$wt = "C:\Users\patri\rmp-aux-20261125"
$pw = "C:\Users\patri\lt-main-push-20261126"
Set-Location $wt
$env:PYTHONPATH = $wt
$env:RMP_BF16 = "0"
$log = "$aux\ffu_modnc_s1.log"
"$(Get-Date -Format o) FF-U modnc s1 start" | Add-Content $log
# Same settings as LOOP-T-modnc-s1: modular_recurrence only, all depths from step 0 (no -Curriculum), plain stage, 30,000 steps.
& powershell -NoProfile -ExecutionPolicy Bypass -File "$wt\scripts\run_rmp_training_cell.ps1" `
    -Arm FF-U -Seed 1 -Families "modular_recurrence" -MaxSteps 30000 -PoolDir "$aux\pool" -ArtifactDir "$aux\artifacts" `
    -Variant modnc -Python $py -Device cuda -GpuPolicy shared -SharedFreeMb 2000 -GpuWaitMinutes 1440 -SliceSecondsOverride 1200 `
    -BatchPerFamily 256 -Microbatch 256 -Beta 1.0 -StopAtGate -MaxAttempts 2000 -MaxConsecutiveFailures 30 *>> $log
"$(Get-Date -Format o) FF-U modnc s1 finished" | Add-Content $log
"done" | Set-Content "$aux\ffu_modnc_done.marker"
