param([Parameter(Mandatory = $true)][int]$Seed)
$py = "C:\Users\patri\.venvs\bluebeam\Scripts\python.exe"
$aux = "C:\Users\patri\Documents\Codex\rmp_v0_aux"
$wt = "C:\Users\patri\rmp-aux-20261125"
Set-Location $wt
$env:PYTHONPATH = $wt
$env:RMP_BF16 = "0"   # GPU bench: bf16 0.93x of fp32 on this model
$log = "$aux\overnight2_s$Seed.log"
"$(Get-Date -Format o) seed $Seed start" | Add-Content $log

# One seed of the masked_pointer_chase gym: both arms, plain -> deep supervision -> deep supervision + re-injection.
# Invoked in-process so -Arms stays an array (powershell -File flattens it into one string).
& "$wt\scripts\run_rmp_phase0_queue.ps1" -Python $py -PoolDir "$aux\pool" -ArtifactDir "$aux\artifacts" `
    -Arms @("LOOP-T", "FF-U") -Seeds @($Seed) -Families @("masked_pointer_chase") -Variant mask2 `
    -SecondStage -ThirdStage *>> $log
"$(Get-Date -Format o) seed $Seed masked gym finished" | Add-Content $log

if ($Seed -eq 1) {
    # modular_recurrence without the depth curriculum (its depth-1 train pool has only 2,448 unique inputs).
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$wt\scripts\run_rmp_training_cell.ps1" `
        -Arm LOOP-T -Seed 1 -Families "modular_recurrence" -MaxSteps 30000 -PoolDir "$aux\pool" -ArtifactDir "$aux\artifacts" `
        -Variant modnc -Python $py -Device cuda -GpuPolicy shared -SharedFreeMb 2000 -GpuWaitMinutes 1440 -SliceSecondsOverride 1200 `
        -BatchPerFamily 256 -Microbatch 256 -Beta 1.0 -StopAtGate -MaxAttempts 2000 -MaxConsecutiveFailures 30 *>> $log
    "$(Get-Date -Format o) modular no-curriculum cell finished" | Add-Content $log
}
