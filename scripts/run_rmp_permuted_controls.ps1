param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$PoolDir,
    [Parameter(Mandatory = $true)][string]$ArtifactDir,
    [int]$MaxSteps = 20000,
    [string[]]$Arms = @("LOOP-T", "FF-U")
)

# Phase 1 permuted-target control models (docs/rmp_phase1_plan.md, E2): one per arm, seed 1, the stage-1
# recipe without the depth curriculum or the zero-hop pool, trained on each family's shuffled targets.
# Each runner waits for mains power and free VRAM and resumes its cell from the last evaluation point.

$ErrorActionPreference = "Stop"
$Runner = Join-Path $PSScriptRoot "run_rmp_training_cell.ps1"
foreach ($arm in $Arms) {
    Write-Output "$(Get-Date -Format o) controls: starting $arm-core4-perm-s1"
    & $Runner -Arm $arm -Seed 1 -Families "pointer_chase,gated_pointer_chase,reachability,sudoku4" `
        -MaxSteps $MaxSteps -PoolDir $PoolDir -ArtifactDir $ArtifactDir -Variant "core4-perm" -Python $Python `
        -Device cuda -GpuPolicy shared -SharedFreeMb 2000 -GpuWaitMinutes 1440 -SliceSecondsOverride 1200 `
        -BatchPerFamily 256 -Microbatch 256 -Beta 1.0 -PermuteTargets -MaxAttempts 2000 `
        -MaxConsecutiveFailures 30 | Out-Host
    Write-Output "$(Get-Date -Format o) controls: $arm-core4-perm-s1 runner exit $LASTEXITCODE"
}
Write-Output "$(Get-Date -Format o) controls: finished"
