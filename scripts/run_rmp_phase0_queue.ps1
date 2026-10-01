param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$PoolDir,
    [Parameter(Mandatory = $true)][string]$ArtifactDir,
    [string[]]$Arms = @("LOOP-T", "FF-U"),
    [int[]]$Seeds = @(1, 2, 3),
    [string[]]$Families = @("pointer_chase", "gated_pointer_chase", "reachability", "sudoku4"),
    [string]$Variant = "core4",
    [int]$MaxSteps = 40000,
    [int]$BatchPerFamily = 256,
    [int]$Microbatch = 128,
    [int]$SharedFreeMb = 2000
)

# Runs the gate-G0 cells one after another: seed by seed, alternating arms, so that an
# interrupted queue leaves the arms equally covered. One cell at a time keeps the host's
# committed memory and the shared GPU within reach of the other work on this machine.
# Each cell stops at C3; C4 is produced later by resuming the cell without -StopAtGate.

$ErrorActionPreference = "Continue"
$runner = Join-Path $PSScriptRoot "run_rmp_training_cell.ps1"
foreach ($seed in $Seeds) {
    foreach ($arm in $Arms) {
        Write-Output "$(Get-Date -Format o) queue: starting $arm-$Variant-s$seed"
        & powershell -NoProfile -ExecutionPolicy Bypass -File $runner `
            -Arm $arm -Seed $seed -Families $Families -MaxSteps $MaxSteps `
            -PoolDir $PoolDir -ArtifactDir $ArtifactDir -Variant $Variant -Python $Python `
            -Device cuda -GpuPolicy shared -SharedFreeMb $SharedFreeMb -GpuWaitMinutes 1440 `
            -SliceSecondsOverride 1200 -BatchPerFamily $BatchPerFamily -Microbatch $Microbatch `
            -Beta 1.0 -Curriculum -Staircase -StopAtGate -MaxAttempts 2000 -MaxConsecutiveFailures 30
        Write-Output "$(Get-Date -Format o) queue: $arm-$Variant-s$seed runner exit $LASTEXITCODE"
    }
}
Write-Output "$(Get-Date -Format o) queue: finished"
