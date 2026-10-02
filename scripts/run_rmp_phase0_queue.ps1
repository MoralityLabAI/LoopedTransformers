param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$PoolDir,
    [Parameter(Mandatory = $true)][string]$ArtifactDir,
    [string[]]$Arms = @("LOOP-T", "FF-U"),
    [int[]]$Seeds = @(1, 2, 3),
    [string[]]$Families = @("pointer_chase", "gated_pointer_chase", "reachability", "sudoku4"),
    [string]$Variant = "core4",
    [int]$MaxSteps = 60000,
    [int]$BatchPerFamily = 256,
    [int]$Microbatch = 256,
    [int]$SharedFreeMb = 2000,
    [switch]$SecondStage,
    [int]$SecondStageMaxSteps = 40000,
    [string]$SnapshotDir = ""
)

# Runs the gate-G0 cells one after another: seed by seed, alternating arms, so that an
# interrupted queue leaves the arms equally covered. One cell at a time keeps the host's
# committed memory and the shared GPU within reach of the other work on this machine.
# Each cell stops at C3; C4 is produced later by resuming the cell without -StopAtGate.
#
# With -SecondStage, a cell that exhausts its budget without passing the gate is continued
# with deep supervision at every visit (rung 3 of the fallback ladder): its final weights are
# snapshotted with a SHA-256 receipt and warm-start a "<Variant>-ds" cell.
# Rerunning the queue resumes every cell where it stopped.

$ErrorActionPreference = "Continue"
$runner = Join-Path $PSScriptRoot "run_rmp_training_cell.ps1"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runs = Join-Path $repo "experiments/recurrence_monitorability_v0/runs"
if (-not $SnapshotDir) { $SnapshotDir = Join-Path (Split-Path $ArtifactDir -Parent) "snapshots" }

function Invoke-Cell([string]$Arm, [int]$Seed, [string]$CellVariant, [int]$Steps, [string[]]$Extra) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $runner `
        -Arm $Arm -Seed $Seed -Families ($Families -join ",") -MaxSteps $Steps `
        -PoolDir $PoolDir -ArtifactDir $ArtifactDir -Variant $CellVariant -Python $Python `
        -Device cuda -GpuPolicy shared -SharedFreeMb $SharedFreeMb -GpuWaitMinutes 1440 `
        -SliceSecondsOverride 1200 -BatchPerFamily $BatchPerFamily -Microbatch $Microbatch `
        -Beta 1.0 -Curriculum -Staircase -StopAtGate -MaxAttempts 2000 -MaxConsecutiveFailures 30 @Extra | Out-Host
    return $LASTEXITCODE
}

function Get-CellStatus([string]$CellId) {
    $path = Join-Path $runs "$CellId/result.json"
    if (-not (Test-Path $path)) { return $null }
    return (Get-Content -Raw $path | ConvertFrom-Json).status
}

foreach ($seed in $Seeds) {
    foreach ($arm in $Arms) {
        $cell = "$arm-$Variant-s$seed"
        Write-Output "$(Get-Date -Format o) queue: starting $cell"
        $exit = Invoke-Cell $arm $seed $Variant $MaxSteps @()
        $status = Get-CellStatus $cell
        Write-Output "$(Get-Date -Format o) queue: $cell runner exit $exit, status $status"
        if (-not $SecondStage -or $status -ne "budget_exhausted") { continue }

        $source = Join-Path $ArtifactDir "$cell/checkpoints/latest.pt"
        $snapshot = Join-Path $SnapshotDir "$cell-final.pt"
        if (-not (Test-Path $snapshot)) {
            New-Item -ItemType Directory -Force -Path $SnapshotDir | Out-Null
            Copy-Item $source $snapshot
            Copy-Item "$source.receipt.json" "$snapshot.receipt.json"
        }
        $receipt = Get-Content -Raw "$snapshot.receipt.json" | ConvertFrom-Json
        if ((Get-FileHash $snapshot -Algorithm SHA256).Hash.ToLower() -ne $receipt.sha256) {
            Write-Output "$(Get-Date -Format o) queue: $cell snapshot hash mismatch; second stage skipped"
            continue
        }
        $second = "$Variant-ds"
        Write-Output "$(Get-Date -Format o) queue: starting $arm-$second-s$seed from $snapshot"
        $exit = Invoke-Cell $arm $seed $second $SecondStageMaxSteps @("-DeepSupervision", "-InitFrom", $snapshot)
        Write-Output "$(Get-Date -Format o) queue: $arm-$second-s$seed runner exit $exit, status $(Get-CellStatus "$arm-$second-s$seed")"
    }
}
Write-Output "$(Get-Date -Format o) queue: finished"
