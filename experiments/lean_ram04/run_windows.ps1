<#
RAM-04 local Windows RTX 3050 / CPU training and exact Lean evaluation.
Runs a NEW experiment directory with the source and train selectors frozen.
Requires pinned Lean4.35, original MathCan and Compactification-Theory.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$MathCan,
    [Parameter(Mandatory=$true)][string]$Gym,
    [Parameter(Mandatory=$true)][string]$OriginalFC12,
    [Parameter(Mandatory=$true)][string]$Out,
    [string]$Python = "python",
    [Parameter(Mandatory=$true)][string]$LeanBin
)
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
function Invoke-Checked([string]$Step, [scriptblock]$Action) {
    Write-Host "[RAM04] $Step"
    & $Action
    if ($LASTEXITCODE -ne 0) { throw "RAM04 failed at $Step (exit $LASTEXITCODE)" }
}
$repo = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$mathcanPath = (Resolve-Path $MathCan).Path
$gymPath = (Resolve-Path $Gym).Path
$inputRoot = (Resolve-Path $OriginalFC12).Path
$leanHome = (Resolve-Path $LeanBin).Path
$runRoot = [IO.Path]::GetFullPath($Out)
if (Test-Path $runRoot) { throw "RAM04 requires a fresh output directory: $runRoot" }
if (!(Test-Path (Join-Path $inputRoot "train_source/actual_goal_states.jsonl"))) {
    throw "The original FC12 train_source/actual_goal_states.jsonl is missing"
}
if (!(Test-Path (Join-Path $inputRoot "train_probes/train_candidate_outcomes.jsonl"))) {
    throw "The original FC12 actual compiler probe JSONL is missing"
}
if (!(Test-Path (Join-Path $gymPath "research/refutation_bench/fc12/core.py"))) {
    throw "This is not the pinned Compactification-Theory gym checkout"
}
$lean = Join-Path $leanHome "lean.exe"
$lake = Join-Path $leanHome "lake.exe"
if (!(Test-Path $lean) -or !(Test-Path $lake)) { throw "Lean4.35 and lake binaries required in $leanHome" }
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $runRoot "temp") -Force | Out-Null
$env:Path = "$leanHome;$env:Path"
$env:PYTHONPATH = $gymPath
$env:LEAN_PATH = "$(Join-Path $repo 'experiments/lean_ram03');$(Join-Path $mathcanPath '.lake/build/lib/lean')"
$env:LEAN_NUM_THREADS = "2"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$env:PYTHONHASHSEED = "0"
$env:TEMP = Join-Path $runRoot "temp"
$env:TMP = $env:TEMP
$config = Join-Path $repo "experiments/lean_ram04/config.json"
$data = Join-Path $runRoot "data"
$weights = Join-Path $runRoot "training"
$proofs = Join-Path $runRoot "prospective"
$audit = Join-Path $runRoot "independent-audit"
$notes = [ordered]@{
    experiment = "RAM04"
    runner = "Windows PowerShell"
    source_mathcan = $mathcanPath
    source_gym = $gymPath
    original_training = $inputRoot
    config = $config
    output_root = $runRoot
    started_utc = [DateTime]::UtcNow.ToString("o")
}
$notes | ConvertTo-Json | Set-Content -Path (Join-Path $runRoot "local_manifest.json")
$oldLocation = Get-Location
try {
    Set-Location $repo
    $gitCommit = (& git -C $mathcanPath rev-parse HEAD).Trim()
    if ($gitCommit -ne "c84e2025334e9a967c1297425a490f98d5b3d189") {
        throw "MathCan source commit has changed: $gitCommit"
    }
    $toolchain = (Get-Content (Join-Path $mathcanPath "lean-toolchain") -Raw).Trim()
    if ($toolchain -ne "leanprover/lean4:v4.35.0") { throw "Lean toolchain mismatch: $toolchain" }

    Invoke-Checked "RAM04 frozen protocol unit tests" {
        & $Python -m unittest discover -s experiments/lean_ram04 -p "test_*.py" -v
    }
    Set-Location $mathcanPath
    Invoke-Checked "Original MathCan Lake build" { & $lake build MathCan }
    Set-Location $repo
    Invoke-Checked "Compile immutable RAM03 persistent Lean server" {
        & $lean (Join-Path $repo "experiments/lean_ram03/ProbeServer.lean") -o (Join-Path $repo "experiments/lean_ram03/ProbeServer.olean")
    }
    Invoke-Checked "Collect 27 original train roots and independently replay compiler actions" {
        & $Python experiments/lean_ram03/collect.py --data $inputRoot --mathcan $mathcanPath --lean $lean --config $config --out $data
    }
    Invoke-Checked "CPU/CUDA throughput calibration and nine tiny model fits" {
        & $Python experiments/lean_ram03/train_local.py --data $data --config $config --out $weights
    }
    Invoke-Checked "First prospective twelve original MathCan theorems, native and small models" {
        & $Python experiments/lean_ram04/evaluate.py --mathcan $mathcanPath --data $data --training $weights --lean $lean --config $config --out $proofs
    }
    Invoke-Checked "Independent exact-source Lean compiler replay of all accepted theorems" {
        & $Python experiments/lean_ram04/audit_results.py --results $proofs --lean $lean --out $audit
    }
    $summary = Get-Content (Join-Path $proofs "RAM04_report.json") -Raw | ConvertFrom-Json
    if ($summary.status -ne "complete") { throw "RAM04 scoring incomplete" }
    $notes.finished_utc = [DateTime]::UtcNow.ToString("o")
    $notes.result = "complete"
    $notes.primary_model_only = $summary.primary_model_only
    $notes.recurrence_specific = $summary.recurrence_specific
    $notes | ConvertTo-Json | Set-Content -Path (Join-Path $runRoot "local_manifest.json")
    Write-Host "[RAM04] complete: $proofs"
} catch {
    $notes.failed_utc = [DateTime]::UtcNow.ToString("o")
    $notes.result = "incomplete"
    $notes.error = $_.Exception.Message
    $notes | ConvertTo-Json | Set-Content -Path (Join-Path $runRoot "local_manifest.json")
    throw
} finally {
    Set-Location $oldLocation
}
