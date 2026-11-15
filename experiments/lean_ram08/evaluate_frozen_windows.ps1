<#
RAM-08 frozen PEF heldout evaluation ONLY.
Uses the SHA-pinned original CPU train artifact from GitHub run 43689874169.
Will not collect data, retrain, select seeds, change theorem names, or use
original reference proofs as policy input.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$TestMathCan,
    [Parameter(Mandatory=$true)][string]$Gym,
    [Parameter(Mandatory=$true)][string]$FrozenRoot,
    [Parameter(Mandatory=$true)][string]$Out,
    [string]$Python = "python",
    [Parameter(Mandatory=$true)][string]$LeanBin
)
$ErrorActionPreference="Stop"
Set-StrictMode -Version Latest

function Invoke-Checked([string]$Description,[scriptblock]$Code) {
    Write-Host "[RAM08 FROZEN TEST] $Description"
    & $Code
    if ($LASTEXITCODE -ne 0) {
        throw "RAM08 stopped at '$Description': command exited $LASTEXITCODE"
    }
}
$repo=(Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$mathcan=(Resolve-Path $TestMathCan).Path
$gym=(Resolve-Path $Gym).Path
$frozen=(Resolve-Path $FrozenRoot).Path
$leanHome=(Resolve-Path $LeanBin).Path
$root=[IO.Path]::GetFullPath($Out)
if (Test-Path $root) { throw "Output exists; first prospective comparison must use a new empty path: $root" }
if (!(Test-Path (Join-Path $frozen 'training/training_report.json'))) {
    throw "Unzip the entire original train artifact under FrozenRoot; missing training/training_report.json"
}
if (!(Test-Path (Join-Path $frozen 'data/transitions.jsonl'))) {
    throw "The original train-only compiler-observation data are missing"
}
$lean=Join-Path $leanHome 'lean.exe'
$lake=Join-Path $leanHome 'lake.exe'
if (!(Test-Path $lean) -or !(Test-Path $lake)) {
    throw "Lean 4.35 lean.exe/lake.exe required under $leanHome"
}
$trainedConfig=Join-Path $repo 'experiments/lean_ram08/config.json'
$testSha=(& git -C $mathcan rev-parse HEAD).Trim()
if ($testSha -ne '83f61997948818b68da94429c2fe736d93ea3093') {
    throw "PEF TEST checkout SHA mismatch; expected original PR #44 pinned 83f619979..."
}
$gymSha=(& git -C $gym rev-parse HEAD).Trim()
if ($gymSha -ne 'e8a780254b5cf41d479095f4f22bcfc4aa76ed72') {
    throw "Source-gym checkout SHA mismatch"
}
$toolchain=(Get-Content (Join-Path $mathcan 'lean-toolchain') -Raw).Trim()
if ($toolchain -ne 'leanprover/lean4:v4.35.0') {
    throw "Lean toolchain mismatch: $toolchain"
}
New-Item -ItemType Directory -Force -Path $root | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $root 'temp') | Out-Null
$env:PATH="$leanHome;$env:PATH"
$env:PYTHONPATH=$gym
$env:PYTHONHASHSEED="0"
$env:LEAN_NUM_THREADS="2"
$env:OMP_NUM_THREADS="2"
$env:MKL_NUM_THREADS="2"
$env:TEMP=Join-Path $root 'temp'
$env:TMP=$env:TEMP
$proof=Join-Path $root 'first_frozen_prospective'
$audit=Join-Path $root 'third_independent_replay'
$manifest=[ordered]@{
    experiment = 'RAM08_original_PEF_first_whole_suite'
    old_source_train = 'c84e2025334e9a967c1297425a490f98d5b3d189'
    new_source_test = $testSha
    source_gym_sha = $gymSha
    frozen_train_artifact = 'ram08-frozen-audited-train-43689874169'
    started_utc = [DateTime]::UtcNow.ToString('o')
    status = 'running'
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $root 'first_run_manifest.json')
$originalLocation=Get-Location
try {
    Set-Location $repo
    Invoke-Checked "Verify original 20,930 compiler effects and nine frozen byte-identical model checkpoints" {
        & $Python -c @'
import os,json,hashlib,sys
from pathlib import Path
root=Path(sys.argv[1]);t=json.loads((root/'training/training_report.json').read_text())
a=json.loads((root/'data/independent_audit.json').read_text())
expected={
'loop32':(29,'4a88bc80456743a89a70882911060cc90e76785a5ae1d435a0e46a8f8c09778f'),
'loop64':(43,'5cbd327904714ac698897e1c2333cae9ec4125aeb70e7c56cd6236db8a737dac'),
'flat64':(29,'2aa101edbd56f6563ebb2edcfc8c61c07dcc027559ea4cdc551187d6ec5303d8')}
assert t['status']=='complete' and len(t['models'])==9
assert a['status']=='passed' and a['checked']==20930
assert a['dataset_sha256']=='f92d5654fdb8fc1d4bd43e7fb4c0b1f03b7a27c1d7335dd86e6ec947b582542f'
assert hashlib.sha256((root/'data/transitions.jsonl').read_bytes()).hexdigest()==a['dataset_sha256']
assert t['natural_exact']['distinct_original_train_theorems']==3
assert t['natural_exact']['unique_positive_actions']==5
for name,(seed,sha) in expected.items():
    assert t['selected'][name]['seed']==seed
    assert t['selected'][name]['checkpoint_sha256']==sha
    f=root/'training'/f'{name}_s{seed}'/'best.pt'
    assert hashlib.sha256(f.read_bytes()).hexdigest()==sha
print('RAM08_FROZEN_TRAIN_VALIDATED',expected)
'@ $frozen
    }
    Invoke-Checked "Frozen source-only PEF theorem/grammar contract tests" {
        & $Python -m unittest discover -s experiments/lean_ram08 -p "test_*.py" -v
    }
    Set-Location $mathcan
    Invoke-Checked "Original PEF MathCan Lean4.35 dependencies" { & $lake build MathCan }
    $lakeEnv=(& $lake env printenv LEAN_PATH | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or !$lakeEnv) { throw 'Could not resolve original MathCan LEAN_PATH' }
    Set-Location $repo
    $env:LEAN_PATH="$(Join-Path $repo 'experiments/lean_ram03');$lakeEnv"
    Invoke-Checked "Compile unchanged RAM-03 persistent Lean server" {
        & $lean (Join-Path $repo 'experiments/lean_ram03/ProbeServer.lean') -o (Join-Path $repo 'experiments/lean_ram03/ProbeServer.olean')
    }
    Invoke-Checked "First complete twelve PEF theorem comparison with six primary and four secondary controls" {
        & $Python experiments/lean_ram08/evaluate.py --mathcan $mathcan --data (Join-Path $frozen 'data') --training (Join-Path $frozen 'training') --config $trainedConfig --lean $lean --out $proof
    }
    Invoke-Checked "Third independent original Lean4.35 complete theorem source recompile and axiom gate" {
        & $Python experiments/lean_ram08/audit_results.py --results $proof --lean $lean --out $audit
    }
    $scores=Get-Content (Join-Path $proof 'evaluation_report.json') -Raw | ConvertFrom-Json
    $trust=Get-Content (Join-Path $audit 'RAM08_independent_audit.json') -Raw | ConvertFrom-Json
    if ($scores.status -ne 'complete' -or $trust.scored_cases -ne 120 -or $trust.false_positive_certificates -ne 0) {
        throw "Incomplete original source-family proof audit"
    }
    $manifest.status='complete'
    $manifest.finished_utc=[DateTime]::UtcNow.ToString('o')
    $manifest.registered_model_only_success=$scores.registered_model_only_success
    $manifest.registered_recurrence_specific_success=$scores.registered_recurrence_specific_success
    $manifest.original_accepted_policy_theorem_instances=$trust.independently_recompiled_successes
    $manifest | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $root 'first_run_manifest.json')
    Write-Host "RAM08 first PEF result: $proof"
} catch {
    $manifest.status='aborted'
    $manifest.failed_utc=[DateTime]::UtcNow.ToString('o')
    $manifest.error=$_.Exception.Message
    $manifest | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $root 'first_run_manifest.json')
    throw
} finally { Set-Location $originalLocation }
