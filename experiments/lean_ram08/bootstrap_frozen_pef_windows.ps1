<# RAM-08 operational recovery. Never mines, retrains, changes the PEF selector, or merges.
   Requires gh (authenticated for private LoopedTransformers artifacts),
   git, Python 3.11 with torch 2.6.0/psutil 7.1.3, and Lean4.35 bin.
   The original evaluate_frozen_windows.ps1 performs source/checkpoint tests.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Workspace,
    [Parameter(Mandatory=$true)][string]$LeanBin,
    [string]$Python = 'python'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-Verified([string]$Label, [scriptblock]$Command) {
    Write-Host "[RAM-08 PEF] $Label"
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE" }
}

$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$root = [IO.Path]::GetFullPath($Workspace)
$originalScript = Join-Path $PSScriptRoot 'evaluate_frozen_windows.ps1'
if (!(Test-Path $originalScript)) { throw "Frozen scorer missing: $originalScript" }
if (!(Test-Path $root)) { New-Item -ItemType Directory -Force -Path $root | Out-Null }
$test = Join-Path $root 'MathCan-PEF-test'
$gym = Join-Path $root 'Compactification-Theory-FC12-gym'
$frozen = Join-Path $root 'RAM08-immutable-train'
$out = Join-Path $root 'RAM08-first-original-PEF-evaluation'

# Preserve a single prospective full-suite attempt and its original name.
if (Test-Path $out) { throw "Prospective output already exists: $out. Preserve it; select a fresh Workspace for any explicitly labeled additional attempt." }
Invoke-Verified 'Read authenticated GitHub CLI status' { gh auth status }
$expectedMath = '83f61997948818b68da94429c2fe736d93ea3093'
$expectedGym = 'e8a780254b5cf41d479095f4f22bcfc4aa76ed72'

function Ensure-PinnedCheckout([string]$Destination, [string]$Source, [string]$Expected) {
    if (!(Test-Path $Destination)) {
        Invoke-Verified "Clone $Source" { gh repo clone $Source $Destination -- --no-checkout }
        Invoke-Verified "Fetch exact $Expected" { git -C $Destination fetch origin $Expected }
        Invoke-Verified "Check out exact $Expected" { git -C $Destination checkout --detach $Expected }
    }
    if (!(Test-Path (Join-Path $Destination '.git'))) { throw "Checkout is not a Git repository: $Destination" }
    $sha = (& git -C $Destination rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $sha -ne $Expected) {
        throw "Pinned source mismatch for $Source. Expected $Expected; found $sha. Existing checkout left untouched."
    }
}
Ensure-PinnedCheckout $test 'patrickdugan/MathCan' $expectedMath
Ensure-PinnedCheckout $gym 'MoralityLabAI/Compactification-Theory' $expectedGym

if (!(Test-Path (Join-Path $frozen 'training/training_report.json'))) {
    if (Test-Path $frozen) {
        throw "Existing frozen artifact directory is incomplete: $frozen. Preserve it and choose a new Workspace."
    }
    New-Item -ItemType Directory -Force -Path $frozen | Out-Null
    Invoke-Verified 'Download original frozen train artifact only, without retraining' {
        gh run download 43689874169 -R MoralityLabAI/LoopedTransformers --name ram08-frozen-audited-train-43689874169 --dir $frozen
    }
}

# The original evaluator verifies full MathCan/Gym commits, Lean version,
# 20,930 independent old-source effects, dataset SHA and each checkpoint SHA.
# It runs 12*6 primary and 12*4 secondary searches and external Lean replay.
Invoke-Verified 'Execute unchanged original PEF suite against frozen checkpoints' {
    & $originalScript -TestMathCan $test -Gym $gym -FrozenRoot $frozen -Out $out -Python $Python -LeanBin $LeanBin
}
Write-Host "[RAM-08 PEF] Original fully audited results under $out"
