<# RAM-09B offline retrospective (no Lean needed). First-time clean output only.
   Requires original RAM-08 Actions ZIP and Python with torch.
   No new PEF theorem access, no training modification to original RAM08.
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$Archive,
  [Parameter(Mandatory=$true)][string]$Out,
  [string]$Python='python'
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$repo=(Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$archiveFile=(Resolve-Path $Archive).Path
$output=[IO.Path]::GetFullPath($Out)
if (Test-Path $output) {throw "RAM-09B result directory already exists: $output"}
if (!(Test-Path $archiveFile)) {throw 'Missing original audited RAM-08 training archive ZIP'}
$env:CUDA_VISIBLE_DEVICES=''
$env:PYTHONHASHSEED='0'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$start=[DateTime]::UtcNow.ToString('o')
New-Item -ItemType Directory -Force -Path $output | Out-Null
$meta=[ordered]@{ experiment='RAM09B_retrospective_balanced';status='running';original_ram08_heldout_PEF_cases=0;started_utc=$start }
$meta | ConvertTo-Json | Set-Content (Join-Path $output 'manifest.json')
function Checked([string]$Label,[scriptblock]$Block) {
  Write-Host "[RAM09B] $Label"
  & $Block
  if ($LASTEXITCODE -ne 0) {throw "$Label exit code $LASTEXITCODE"}
}
$original=Get-Location
try {
  Set-Location $repo
  Checked 'Six RAM-09B unit tests' { & $Python -m unittest discover -s experiments/lean_ram09b -p 'test_*.py' -v }
  Checked 'Frozen full family-balanced 200-step three-model CPU fit' {
    & $Python experiments/lean_ram09b/ram09b_balanced_cpu.py $archiveFile --out (Join-Path $output 'balanced_fit')
  }
  Checked 'Post-score original frozen rank-margin feasibility audit' {
    & $Python experiments/lean_ram09b/ram09b_bounds.py $archiveFile --out (Join-Path $output 'rank_feasibility.json')
  }
  $fit=Get-Content (Join-Path $output 'balanced_fit/report.json') -Raw | ConvertFrom-Json
  $bounds=Get-Content (Join-Path $output 'rank_feasibility.json') -Raw | ConvertFrom-Json
  if ($fit.status -ne 'complete' -or $fit.prospective_pef_case_scores -ne 0 -or $bounds.original_PEF_theorems_read -ne 0) {
    throw 'Audit incomplete or PEF source access violated'
  }
  $meta.status='complete'
  $meta.registered_descriptive_target_met=[bool]$fit.descriptive_target_met
  $meta.finished_utc=[DateTime]::UtcNow.ToString('o')
  $meta | ConvertTo-Json | Set-Content (Join-Path $output 'manifest.json')
  Write-Host "[RAM09B] Complete; retrospective old-source evidence only. $output"
} catch {
  $meta.status='aborted';$meta.error=$_.Exception.Message
  $meta.failed_utc=[DateTime]::UtcNow.ToString('o')
  $meta | ConvertTo-Json | Set-Content (Join-Path $output 'manifest.json')
  throw
} finally { Set-Location $original }
