param(
  [ValidateSet("smoke", "full")]
  [string]$Phase = "smoke",
  [Parameter(Mandatory = $true)]
  [ValidateSet("primitive_basis_gct", "primitive_basis_masked_gct")]
  [string]$Candidate,
  [ValidateSet(17, 29, 43)]
  [int]$Seed = 17,
  [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "run_cdbt_factored_v3_1.ps1"
& $runner `
  -Study intellect3_gct_v2 `
  -Phase $Phase `
  -Candidate $Candidate `
  -Seed $Seed `
  -Config "configs\intellect3_gct_mechanism_proxy_v2.json" `
  -Output $Output
exit $LASTEXITCODE
