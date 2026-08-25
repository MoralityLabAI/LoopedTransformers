param(
  [ValidateSet("smoke", "full")]
  [string]$Phase = "smoke",
  [Parameter(Mandatory = $true)]
  [ValidateSet("shared_gcpt", "low_high_gcpt", "no_role_key_control", "opaque_instruction_loop")]
  [string]$Candidate,
  [ValidateSet(17, 29, 43)]
  [int]$Seed = 17,
  [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "run_cdbt_factored_v3_1.ps1"
& $runner `
  -Study gcpt_register `
  -Phase $Phase `
  -Candidate $Candidate `
  -Seed $Seed `
  -Config "configs\gcpt_register_v4.json" `
  -Output $Output
exit $LASTEXITCODE
