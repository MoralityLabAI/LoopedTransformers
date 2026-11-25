$env:RMP_BF16 = "1"
Set-Location "C:\Users\patri\rmp-aux-20261125"
$aux = "C:\Users\patri\Documents\Codex\rmp_v0_aux"
& powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_rmp_training_cell.ps1 `
  -Arm LOOP-T -Seed 1 -Families masked_pointer_chase,modular_recurrence -MaxSteps 60000 `
  -PoolDir "$aux\pool" -ArtifactDir "$aux\artifacts_cpu" -Variant aux2cpu `
  -Python "C:\Users\patri\.venvs\bluebeam\Scripts\python.exe" -Device cpu `
  -BatchPerFamily 256 -Microbatch 256 -Beta 1.0 -Curriculum -StopAtGate -EvalEvery 250 `
  -MaxAttempts 5000 -MaxConsecutiveFailures 30
