param(
  [ValidateSet('smoke','5m','15m','30m')][string]$Preset = '5m',
  [int]$TotalSteps = 10000,
  [int]$TemperatureLimitC = 82,
  [string]$Output = 'runs/overnight',
  [int]$Batch = 1,
  [int]$Accumulation = 8,
  [ValidateSet('cuda','cpu')][string]$Device = 'cuda',
  [int]$MinimumFreeVramMB = 2300,
  [int]$Threads = 1,
  [int]$EvalCount = 8,
  [int]$MaxRules = 48,
  [int]$MaxTrainDepth = 5
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
# Keep host thread count low. VRAM allocator setting may reduce fragmentation.
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
$env:OMP_NUM_THREADS = "$Threads"
$env:MKL_NUM_THREADS = "$Threads"

if ($Batch -lt 1 -or $Accumulation -lt 1 -or $Threads -lt 1) {
  throw 'Batch, accumulation, and threads must be positive.'
}
if ($Device -eq 'cuda') {
  $nv = Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue
  if (-not $nv) { throw 'CUDA requested but nvidia-smi is unavailable.' }
  & python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 2)"
  if ($LASTEXITCODE -ne 0) { throw 'CUDA requested but PyTorch cannot access a CUDA GPU. Install a CUDA-enabled PyTorch wheel.' }
}
$failures = 0
while ($true) {
  if ($Device -eq 'cuda') {
    $rows = @(& nvidia-smi --query-gpu=temperature.gpu,memory.free --format=csv,noheader,nounits 2>$null)
    if ($LASTEXITCODE -ne 0 -or $rows.Count -eq 0) { throw 'Unable to check GPU health.' }
    # First GPU. Use CUDA_VISIBLE_DEVICES before launching to change selection.
    $values = ($rows[0] -split ',') | ForEach-Object { $_.Trim() }
    $temp = [int]$values[0]
    $freeMB = [int]$values[1]
    if ($temp -ge $TemperatureLimitC -or $freeMB -lt $MinimumFreeVramMB) {
      Write-Host ('Pausing: GPU ' + $temp + 'C, free VRAM ' + $freeMB + 'MiB')
      Start-Sleep -Seconds 120
      continue
    }
  }
  $trainArgs = @('gym.py','train','--preset',$Preset,'--device',$Device,'--out',$Output,
                '--steps',"$TotalSteps",'--batch',"$Batch",'--accumulate',"$Accumulation",
                '--threads',"$Threads",'--eval-count',"$EvalCount",
                '--max-rules',"$MaxRules",'--max-train-depth',"$MaxTrainDepth",
                '--eval-every','100','--save-every','25','--resume')
  & python @trainArgs
  if ($LASTEXITCODE -eq 0) { Write-Host 'Training completed'; break }
  $failures += 1
  if ($failures -ge 3) { throw ('Training failed ' + $failures + ' times; inspect log and VRAM pressure.') }
  Write-Warning ('Training exited ' + $LASTEXITCODE + '. Inspect logs and GPU memory.')
  Start-Sleep -Seconds 300
}
