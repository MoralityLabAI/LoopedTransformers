param(
  [ValidateSet('smoke','5m','15m','30m')][string]$Preset = '5m',
  [int]$TotalSteps = 10000,
  [int]$TemperatureLimitC = 82,
  [string]$Output = 'runs/overnight',
  [int]$Batch = 1,
  [int]$Accumulation = 8
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
while ($true) {
  $temp = $null
  $nv = Get-Command 'nvidia-smi' -ErrorAction SilentlyContinue
  if ($nv) {
    $reading = & nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -eq 0 -and $reading) {
      $temp = [int](($reading | Select-Object -First 1).Trim())
    }
  }
  if ($null -ne $temp -and $temp -ge $TemperatureLimitC) {
    Write-Host "GPU temperature ${temp}C at limit; pausing 120s"
    Start-Sleep -Seconds 120
    continue
  }
  $args = @('gym.py','train','--preset',$Preset,'--out',$Output,
            '--steps',"$TotalSteps",'--batch',"$Batch",'--accumulate',"$Accumulation",
            '--eval-every','100','--save-every','25','--resume')
  & python @args
  if ($LASTEXITCODE -eq 0) { Write-Host 'Training completed'; break }
  Write-Warning "Training exited $LASTEXITCODE. Check logs/temperature before continued work."
  Start-Sleep -Seconds 300
}
