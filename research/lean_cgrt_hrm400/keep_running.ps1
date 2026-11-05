<#
Unattended worker for the Lean-CGRT gym (Windows PowerShell 5.1+).

  .\keep_running.ps1                         # probe hardware, pick a queue, run it to the end
  .\keep_running.ps1 -ProbeOnly              # hardware receipt only (no GPU work)
  .\keep_running.ps1 -Diagnostics            # hardware receipt + measured throughput/VRAM per preset
  .\keep_running.ps1 -Queue configs\campaign_c1.jsonl -Campaign c1

What it does
  * probes CPU, RAM, GPU (nvidia-smi), VRAM, disk, AC power; writes hardware.json;
  * picks a queue by VRAM unless -Queue is given (>=12 GB: campaign_c1, else campaign_small_vram);
  * refuses to start below -MinFreeDiskGB, and waits (logging why) while on battery;
  * runs `python -m lcgrt.runq`, which admits one GPU job at a time through vramorama when it is
    installed. Every training job resumes from its newest hash-verified checkpoint (latest, else
    last-known-good) and exits 75 to pause on GPU heat, a GPU power cap, low disk, low RAM or
    battery; vramorama restarts it when the card is clear. Evaluation jobs in the queue score
    finished checkpoints on the frozen held-out splits;
  * logs telemetry every -TelemetrySeconds (GPU temp/power/util/VRAM, CPU, RAM, disk, current job,
    last training line) to telemetry_<campaign>.jsonl, rotating logs above -RotateMB (keeps 5);
  * stops for good when runq reports repeated failures (exit 3), writing stopped.json, instead of
    restarting a broken experiment; queue completion writes done.json;
  * a lock file keeps a second worker from starting. After sleep or a reboot, run it again (or let
    the scheduled task from install_task.ps1 do it): finished jobs are skipped, the rest resume.
CPU temperature is not readable without elevation on this laptop; it is logged as null.
Python is started through `cmd /c ... > log`: Start-Process with -RedirectStandardOutput has
killed Python children silently on this machine (exception 0xc000070a).
#>
param(
  [string]$Campaign = '',
  [string]$Queue = '',
  [string]$Python = '',
  [string]$WorkRoot = '',
  [int]$MinFreeDiskGB = 10,
  [int]$TelemetrySeconds = 60,
  [int]$RotateMB = 20,
  [switch]$Diagnostics,
  [switch]$ProbeOnly
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

if (-not $WorkRoot) {
  $WorkRoot = if ($env:LCGRT_WORK_ROOT) { $env:LCGRT_WORK_ROOT } elseif (Test-Path 'D:\') { 'D:\lean_cgrt' } else { "$env:USERPROFILE\lean_cgrt" }
}
$env:LCGRT_WORK_ROOT = $WorkRoot
$LogDir = Join-Path $WorkRoot 'logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$env:PYTHONIOENCODING = 'utf-8'; $env:PYTHONUNBUFFERED = '1'

if (-not $Python) {
  $cands = @($env:LCGRT_PYTHON, "$env:USERPROFILE\.venvs\lcgrt\Scripts\python.exe", "$env:USERPROFILE\.venvs\bluebeam\Scripts\python.exe")
  $Python = ($cands | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1)
  if (-not $Python) { $c = Get-Command python -ErrorAction SilentlyContinue; if ($c) { $Python = $c.Source } }
}
if (-not $Python) { throw 'No Python found; pass -Python or set LCGRT_PYTHON.' }

function Write-Line($file, $obj) {
  $obj['unix'] = [int][double]::Parse((Get-Date -UFormat %s))
  ($obj | ConvertTo-Json -Compress -Depth 6) | Out-File -FilePath $file -Append -Encoding utf8
}

function Rotate($file) {
  if ((Test-Path $file) -and ((Get-Item $file).Length -gt $RotateMB * 1MB)) {
    for ($i = 4; $i -ge 1; $i--) { if (Test-Path "$file.$i") { Move-Item -Force "$file.$i" "$file.$($i + 1)" } }
    Move-Item -Force $file "$file.1"
  }
}

function Get-Gpu {
  $nv = Get-Command nvidia-smi -ErrorAction SilentlyContinue
  if (-not $nv) { return $null }
  $q = 'name,memory.total,memory.used,temperature.gpu,power.draw,enforced.power.limit,utilization.gpu'
  $row = @(& nvidia-smi "--query-gpu=$q" --format=csv,noheader,nounits 2>$null)
  if ($LASTEXITCODE -ne 0 -or $row.Count -eq 0) { return $null }
  $v = ($row[0] -split ',') | ForEach-Object { $_.Trim() }
  return @{ name = $v[0]; vram_mib = [int]$v[1]; used_mib = [int]$v[2]; temp_c = [int]$v[3];
            power_w = $v[4]; power_limit_w = $v[5]; util_pct = $v[6] }
}

function Get-Hardware {
  $os = Get-CimInstance Win32_OperatingSystem
  $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
  $drive = Split-Path -Qualifier $WorkRoot
  $disk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$drive'"
  $bat = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue | Select-Object -First 1
  $onAc = $true
  if ($bat) { $onAc = ($bat.BatteryStatus -eq 2) -or ($bat.BatteryStatus -ge 6) }
  return @{
    machine = (Get-CimInstance Win32_ComputerSystem).Model
    cpu = $cpu.Name; cores = $cpu.NumberOfCores; threads = $cpu.NumberOfLogicalProcessors
    cpu_pct = $cpu.LoadPercentage
    ram_gb = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
    ram_avail_gb = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
    commit_free_gb = [math]::Round($os.FreeVirtualMemory / 1MB, 1)
    disk_free_gb = [math]::Round($disk.FreeSpace / 1GB, 1)
    on_ac = $onAc
    battery_pct = if ($bat) { $bat.EstimatedChargeRemaining } else { $null }
    cpu_temp_c = $null
    gpu = Get-Gpu
  }
}

$lock = Join-Path $LogDir 'worker.lock'
if (Test-Path $lock) {
  $old = Get-Content $lock -ErrorAction SilentlyContinue
  if ($old -and (Get-Process -Id ([int]$old) -ErrorAction SilentlyContinue)) { Write-Host "Worker already running (pid $old)."; exit 0 }
}
Set-Content -Path $lock -Value $PID

try {
  $hw = Get-Hardware
  ($hw | ConvertTo-Json -Depth 6) | Out-File -FilePath (Join-Path $LogDir 'hardware.json') -Encoding utf8
  Write-Host ("Hardware: {0}; {1} GB RAM; GPU {2} ({3} MiB); disk free {4} GB" -f $hw.machine, $hw.ram_gb,
              $hw.gpu.name, $hw.gpu.vram_mib, $hw.disk_free_gb)

  if ($ProbeOnly) { exit 0 }
  if ($Diagnostics) {
    & $Python -m lcgrt.diag --out (Join-Path $LogDir 'hw_diag.json')
    exit $LASTEXITCODE
  }
  if (-not $Queue) {
    $vram = 0
    if ($hw.gpu) { $vram = $hw.gpu.vram_mib }
    $Queue = if ($vram -ge 12000) { 'configs\campaign_c1.jsonl' } else { 'configs\campaign_small_vram.jsonl' }
  }
  if (-not $Campaign) { $Campaign = [IO.Path]::GetFileNameWithoutExtension($Queue) -replace '^campaign_', '' }
  $root = Join-Path (Join-Path $WorkRoot 'runs') $Campaign
  New-Item -ItemType Directory -Force -Path $root | Out-Null
  $tele = Join-Path $LogDir "telemetry_$Campaign.jsonl"
  $wlog = Join-Path $LogDir "worker_$Campaign.jsonl"

  while ($hw.disk_free_gb -lt $MinFreeDiskGB -or -not $hw.on_ac) {
    Write-Line $wlog @{ event = 'waiting'; disk_free_gb = $hw.disk_free_gb; on_ac = $hw.on_ac }
    Start-Sleep -Seconds 300
    $hw = Get-Hardware
  }

  Write-Line $wlog @{ event = 'start'; queue = $Queue; campaign = $Campaign; python = $Python }
  $out = Join-Path $root 'runq.out'
  Rotate $out
  $cmdline = "/c `"`"$Python`" -u -m lcgrt.runq --campaign $Campaign --queue `"$Queue`" >> `"$out`" 2>&1`""
  $p = Start-Process -FilePath 'cmd.exe' -ArgumentList $cmdline -WorkingDirectory $PSScriptRoot -PassThru -WindowStyle Hidden
  while (-not $p.HasExited) {
    Start-Sleep -Seconds $TelemetrySeconds
    Rotate $tele
    $job = $null; $last = $null
    $hb = Join-Path $root 'heartbeat.json'
    if (Test-Path $hb) { $job = (Get-Content $hb -Raw | ConvertFrom-Json).job }
    if ($job) {
      $tl = Join-Path (Join-Path $root $job) 'train.log'
      if (Test-Path $tl) { $last = Get-Content $tl -Tail 1 }
    }
    $h = Get-Hardware
    Write-Line $tele @{ job = $job; gpu = $h.gpu; ram_avail_gb = $h.ram_avail_gb; commit_free_gb = $h.commit_free_gb;
                        disk_free_gb = $h.disk_free_gb; on_ac = $h.on_ac; cpu_pct = $h.cpu_pct; cpu_temp_c = $null;
                        last_line = $last }
  }
  $p.WaitForExit()
  $rc = $p.ExitCode
  if ($rc -eq 0) {
    Write-Line $wlog @{ event = 'done'; returncode = $rc }
    @{ campaign = $Campaign; finished = (Get-Date).ToString('s') } | ConvertTo-Json | Out-File (Join-Path $root 'done.json') -Encoding utf8
  } else {
    Write-Line $wlog @{ event = 'stopped'; returncode = $rc }
    @{ campaign = $Campaign; returncode = $rc; reason = 'runq stopped (3 = repeated job failures); see queue.log';
       stopped = (Get-Date).ToString('s') } | ConvertTo-Json | Out-File (Join-Path $root 'stopped.json') -Encoding utf8
  }
  exit $rc
} finally {
  Remove-Item -Force $lock -ErrorAction SilentlyContinue
}
