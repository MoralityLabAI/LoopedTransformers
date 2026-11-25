$ErrorActionPreference = "Stop"
$root = "C:\Users\patri\lbf20m-run-20261125"
$src  = "$root\src"
$py   = "C:\Users\patri\AppData\Local\Programs\Python\Python312\python.exe"
Remove-Item Env:PYTHONOPTIMIZE -ErrorAction SilentlyContinue
$env:PYTHONHASHSEED = "0"
$env:PYTHONUTF8 = "1"
Set-Location $src

# environment record
$gitexe = (Get-ChildItem "$env:LOCALAPPDATA\GitHubDesktop\app-*\resources\app\git\cmd\git.exe" | Select-Object -Last 1).FullName
$cpu = Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed
$os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,TotalVisibleMemorySize
$leanv = & "$env:USERPROFILE\.elan\toolchains\leanprover--lean4---v4.19.0\bin\lean.exe" --version
$chk = & $py -c "import sys,json;print(json.dumps({'version':sys.version,'optimize':sys.flags.optimize,'hashseed_env':__import__('os').environ.get('PYTHONHASHSEED'),'executable':sys.executable}))"
$meta = [ordered]@{
  started_utc = (Get-Date).ToUniversalTime().ToString("o")
  source_head = (& $gitexe -C $src rev-parse HEAD)
  worktree_status_lines = @(& $gitexe -C $src status --short).Count
  os = $os; cpu = $cpu; lean = $leanv
  python = ($chk | ConvertFrom-Json)
  deviation = "Python 3.12.x used instead of the registered Python 3.11 (user decision; 3.11 not installed)"
  command = "python -u research/lite_braille_fresco/lbf20m_factor.py --out <run-root>/repeat-NN/lbf20m"
}
$meta | ConvertTo-Json -Depth 5 | Set-Content "$root\environment.json" -Encoding utf8

for ($i = 1; $i -le 5; $i++) {
  $n = "{0:D2}" -f $i
  $dir = "$root\repeat-$n"
  New-Item -ItemType Directory -Force $dir | Out-Null
  $out = "$dir\lbf20m"
  if (Test-Path $out) { throw "output exists: $out" }
  $t0 = Get-Date
  $p = Start-Process -FilePath $py -ArgumentList @("-u","research/lite_braille_fresco/lbf20m_factor.py","--out",$out) `
        -WorkingDirectory $src -NoNewWindow -PassThru -RedirectStandardOutput "$dir\console.log" -RedirectStandardError "$dir\console.err"
  $peak = 0
  while (-not $p.HasExited) {
    try { $p.Refresh(); if ($p.PeakWorkingSet64 -gt $peak) { $peak = $p.PeakWorkingSet64 } } catch {}
    Start-Sleep -Milliseconds 500
  }
  $p.WaitForExit()
  $wall = ((Get-Date) - $t0).TotalSeconds
  [ordered]@{ repeat = $n; exit_code = $p.ExitCode; wall_seconds = $wall; peak_working_set_bytes_sampled = $peak
              note = "peak RSS is whole process (audit + planner + oracle + arms), sampled every 0.5 s"
              start_utc = $t0.ToUniversalTime().ToString("o") } | ConvertTo-Json | Set-Content "$dir\run.json" -Encoding utf8
}
"ALL DONE" | Set-Content "$root\launcher.done"
