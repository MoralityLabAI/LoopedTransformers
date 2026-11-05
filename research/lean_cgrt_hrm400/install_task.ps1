<#
Register (or remove) a Task Scheduler task that starts keep_running.ps1 at your logon.
Nothing is registered unless you run this script yourself.

  .\install_task.ps1                  # register "LeanCGRT-Worker" for the current user
  .\install_task.ps1 -Uninstall       # remove it

The task runs only on AC power, restarts the worker up to 3 times (10 minutes apart) if it
crashes, has no time limit, and never wakes the machine. The worker itself skips finished jobs,
resumes the rest from verified checkpoints, and stops for good after repeated job failures.
#>
param([switch]$Uninstall, [string]$Queue = '', [string]$TaskName = 'LeanCGRT-Worker')
$ErrorActionPreference = 'Stop'
if ($Uninstall) {
  Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
  Write-Host "Removed $TaskName"; exit 0
}
$script = Join-Path $PSScriptRoot 'keep_running.ps1'
$arg = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`""
if ($Queue) { $arg += " -Queue `"$Queue`"" }
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arg -WorkingDirectory $PSScriptRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries:$false -DontStopIfGoingOnBatteries:$false `
  -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 10) `
  -ExecutionTimeLimit (New-TimeSpan -Seconds 0) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
  -Description 'Lean-CGRT gym worker (research/lean_cgrt_hrm400/keep_running.ps1)' | Out-Null
Write-Host "Registered $TaskName (at logon of $env:USERNAME, AC power only). Remove with -Uninstall."
