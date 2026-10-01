param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$RunnerArguments,
    [string]$LogDir = "experiments/recurrence_monitorability_v0/runs/logs"
)

# Starts scripts/run_rmp_training_cell.ps1 as a process that outlives the calling session.
# A runner started as a child of an interactive agent session is killed when that session ends.

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogPath = Join-Path $Repo $LogDir
New-Item -ItemType Directory -Force -Path $LogPath | Out-Null
$log = Join-Path $LogPath "$Name.runner.log"
$command = 'cmd /c "powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_rmp_training_cell.ps1 {0} >> "{1}" 2>&1"' -f $RunnerArguments, $log
$created = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $command; CurrentDirectory = $Repo }
if ($created.ReturnValue -ne 0) { throw "process creation failed with code $($created.ReturnValue)" }
Write-Output "started $Name as pid $($created.ProcessId); log: $log"
