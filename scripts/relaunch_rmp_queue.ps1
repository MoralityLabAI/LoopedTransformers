# Start whichever part of the RMP queue is not running: the main runner, lane 2, and
# the watchdog. Safe to run at any time, for example after a reboot; every job resumes or skips finished work.
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\relaunch_rmp_queue.ps1
$root = 'C:\Users\patri\Documents\Codex\rmp_v0'
$repo = 'C:\Users\patri\Documents\Codex\LoopedTransformers-rmp'
$bash = 'C:\Program Files\Git\bin\bash.exe'

function Start-Detached([string]$command) {
    # Detached, below normal priority, with this shell's environment (vramorama tags jobs by it).
    $environment = [string[]](Get-ChildItem env: | ForEach-Object { "$($_.Name)=$($_.Value)" })
    $startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{
        EnvironmentVariables = $environment; PriorityClass = [uint32]16384 }
    $result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = $command; CurrentDirectory = $repo; ProcessStartupInformation = $startup }
    "started pid $($result.ProcessId): $command"
}

$running = Get-CimInstance Win32_Process
foreach ($lane in 'main', 'lane2') {
    $queue = "$root\queue\$lane.queue"
    if ($running | Where-Object { $_.CommandLine -match "run_rmp_queue\.sh .*$lane\.queue" }) { "$lane runner already running"; continue }
    # A runner with nothing left to do exits by itself after an hour.
    if (Test-Path "$queue.failed") { "note: $lane has a failed line, which the runner retries: $(Get-Content "$queue.failed" -Tail 1)" }
    Start-Detached ('"{0}" /c/Users/patri/Documents/Codex/LoopedTransformers-rmp/scripts/run_rmp_queue.sh /c/Users/patri/Documents/Codex/rmp_v0 /c/Users/patri/Documents/Codex/rmp_v0/queue/{1}.queue' -f $bash, $lane)
}
Start-Sleep -Seconds 5
if (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'rmp_queue_watchdog' }) { "watchdog already running" }
else { Start-Detached "powershell -NoProfile -ExecutionPolicy Bypass -File $repo\scripts\rmp_queue_watchdog.ps1" }
