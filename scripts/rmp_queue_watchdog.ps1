# Watchdog for the RMP queue (scripts/run_rmp_queue.sh). After a Windows GPU reset, a CUDA job can hang with
# its context lost: alive, spinning one CPU core, doing no GPU work, never writing another checkpoint (seen
# 2026-10-25, stuck 23 hours). vramorama's guards do not catch that. Every 2 minutes this checks each of the
# queue's GPU jobs; one older than 10 minutes that shows no GPU engine activity for 10 checks in a row
# (20 minutes) has its process tree stopped. The queue then resumes it from its last checkpoint (training
# resumes exactly; the Phase 2 scripts skip finished work). It touches only this queue's scripts, and it
# exits when no queue runner is left.
param([string]$Log = "C:\Users\patri\Documents\Codex\rmp_v0\queue\watchdog.log")

$pattern = 'research_gym\.scripts\.(run_rmp_training_cell|rmp_phase2_)'
$idle = @{}
function Note([string]$text) { Add-Content -Path $Log -Value ("{0} {1}" -f (Get-Date -Format s), $text) }
Note "watchdog started"
while ($true) {
    $all = Get-CimInstance Win32_Process
    if (-not ($all | Where-Object { $_.CommandLine -match 'run_rmp_queue\.sh' })) { Note "no queue runner left; watchdog exits"; break }
    # The real interpreters (the venv's python.exe is a launcher whose child does the work).
    $jobs = $all | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match $pattern -and
        $_.CommandLine -notmatch '--device cpu' -and $_.ExecutablePath -notlike '*\.venvs\*' }
    $samples = (Get-Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction SilentlyContinue).CounterSamples
    $seen = @{}
    foreach ($job in $jobs) {
        $id = $job.ProcessId
        $seen[$id] = $true
        $busy = ($samples | Where-Object { $_.InstanceName -match "pid_$($id)_" } | Measure-Object CookedValue -Sum).Sum
        if ($busy -gt 0) { $idle[$id] = 0; continue }
        $idle[$id] = 1 + [int]$idle[$id]
        $age = (Get-Date) - $job.CreationDate
        if ($idle[$id] -ge 10 -and $age.TotalMinutes -ge 10) {
            $parent = $all | Where-Object { $_.ProcessId -eq $job.ParentProcessId -and $_.CommandLine -match $pattern }
            $target = if ($parent) { $parent.ProcessId } else { $id }
            Note ("no GPU activity for {0} checks: stopping {1} ({2})" -f $idle[$id], $target, $job.CommandLine.Substring(0, [Math]::Min(160, $job.CommandLine.Length)))
            taskkill /PID $target /T /F | Out-Null
            $idle.Remove($id)
        }
    }
    foreach ($key in @($idle.Keys)) { if (-not $seen.ContainsKey($key)) { $idle.Remove($key) } }
    Start-Sleep -Seconds 120
}
