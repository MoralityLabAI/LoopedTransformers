param([int]$TargetProcessId,[string]$OutputPath,[int]$MaximumSeconds=2100)
$ErrorActionPreference='Stop'
$samples=[System.Collections.Generic.List[object]]::new()
$watch=[Diagnostics.Stopwatch]::StartNew()
while ($watch.Elapsed.TotalSeconds -lt $MaximumSeconds) {
    try { $running=Get-Process -Id $TargetProcessId -ErrorAction Stop } catch { break }
    $gpu=@(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory | Where-Object { $_.Name -like "pid_${TargetProcessId}_*" } | Select-Object Name,DedicatedUsage,SharedUsage,TotalCommitted)
    $samples.Add(@{elapsed_seconds=$watch.Elapsed.TotalSeconds; timestamp_utc=[DateTime]::UtcNow.ToString('o'); pid=$TargetProcessId; working_set_bytes=$running.WorkingSet64; gpu_memory=$gpu})
    ConvertTo-Json -InputObject $samples.ToArray() -Depth 6 | Set-Content -LiteralPath $OutputPath -Encoding utf8
    Start-Sleep -Seconds 10
}
