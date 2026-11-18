param([string]$OutputPath)
$os=Get-CimInstance Win32_OperatingSystem
$gpu=& nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader
$workloads=@(Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 12 ProcessName,Id,WorkingSet64,CPU)
$audit=@{timestamp_utc=[DateTime]::UtcNow.ToString('o'); operating_system=$os.Caption; total_ram_kib=$os.TotalVisibleMemorySize; free_ram_kib=$os.FreePhysicalMemory; total_virtual_kib=$os.TotalVirtualMemorySize; free_virtual_kib=$os.FreeVirtualMemory; nvidia_smi=$gpu; largest_resident_workloads=$workloads; other_workloads_terminated=$false; sealed_test_accessed=$false}
$audit | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $OutputPath -Encoding utf8
