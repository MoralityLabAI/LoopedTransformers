param(
    [Parameter(Mandatory = $true)][ValidateSet("LOOP-T", "FF-U")][string]$Arm,
    [Parameter(Mandatory = $true)][int]$Seed,
    [Parameter(Mandatory = $true)][string[]]$Families,
    [Parameter(Mandatory = $true)][int]$MaxSteps,
    [Parameter(Mandatory = $true)][string]$PoolDir,
    [Parameter(Mandatory = $true)][string]$ArtifactDir,
    [string]$Variant = "base",
    [string]$Python = "python",
    [ValidateSet("cpu", "cuda", "auto")][string]$Device = "cuda",
    [ValidateSet("idle", "shared")][string]$GpuPolicy = "idle",
    [int]$MaxAttempts = 48,
    [int]$GpuWaitMinutes = 0,
    [int]$SharedFreeMb = 8000,
    [int]$SliceSecondsOverride = 0,
    [int]$OomRetryWaitSeconds = 180,
    [int]$BatchPerFamily = 64,
    [int]$Microbatch = 0,
    [int]$Visits = 8,
    [string]$InitFrom = "",
    [double]$LearningRate = 0.001,
    [int]$WarmupSteps = 200,
    [int]$EvalEvery = 250,
    [switch]$DeepSupervision,
    [switch]$InputInjection,
    [switch]$PermuteTargets,
    [double]$Beta = 0.25,
    [switch]$Curriculum,
    [switch]$Staircase,
    [switch]$StopAtGate,
    [string]$ResultDir = "experiments/recurrence_monitorability_v0/runs",
    [string]$BundleDir = "experiments/recurrence_monitorability_v0/datasets",
    [string]$ResourceConfig = "configs/rmp_resource_profile_v0.json",
    [int]$MaxConsecutiveFailures = 5
)

# Runs one RMP training cell in time slices under the caps of the registered resource profile.
# A slice that exits with code 10 has paused cleanly and is resumed by the next attempt.

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
. (Join-Path $PSScriptRoot "lspg_gpu_process_policy.ps1")
# "powershell -File" cannot pass an array, so a comma-separated list is accepted as well.
$Families = @($Families | ForEach-Object { $_ -split "," } | Where-Object { $_ })

function Resolve-RepoPath([string]$Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) { return [System.IO.Path]::GetFullPath($Value) }
    return [System.IO.Path]::GetFullPath((Join-Path $Repo $Value))
}

$Profile = Get-Content -Raw -LiteralPath (Resolve-RepoPath $ResourceConfig) | ConvertFrom-Json
$Caps = $Profile.caps
$CellName = "$Arm-$Variant-s$Seed"
$ResultPath = Resolve-RepoPath $ResultDir
$LogDir = Join-Path $ResultPath "logs"
$ReceiptDir = Join-Path $ResultPath "resource_receipts"
New-Item -ItemType Directory -Force -Path $ResultPath, $LogDir, $ReceiptDir | Out-Null
$TimeoutSeconds = [int]$Caps.child_timeout_seconds
$SliceSeconds = [Math]::Max(60, $TimeoutSeconds - 300)
if ($SliceSecondsOverride -gt 0) { $SliceSeconds = [Math]::Min($SliceSeconds, $SliceSecondsOverride) }
$Threads = [Math]::Max(1, [int]([Environment]::ProcessorCount * [int]$Caps.cpu_pct / 100))

# The owned process is a cmd wrapper around the venv launcher, which starts the real interpreter as
# its own child; stopping only the wrapper leaves training running. Stop the whole tree.
function Stop-OwnedTree([System.Diagnostics.Process]$Process) {
    if ($null -eq $Process -or $Process.HasExited) { return }
    & taskkill.exe /PID $Process.Id /T /F 2>$null | Out-Null
    if (-not $Process.HasExited) { Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue }
}

function Get-ForeignGpuProcesses {
    $foreign = @()
    foreach ($line in @(& nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>$null)) {
        $candidatePid = $line.Trim()
        if ($candidatePid -notmatch "^[0-9]+$") { continue }
        $candidate = Get-CimInstance Win32_Process -Filter "ProcessId=$candidatePid" -ErrorAction SilentlyContinue
        if (
            $null -ne $candidate -and
            -not [string]::IsNullOrWhiteSpace($candidate.CommandLine) -and
            (Test-LspgCpuOnlyNvidiaRegistration -ProcessName $candidate.Name -CommandLine $candidate.CommandLine)
        ) { continue }
        $foreign += [int]$candidatePid
    }
    return $foreign
}

# "idle" is the registered preflight. "shared" tolerates other work on the card but needs
# $SharedFreeMb free in every sample: a neighbour whose usage swings must not be pushed out of
# memory by a cell that started in one of its dips. No process is stopped under either policy.
function Test-GpuReady {
    $ready = $true; $utilization = 0.0; $freeMb = [double]::MaxValue; $foreign = @()
    # On battery this laptop throttles the GPU to a fraction of its speed and drains within the
    # hour, so a slice only starts on mains power. Hosts without a battery report nothing here.
    $power = Get-CimInstance -Namespace root\wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $power -and -not $power.PowerOnline) {
        return [ordered]@{ ready = $false; on_battery = $true; utilization_pct = $null; free_mb = $null; foreign_pids = @() }
    }
    for ($sample = 0; $sample -lt [int]$Profile.preflight.idle_samples; $sample++) {
        if ($sample -gt 0) { Start-Sleep -Seconds 20 }
        $row = (& nvidia-smi --query-gpu=utilization.gpu,memory.total,memory.used --format=csv,noheader,nounits | Select-Object -First 1) -split ","
        $utilization = [Math]::Max($utilization, [double]$row[0].Trim())
        $freeMb = [Math]::Min($freeMb, [double]$row[1].Trim() - [double]$row[2].Trim())
        $foreign = @(Get-ForeignGpuProcesses)
        if ($GpuPolicy -eq "idle") {
            $ready = $ready -and ($utilization -le [double]$Profile.preflight.gpu_utilization_max_pct) -and ($foreign.Count -eq 0)
        } else {
            $ready = $ready -and ($freeMb -ge $SharedFreeMb)
        }
        if (-not $ready) { break }
    }
    return [ordered]@{ ready = $ready; utilization_pct = $utilization; free_mb = $freeMb; foreign_pids = $foreign }
}

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class RmpCellJobObject {
  [DllImport("kernel32.dll", CharSet = CharSet.Unicode)] public static extern IntPtr CreateJobObject(IntPtr a, string n);
  [DllImport("kernel32.dll")] public static extern bool AssignProcessToJobObject(IntPtr j, IntPtr p);
  [DllImport("kernel32.dll")] public static extern bool SetInformationJobObject(IntPtr j, int t, IntPtr p, uint n);
  [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr h);
  public const int Extended = 9;
  public const int Cpu = 15;
  public const uint ProcessMemory = 0x100;
  public const uint KillOnClose = 0x2000;
  public const uint CpuEnable = 0x1;
  public const uint CpuHardCap = 0x4;
  [StructLayout(LayoutKind.Sequential)] public struct IO { public ulong a,b,c,d,e,f; }
  [StructLayout(LayoutKind.Sequential)] public struct Basic { public long a,b; public uint flags; public UIntPtr c,d; public uint e; public long f; public uint g,h; }
  [StructLayout(LayoutKind.Sequential)] public struct ExtendedInfo { public Basic basic; public IO io; public UIntPtr processMemory, jobMemory, peakProcess, peakJob; }
  [StructLayout(LayoutKind.Sequential)] public struct CpuInfo { public uint flags, rate; }
}
'@

function New-CappedJob([string]$Name) {
    $job = [RmpCellJobObject]::CreateJobObject([IntPtr]::Zero, $Name)
    $limit = New-Object RmpCellJobObject+ExtendedInfo
    $limit.basic.flags = [RmpCellJobObject]::ProcessMemory -bor [RmpCellJobObject]::KillOnClose
    $limit.processMemory = [UIntPtr]::new([UInt64]$Caps.ram_bytes)
    $size = [Runtime.InteropServices.Marshal]::SizeOf($limit)
    $ptr = [Runtime.InteropServices.Marshal]::AllocHGlobal($size)
    [Runtime.InteropServices.Marshal]::StructureToPtr($limit, $ptr, $false)
    if (-not [RmpCellJobObject]::SetInformationJobObject($job, [RmpCellJobObject]::Extended, $ptr, $size)) { throw "memory cap setup failed" }
    [Runtime.InteropServices.Marshal]::FreeHGlobal($ptr)
    $cpu = New-Object RmpCellJobObject+CpuInfo
    $cpu.flags = [RmpCellJobObject]::CpuEnable -bor [RmpCellJobObject]::CpuHardCap
    $cpu.rate = [uint32]([int]$Caps.cpu_pct * 100)
    $size = [Runtime.InteropServices.Marshal]::SizeOf($cpu)
    $ptr = [Runtime.InteropServices.Marshal]::AllocHGlobal($size)
    [Runtime.InteropServices.Marshal]::StructureToPtr($cpu, $ptr, $false)
    if (-not [RmpCellJobObject]::SetInformationJobObject($job, [RmpCellJobObject]::Cpu, $ptr, $size)) { throw "CPU cap setup failed" }
    [Runtime.InteropServices.Marshal]::FreeHGlobal($ptr)
    return $job
}

$arguments = @(
    "-m", "research_gym.scripts.run_rmp_training_cell",
    "--arm", $Arm, "--seed", [string]$Seed, "--variant", $Variant,
    "--max-steps", [string]$MaxSteps,
    "--batch-per-family", [string]$BatchPerFamily,
    "--microbatch", [string]$Microbatch,
    "--visits", [string]$Visits,
    "--learning-rate", [string]$LearningRate,
    "--warmup-steps", [string]$WarmupSteps,
    "--eval-every", [string]$EvalEvery,
    "--bundle-dir", (Resolve-RepoPath $BundleDir),
    "--result-dir", $ResultPath,
    "--pool-dir", (Resolve-RepoPath $PoolDir),
    "--artifact-dir", (Resolve-RepoPath $ArtifactDir),
    "--vram-mb", [string]$Caps.vram_mb,
    "--max-seconds", [string]$SliceSeconds,
    "--threads", [string]$Threads,
    "--beta", [string]$Beta,
    "--families"
) + $Families
if ($DeepSupervision) { $arguments += "--deep-supervision" }
if ($InputInjection) { $arguments += "--input-injection" }
if ($PermuteTargets) { $arguments += "--permute-targets" }
if ($Curriculum) { $arguments += "--curriculum" }
if ($Staircase) { $arguments += "--staircase" }
if ($StopAtGate) { $arguments += "--stop-at-gate" }
if ($InitFrom) { $arguments += @("--init-from", (Resolve-RepoPath $InitFrom)) }

$final = "attempts_exhausted"
$consecutiveFailures = 0
# Attempts are numbered across invocations so a later run never overwrites an earlier receipt.
$firstAttempt = 1 + @(Get-ChildItem -LiteralPath $ReceiptDir -Filter "$CellName.attempt-*.resource_receipt.json").Count
for ($attempt = $firstAttempt; $attempt -lt $firstAttempt + $MaxAttempts; $attempt++) {
    $started = Get-Date
    $receipt = Join-Path $ReceiptDir "$CellName.attempt-$attempt.resource_receipt.json"
    $stdout = Join-Path $LogDir "$CellName.attempt-$attempt.stdout.log"
    $stderr = Join-Path $LogDir "$CellName.attempt-$attempt.stderr.log"
    $gpu = $null
    $attemptDevice = $Device
    if ($Device -ne "cpu") {
        $deadline = (Get-Date).AddMinutes($GpuWaitMinutes)
        do {
            $gpu = Test-GpuReady
            if ($gpu.ready) { break }
            if ((Get-Date) -lt $deadline) { Start-Sleep -Seconds 60 }
        } while ((Get-Date) -lt $deadline)
        # "auto" takes the GPU when it is ready and otherwise spends this slice on the CPU.
        if ($Device -eq "auto") { $attemptDevice = if ($gpu.ready) { "cuda" } else { "cpu" } }
    }
    $status = "running"; $abortReason = $null; $proc = $null
    $peakRamMb = 0.0; $peakIoMbS = 0.0; $peakVramMb = 0.0; $sumCpuPct = 0.0; $ioViolations = 0; $samples = 0
    if ($attemptDevice -eq "cuda" -and -not $gpu.ready) {
        $status = "construction_failure"; $abortReason = "gpu_not_ready_under_${GpuPolicy}_policy"
    } else {
        $job = New-CappedJob "rmp-cell-$PID-$CellName-$attempt"
        try {
            $attemptArguments = $arguments + @("--device", $attemptDevice)
            # A CPU slice taken while waiting for the GPU is kept short so the card is re-checked often.
            if ($Device -eq "auto" -and $attemptDevice -eq "cpu") { $attemptArguments += @("--max-seconds", "600") }
            # On this host, Python children started with Start-Process -RedirectStandardOutput die at
            # random with exception 0xc000070a. The shell does the redirection instead, and a short
            # delay lets the job object be attached before the interpreter starts.
            $inner = '"{0}" -u {1} > "{2}" 2> "{3}"' -f $Python, ($attemptArguments -join " "), $stdout, $stderr
            $proc = Start-Process -FilePath "cmd.exe" -ArgumentList ('/c "ping -n 3 127.0.0.1 > nul & ' + $inner + '"') -WorkingDirectory $Repo -PassThru -WindowStyle Hidden
            if (-not [RmpCellJobObject]::AssignProcessToJobObject($job, $proc.Handle)) { throw "process assignment to job failed" }
            # The host is shared: yield the CPU to whatever else is running.
            $proc.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
            $procStarted = Get-Date  # the timeout covers the child, not the wait for the GPU
            $lastIo = 0.0; $lastCpuSeconds = 0.0; $lastSample = Get-Date
            while (-not $proc.HasExited) {
                Start-Sleep -Seconds 2
                $proc.Refresh(); $samples++
                # The shell starts a venv launcher, which starts the interpreter; all share the job.
                $all = @(Get-CimInstance Win32_Process -Property ProcessId, ParentProcessId, WorkingSetSize, KernelModeTime, UserModeTime, ReadTransferCount, WriteTransferCount -ErrorAction SilentlyContinue)
                $treePids = @([string]$proc.Id)
                do {
                    $grown = @($all | Where-Object { $treePids -contains [string]$_.ParentProcessId -and $treePids -notcontains [string]$_.ProcessId } | ForEach-Object { [string]$_.ProcessId })
                    $treePids += $grown
                } while ($grown.Count -gt 0)
                $tree = @($all | Where-Object { $treePids -contains [string]$_.ProcessId })
                $ramMb = ($tree | Measure-Object -Property WorkingSetSize -Sum).Sum / 1MB
                $peakRamMb = [Math]::Max($peakRamMb, $ramMb)
                $now = Get-Date
                $elapsed = [Math]::Max(0.001, ($now - $lastSample).TotalSeconds)
                $cpuSeconds = (($tree | ForEach-Object { [double]$_.KernelModeTime + [double]$_.UserModeTime }) | Measure-Object -Sum).Sum / 1e7
                $sumCpuPct += [Math]::Max(0.0, 100.0 * ($cpuSeconds - $lastCpuSeconds) / ($elapsed * [Environment]::ProcessorCount))
                $lastCpuSeconds = $cpuSeconds
                if ($tree.Count -gt 0) {
                    $currentIo = (($tree | ForEach-Object { [double]$_.ReadTransferCount + [double]$_.WriteTransferCount }) | Measure-Object -Sum).Sum
                    if ($lastIo -gt 0) {
                        $ioRate = ($currentIo - $lastIo) / $elapsed
                        $peakIoMbS = [Math]::Max($peakIoMbS, $ioRate / 1MB)
                        $ioViolations = if ($ioRate -gt [double]$Caps.io_bytes_per_second) { $ioViolations + 1 } else { 0 }
                    }
                    $lastIo = $currentIo
                }
                if ($attemptDevice -eq "cuda") {
                    foreach ($line in @(& nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>$null)) {
                        $parts = $line -split ","
                        if ($parts.Count -ge 2 -and $treePids -contains $parts[0].Trim() -and $parts[1].Trim() -match "^[0-9]+(?:\.[0-9]+)?$") {
                            $peakVramMb = [Math]::Max($peakVramMb, [double]$parts[1].Trim())
                        }
                    }
                }
                $lastSample = $now
                if ($ioViolations -ge [int]$Caps.io_sustained_samples) { $abortReason = "sustained_io_cap_exceeded"; break }
                if ($peakVramMb -gt [double]$Caps.vram_mb) { $abortReason = "vram_cap_exceeded"; break }
                if (($now - $procStarted).TotalSeconds -gt $TimeoutSeconds) { $abortReason = "timeout"; break }
            }
            if ($null -ne $abortReason -and -not $proc.HasExited) {
                Stop-OwnedTree $proc
                $proc.WaitForExit()
                $status = "aborted"
            } else {
                $proc.WaitForExit()
                $status = switch ($proc.ExitCode) { 0 { "completed" } 10 { "paused" } default { "failed" } }
                if ($status -eq "failed") { $abortReason = "process_exit_$($proc.ExitCode)" }
                # On a shared card another job can exhaust memory mid-slice. The cell resumes from
                # its last saved evaluation point, so this is retried after a pause, not treated as fatal.
                if ($status -eq "failed" -and (Select-String -LiteralPath $stderr -Pattern "out of memory" -Quiet)) {
                    $status = "gpu_out_of_memory"
                }
            }
        } catch {
            $status = "construction_failure"; $abortReason = $_.Exception.Message
            Stop-OwnedTree $proc
        } finally {
            [RmpCellJobObject]::CloseHandle($job) | Out-Null
        }
    }
    $finished = Get-Date
    $json = ([ordered]@{
        cell_id = $CellName
        attempt = $attempt
        status = $status
        abort_reason = $abortReason
        device = $attemptDevice
        gpu_policy = if ($Device -ne "cpu") { $GpuPolicy } else { $null }
        gpu_preflight = $gpu
        started_utc = $started.ToUniversalTime().ToString("o")
        finished_utc = $finished.ToUniversalTime().ToString("o")
        elapsed_seconds = [Math]::Round(($finished - $started).TotalSeconds, 3)
        caps = $Caps
        peak_ram_mb = [Math]::Round($peakRamMb, 3)
        peak_io_mb_s = [Math]::Round($peakIoMbS, 3)
        peak_vram_mb = [Math]::Round($peakVramMb, 3)
        avg_cpu_pct = if ($samples) { [Math]::Round($sumCpuPct / $samples, 3) } else { 0 }
        owned_pid = if ($null -eq $proc) { $null } else { $proc.Id }
        cleanup_passed = if ($null -eq $proc) { $true } else { $proc.HasExited }
        foreign_process_policy = "observed but never terminated"
    }) | ConvertTo-Json -Depth 8
    [System.IO.File]::WriteAllText($receipt, $json + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
    Write-Output "$CellName attempt ${attempt}: $status $abortReason"
    # A slice cut off by the wall-clock timeout (for example after the host slept) is simply resumed.
    if ($status -eq "aborted" -and $abortReason -eq "timeout") { continue }
    # A failed slice loses only the steps since the cell's last saved evaluation, so it is retried
    # after a pause; repeated failures in a row stop the run. A slice stopped for sustained I/O is
    # treated the same way: the cap is enforced by stopping it, and the burst is usually the cell
    # reloading its training pools on resume after the host evicted them from the file cache.
    $ioAbort = $status -eq "aborted" -and $abortReason -eq "sustained_io_cap_exceeded"
    if ($ioAbort -or $status -in @("gpu_out_of_memory", "failed")) {
        $consecutiveFailures++
        if ($consecutiveFailures -ge $MaxConsecutiveFailures) { $final = $status; break }
        Start-Sleep -Seconds $OomRetryWaitSeconds
        continue
    }
    $consecutiveFailures = 0
    if ($status -ne "paused") { $final = $status; break }
}
if ($final -ne "completed") { exit 1 }
