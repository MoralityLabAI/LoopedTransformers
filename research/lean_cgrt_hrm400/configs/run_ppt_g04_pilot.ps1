param(
    [string]$Python = 'python',
    [string]$Out = 'runs/ppt-g04-pilot'
)
$ErrorActionPreference = 'Stop'
$pilotRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Out)
New-Item -ItemType Directory -Path $pilotRoot -Force | Out-Null
Set-Location (Join-Path $PSScriptRoot '..')
$env:PYTHONUTF8 = '1'
$pilotPython = $Python

function Invoke-Gym {
    & $pilotPython -u -m ppt_gym.ram @args
    if ($LASTEXITCODE -ne 0) { throw "PPT gym failed with code $LASTEXITCODE" }
}

function Freeze-Cohort($name, $seed, $families, $rounds) {
    $path = Join-Path $pilotRoot "$name-cohort.json"
    if (-not (Test-Path $path)) {
        Invoke-Gym freeze $path --seed $seed --families $families --queries 1 --rounds $rounds
    }
    return $path
}

function Run-Pilot($name, $cohort, $profile, [string[]]$options) {
    $db = Join-Path $pilotRoot "$name.sqlite"
    $report = Join-Path $pilotRoot "$name-report"
    $audit = Join-Path $pilotRoot "$name-memory.json"
    if (-not (Test-Path $report)) {
        Write-Host "Running $name"
        Invoke-Gym --ram $profile run $cohort $db --cpu-seconds 6 @options --audit $audit
        $staged = Join-Path $pilotRoot "$name-report-$PID"
        Invoke-Gym --ram $profile report $cohort $db $staged
        Move-Item -LiteralPath $staged -Destination $report
    }
}

$smoke = Freeze-Cohort '01' 904 1 1
$smokeDb = Join-Path $pilotRoot '01.sqlite'
$partial = Join-Path $pilotRoot '01-partial.json'
if (-not (Test-Path $partial)) {
    Invoke-Gym --ram 2 run $smoke $smokeDb --cpu-seconds 6 --support 1 --arms sgd frozen horn --steps 5 --audit $partial
}
Run-Pilot '01' $smoke 2 @('--support','1','--arms','sgd','frozen','horn')
if (-not (Test-Path (Join-Path $pilotRoot '01-replay.json'))) {
    Invoke-Gym --ram 2 replay $smoke $smokeDb (Join-Path $pilotRoot '01-replay.json')
}

$curves = Freeze-Cohort '02' 905 1 1
$curveArms = @('--arms','sgd','rls','rank8','retrieval','frozen','shuffled','extra_compute','auto8','horn')
Run-Pilot '02-micro' $curves 4 $curveArms
Run-Pilot '02-tiny' $curves 6 (@('--preset','tiny') + $curveArms)

$depths = Freeze-Cohort '03' 906 1 1
$depthOptions = @('--support','0','--meta-steps','2','--depths','1','2','4','8','16','--max-macs','90000000')
Run-Pilot '03' $depths 4 ($depthOptions + @('--arms','frozen','static','auto8','horn','no_rollback'))
Run-Pilot '04' $depths 4 (@('--plan','b') + $depthOptions + @('--arms','frozen','zero_messages','rolled_messages','horn'))

$persistent = Freeze-Cohort '05' 907 2 2
Run-Pilot '05' $persistent 6 @('--persistent','--rounds','2','--collector','horn','--meta-steps','4','--backbone-every','8','--final-only','--arms','sgd','frozen','horn')
$persistentDb = Join-Path $pilotRoot '05.sqlite'
$persistentReplay = Join-Path $pilotRoot '05-replay.json'
if (-not (Test-Path $persistentReplay)) {
    Invoke-Gym --ram 6 replay $persistent $persistentDb $persistentReplay
}
$bundle = Join-Path $pilotRoot '06-datagoblin'
if (-not (Test-Path $bundle)) {
    Invoke-Gym --ram 6 export $persistent $persistentDb $bundle --replay $persistentReplay
}
