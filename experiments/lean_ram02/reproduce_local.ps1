param([string]$Root='D:\Research Engine\RAM-02')
$ErrorActionPreference='Stop'
$ram02Python=Join-Path $Root '.venv\Scripts\python.exe'
$ram02Experiment=Join-Path $Root 'LoopedTransformers\experiments\lean_ram02'
$ram02Lean=Join-Path $Root 'toolchains\lean-4.35.0-windows\bin\lean.exe'
$ram02Lake=Join-Path $Root 'toolchains\lean-4.35.0-windows\bin\lake.exe'
$ram02Id=[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$ram02Output=Join-Path $ram02Experiment "runs\reproduction-$ram02Id"
$env:TEMP=Join-Path $Root 'tmp'
$env:TMP=$env:TEMP
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONHASHSEED='0'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:LEAN_NUM_THREADS='2'
$env:PYTHONPATH=Join-Path $Root 'Compactification-Theory'
$env:LEAN_PATH=Join-Path $Root 'MathCan\.lake\build\lib\lean'
$env:PATH=(Split-Path -Parent $ram02Lean)+';'+$env:PATH
& $ram02Python -m unittest discover -s "$ram02Experiment\tests" -v
if($LASTEXITCODE){throw 'Unit tests failed'}
& "$ram02Experiment\run_windows.ps1" -Data "$Root\artifacts\fc12-train" -Out "$ram02Output\training" -Python $ram02Python -TempDirectory $env:TEMP -IncludeFlat
& $ram02Python "$ram02Experiment\audit_ram01.py" --data "$Root\artifacts\fc12-train" --weights "$Root\artifacts\ram01-models" --out "$ram02Output\ram01-audit.json"
if($LASTEXITCODE){throw 'RAM-01 artifact audit failed'}
& $ram02Python "$ram02Experiment\evaluate.py" --mathcan "$Root\MathCan" --gym "$Root\Compactification-Theory" --weights "$ram02Output\training" --out "$ram02Output\regression" --lean $ram02Lean --lake $ram02Lake --build
if($LASTEXITCODE){throw 'Lean regression failed'}
& $ram02Python "$ram02Experiment\smoke_branches.py" --lean $ram02Lean --out "$ram02Output\branch-smoke"
if($LASTEXITCODE){throw 'Real Lean branch tests failed'}
& $ram02Python "$ram02Experiment\analyze.py" --data "$Root\artifacts\fc12-train" --training "$ram02Output\training" --regression "$ram02Output\regression" --out "$ram02Output\analysis"
if($LASTEXITCODE){throw 'Evidence analysis failed'}
& $ram02Python "$ram02Experiment\report_results.py" --analysis "$ram02Output\analysis" --audit "$ram02Output\ram01-audit.json" --out "$ram02Output\RESULTS.md"
if($LASTEXITCODE){throw 'Results rendering failed'}
Write-Output "RAM-02 reproduction complete: $ram02Output"
