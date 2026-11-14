param([string]$Root='D:\Research Engine\RAM-04',[string]$PriorRoot='D:\Research Engine\RAM-02')
$ErrorActionPreference='Stop'
$ram04Experiment=Join-Path $Root 'LoopedTransformers\experiments\lean_ram04'
$ram04Python=Join-Path $PriorRoot '.venv\Scripts\python.exe'
$ram04Lean=Join-Path $Root 'toolchains\lean-4.35.0-windows\bin\lean.exe'
if(!(Test-Path -LiteralPath $ram04Lean)){$ram04Lean=Join-Path $PriorRoot 'toolchains\lean-4.35.0-windows\bin\lean.exe'}
$ram04Run=Join-Path $ram04Experiment ('runs\reproduction-'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))
$env:TEMP=Join-Path $Root 'tmp';$env:TMP=$env:TEMP
$env:PYTHONUTF8='1';$env:PYTHONDONTWRITEBYTECODE='1';$env:PYTHONHASHSEED='0';$env:OMP_NUM_THREADS='2';$env:MKL_NUM_THREADS='2';$env:LEAN_NUM_THREADS='2'
$env:PYTHONPATH=Join-Path $PriorRoot 'Compactification-Theory'
$env:LEAN_PATH=$ram04Experiment+';'+(Join-Path $PriorRoot 'MathCan\.lake\build\lib\lean')
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
& $ram04Python "$ram04Experiment\build_probe.py" --lean $ram04Lean --out "$ram04Run\build"
if($LASTEXITCODE){throw 'Helper build failed'}
& $ram04Python -m unittest discover -s "$ram04Experiment\tests" -v
if($LASTEXITCODE){throw 'Integrity tests failed'}
foreach($ram04Smoke in @('smoke','smoke_families','smoke_search')){& $ram04Python "$ram04Experiment\$ram04Smoke.py" --lean $ram04Lean --out "$ram04Run\$ram04Smoke";if($LASTEXITCODE){throw "$ram04Smoke failed"}}
& $ram04Python "$ram04Experiment\collect.py" --data "$PriorRoot\artifacts\fc12-train" --mathcan "$PriorRoot\MathCan" --lean $ram04Lean --out "$ram04Run\data"
if($LASTEXITCODE){throw 'Collection/audit failed'}
& $ram04Python "$ram04Experiment\train_local.py" --data "$ram04Run\data" --out "$ram04Run\training"
if($LASTEXITCODE){throw 'Supervision gate/training failed'}
& $ram04Python "$ram04Experiment\freeze.py" --training "$ram04Run\training" --out "$ram04Run\training\freeze.json"
if($LASTEXITCODE){throw 'Pre-evaluation freeze failed'}
& $ram04Python "$ram04Experiment\evaluate.py" --mathcan "$PriorRoot\MathCan" --data "$ram04Run\data" --training "$ram04Run\training" --lean $ram04Lean --out "$ram04Run\evaluation"
if($LASTEXITCODE){throw 'Proof comparison failed'}
& $ram04Python "$ram04Experiment\analyze.py" --data "$ram04Run\data" --training "$ram04Run\training" --evaluation "$ram04Run\evaluation" --out "$ram04Run\analysis"
if($LASTEXITCODE){throw 'Evidence audit failed'}
& $ram04Python "$ram04Experiment\report_results.py" --analysis "$ram04Run\analysis" --out "$ram04Run\RESULTS.md"
if($LASTEXITCODE){throw 'Results rendering failed'}
Write-Output "RAM-04 completed: $ram04Run"
