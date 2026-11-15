param([string]$Root='D:\Research Engine\RAM-03',[string]$PriorRoot='D:\Research Engine\RAM-02')
$ErrorActionPreference='Stop'
$ram03Experiment=Join-Path $Root 'LoopedTransformers\experiments\lean_ram03'
$ram03Python=Join-Path $PriorRoot '.venv\Scripts\python.exe'
$ram03Lean=Join-Path $PriorRoot 'toolchains\lean-4.35.0-windows\bin\lean.exe'
$ram03Run=Join-Path $ram03Experiment ('runs\reproduction-'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))
$env:TEMP=Join-Path $Root 'tmp'
$env:TMP=$env:TEMP
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONHASHSEED='0'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:LEAN_NUM_THREADS='2'
$env:PYTHONPATH=Join-Path $PriorRoot 'Compactification-Theory'
$env:LEAN_PATH=$ram03Experiment+';'+(Join-Path $PriorRoot 'MathCan\.lake\build\lib\lean')
$env:PATH=(Split-Path -Parent $ram03Lean)+';'+$env:PATH
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
Push-Location $ram03Experiment
try {
    & $ram03Lean ProbeServer.lean -o ProbeServer.olean
    if($LASTEXITCODE){throw 'Lean helper build failed'}
} finally { Pop-Location }
& $ram03Python -m unittest discover -s "$ram03Experiment\tests" -v
if($LASTEXITCODE){throw 'Unit tests failed'}
& $ram03Python "$ram03Experiment\smoke.py" --lean $ram03Lean --out "$ram03Run\smoke"
if($LASTEXITCODE){throw 'Server smoke failed'}
& $ram03Python "$ram03Experiment\smoke_search.py" --lean $ram03Lean --out "$ram03Run\search-smoke"
if($LASTEXITCODE){throw 'Search smoke failed'}
& $ram03Python "$ram03Experiment\collect.py" --data "$PriorRoot\artifacts\fc12-train" --mathcan "$PriorRoot\MathCan" --lean $ram03Lean --out "$ram03Run\data"
if($LASTEXITCODE){throw 'Collection or independent audit failed'}
& $ram03Python "$ram03Experiment\train_local.py" --data "$ram03Run\data" --out "$ram03Run\training"
if($LASTEXITCODE){throw 'Training failed'}
& $ram03Python "$ram03Experiment\evaluate.py" --mathcan "$PriorRoot\MathCan" --data "$ram03Run\data" --training "$ram03Run\training" --lean $ram03Lean --out "$ram03Run\evaluation"
if($LASTEXITCODE){throw 'Proof evaluation failed'}
& $ram03Python "$ram03Experiment\analyze.py" --data "$ram03Run\data" --training "$ram03Run\training" --evaluation "$ram03Run\evaluation" --out "$ram03Run\analysis"
if($LASTEXITCODE){throw 'Final evidence audit failed'}
& $ram03Python "$ram03Experiment\report_results.py" --analysis "$ram03Run\analysis" --out "$ram03Run\RESULTS.md"
if($LASTEXITCODE){throw 'Results rendering failed'}
Write-Output "RAM-03 complete: $ram03Run"
