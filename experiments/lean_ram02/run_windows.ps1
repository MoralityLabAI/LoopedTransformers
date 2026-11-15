param(
    [Parameter(Mandatory=$true)][string]$Data,
    [Parameter(Mandatory=$true)][string]$Out,
    [string]$Python='python',
    [string]$TempDirectory='',
    [switch]$IncludeFlat
)
$ErrorActionPreference='Stop'
if ($TempDirectory) {
    New-Item -ItemType Directory -Path $TempDirectory -Force | Out-Null
    $env:TEMP=$TempDirectory
    $env:TMP=$TempDirectory
}
$env:PYTHONUTF8='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONHASHSEED='0'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$trainArgs=@("$PSScriptRoot/train_local.py",'--data',$Data,'--out',$Out,'--config',"$PSScriptRoot/config_3050.yaml")
if ($IncludeFlat) { $trainArgs+='--include-flat' }
& $Python @trainArgs
if ($LASTEXITCODE) { throw "RAM-02 campaign failed with exit code $LASTEXITCODE; inspect preserved receipts." }
