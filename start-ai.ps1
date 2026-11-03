param(
    [ValidateSet('onnx_alphabet')][string]$Model = 'onnx_alphabet',
    [int]$Port = 8001,
    [string]$EnvironmentRoot = (Join-Path $env:USERPROFILE '.cache\asl-practice-ai\venv-api')
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$aiPython = Join-Path $EnvironmentRoot 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $aiPython)) { throw 'Run ./setup.ps1 first.' }
if (-not (Test-Path -LiteralPath 'models/onnx_alphabet/manifest.json')) { throw 'Run ./setup.ps1 to prepare ONNX assets.' }
$env:AI_MODEL_PROVIDER = $Model
& $aiPython -m uvicorn api.main:app --host 127.0.0.1 --port $Port --workers 1
if ($LASTEXITCODE -ne 0) { throw "AI service exited: $LASTEXITCODE" }
