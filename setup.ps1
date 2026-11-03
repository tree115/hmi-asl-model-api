param(
    [string]$EnvironmentRoot = (Join-Path $env:USERPROFILE '.cache\asl-practice-ai\venv-api'),
    [switch]$PrepareONNX
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($null -eq $uvCommand) { throw 'Install uv, or use the manual Python setup in README.md.' }
function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Executable (exit $LASTEXITCODE)" }
}
$aiPython = Join-Path $EnvironmentRoot 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $aiPython)) {
    Invoke-Checked $uvCommand.Source @('venv', '--python', '3.11', $EnvironmentRoot)
}
Invoke-Checked $aiPython @('-c', 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 required"')
Invoke-Checked $uvCommand.Source @('pip', 'sync', '--link-mode', 'copy', '--python', $aiPython, 'requirements.lock.txt')
Invoke-Checked $aiPython @('scripts/prepare_onnx_alphabet.py')
Write-Host "Ready. Python: $aiPython"
