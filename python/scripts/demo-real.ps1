[CmdletBinding()]
param(
    [string]$BaseUrl = "http://127.0.0.1:8001",
    [ValidateSet(
        "all",
        "heart_failure",
        "stemi_interaction",
        "pneumonia_allergy"
    )]
    [string]$Case = "all",
    [string]$OutputPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$pythonRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $pythonRoot ".venv\Scripts\python.exe"
$validator = Join-Path $PSScriptRoot "validate-real.py"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Missing .venv. Run .\scripts\setup-local.ps1 first."
}

$arguments = @($validator, "--base-url", $BaseUrl, "--case", $Case)
if ($OutputPath) {
    $arguments += @("--output", $OutputPath)
}

Push-Location $pythonRoot
try {
    & $venvPython @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "One or more real-case validation checks failed."
    }
}
finally {
    Pop-Location
}
