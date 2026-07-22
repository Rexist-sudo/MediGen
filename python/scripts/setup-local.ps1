[CmdletBinding()]
param(
    [string]$PythonCommand = "python",
    [switch]$WithTests
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$pythonRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPath = Join-Path $pythonRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$requirementsIntentPath = Join-Path $pythonRoot "requirements-mvp.in"
$requirementsLockPath = Join-Path $pythonRoot "requirements-mvp.lock.txt"
$requirementsPath = if (Test-Path -LiteralPath $requirementsLockPath) {
    $requirementsLockPath
}
else {
    $requirementsIntentPath
}
$envPath = Join-Path $pythonRoot ".env"
$envExamplePath = Join-Path $pythonRoot ".env.example"

if (-not (Test-Path -LiteralPath $envPath)) {
    Copy-Item -LiteralPath $envExamplePath -Destination $envPath
    Write-Host "Created local .env from .env.example."
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating virtual environment at $venvPath"
    & $PythonCommand -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create the virtual environment."
    }
}

Write-Host "Installing runtime dependencies from $requirementsPath"
& $venvPython -m pip install --disable-pip-version-check -r $requirementsPath
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed."
}

& $venvPython -c "import spacy; spacy.load('en_core_web_sm')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing the Presidio English language model."
    & $venvPython -m spacy download en_core_web_sm
    if ($LASTEXITCODE -ne 0) {
        throw "Presidio language-model installation failed."
    }
}

if ($WithTests) {
    & $venvPython -m pip install --disable-pip-version-check pytest
    if ($LASTEXITCODE -ne 0) {
        throw "pytest installation failed."
    }
    Push-Location $pythonRoot
    try {
        & $venvPython -m pytest tests -q
        if ($LASTEXITCODE -ne 0) {
            throw "Test suite failed."
        }
    }
    finally {
        Pop-Location
    }
}

$envLines = Get-Content -LiteralPath $envPath
$backendLine = $envLines | Where-Object { $_ -match '^LLM_BACKEND=' } | Select-Object -First 1
$keyLine = $envLines | Where-Object { $_ -match '^DEEPSEEK_API_KEY=' } | Select-Object -First 1
$backend = if ($backendLine) { $backendLine.Substring("LLM_BACKEND=".Length).Trim() } else { "deepseek" }
$keyConfigured = [bool]($keyLine -and $keyLine.Substring("DEEPSEEK_API_KEY=".Length).Trim())

Write-Host "Local setup complete."
Write-Host "LLM backend: $backend"
Write-Host "DeepSeek key configured: $keyConfigured"
Write-Host "The .env and .venv paths are ignored by Git."
