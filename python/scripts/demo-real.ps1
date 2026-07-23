[CmdletBinding()]
param(
    [string]$BaseUrl = "http://127.0.0.1:8001",
    [ValidateSet(
        "all",
        "none",
        "heart_failure",
        "stemi_interaction",
        "pneumonia_allergy"
    )]
    [string]$Case = "all",
    [string]$OutputPath = "",
    [ValidateSet("mini_onerec_mvp", "rule_v1_fallback")]
    [string]$ExpectedRanking = "mini_onerec_mvp",
    [ValidateSet("auto", "deepseek_generated", "catalog_fallback")]
    [string]$ExpectedContent = "deepseek_generated",
    [string]$ExpectedFallbackReason = "",
    [switch]$ModelScenarios,
    [switch]$RequireFallbackDisabled,
    [switch]$AllowModelNotReady
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$pythonRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$rootPython = Join-Path $pythonRoot ".venv-minionerec\python.exe"
$scriptsPython = Join-Path $pythonRoot ".venv-minionerec\Scripts\python.exe"
$venvPython = if (Test-Path -LiteralPath $rootPython) { $rootPython } else { $scriptsPython }
$validator = Join-Path $PSScriptRoot "validate-real.py"
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Missing .venv-minionerec. Run '.\scripts\local.ps1 setup' first."
}

$arguments = @(
    $validator,
    "--base-url", $BaseUrl,
    "--case", $Case,
    "--expected-ranking", $ExpectedRanking,
    "--expected-content", $ExpectedContent
)
if ($OutputPath) {
    $arguments += @("--output", $OutputPath)
}
if ($ExpectedFallbackReason) {
    $arguments += @("--expected-fallback-reason", $ExpectedFallbackReason)
}
if ($ModelScenarios) { $arguments += "--model-scenarios" }
if ($RequireFallbackDisabled) { $arguments += "--require-fallback-disabled" }
if ($AllowModelNotReady) { $arguments += "--allow-model-not-ready" }

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
