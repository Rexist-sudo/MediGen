[CmdletBinding()]
param(
    [string]$HostAddress = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 8001
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$pythonRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPython = Join-Path $pythonRoot ".venv\Scripts\python.exe"
$envPath = Join-Path $pythonRoot ".env"
$runtimePath = Join-Path $pythonRoot ".runtime"

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Missing .venv. Run .\scripts\setup-local.ps1 first."
}
if (-not (Test-Path -LiteralPath $envPath)) {
    throw "Missing .env. Run .\scripts\setup-local.ps1 and configure it first."
}

$envLines = Get-Content -LiteralPath $envPath
$backendLine = $envLines | Where-Object { $_ -match '^LLM_BACKEND=' } | Select-Object -First 1
$keyLine = $envLines | Where-Object { $_ -match '^DEEPSEEK_API_KEY=' } | Select-Object -First 1
$backend = if ($backendLine) { $backendLine.Substring("LLM_BACKEND=".Length).Trim() } else { "deepseek" }
$keyConfigured = [bool]($keyLine -and $keyLine.Substring("DEEPSEEK_API_KEY=".Length).Trim())
if ($backend -eq "deepseek" -and -not $keyConfigured) {
    throw "DEEPSEEK_API_KEY is empty in .env."
}

$healthUrl = "http://${HostAddress}:${Port}/health"
try {
    $existingHealth = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
    if ($existingHealth.service -eq "medigen-mvp") {
        $existingReadiness = Invoke-RestMethod -Uri "http://${HostAddress}:${Port}/ready" -TimeoutSec 10
        if ($existingReadiness.status -eq "ready") {
            Write-Output "MediGen is already ready at http://${HostAddress}:${Port}"
            return
        }
        throw "A MediGen process is already running but its dependencies are not ready. Stop it and run deploy-local.ps1 again."
    }
}
catch {
    # No MediGen instance answered; continue with a port ownership check.
}

$listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
if ($listeners.Count -gt 0) {
    throw "Port $Port is already in use by another process."
}

New-Item -ItemType Directory -Path $runtimePath -Force | Out-Null
$stdoutPath = Join-Path $runtimePath "uvicorn-${Port}.stdout.log"
$stderrPath = Join-Path $runtimePath "uvicorn-${Port}.stderr.log"

Start-Process `
    -FilePath $venvPython `
    -ArgumentList '-m', 'uvicorn', 'src.api.main:app', '--host', $HostAddress, '--port', $Port `
    -WorkingDirectory $pythonRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath

$ready = $false
$lastReadiness = $null
for ($attempt = 0; $attempt -lt 180; $attempt++) {
    Start-Sleep -Seconds 1
    try {
        $lastReadiness = Invoke-RestMethod -Uri "http://${HostAddress}:${Port}/ready" -TimeoutSec 5
        if ($lastReadiness.status -eq "ready") {
            $ready = $true
            break
        }
    }
    catch {
        # Continue polling until the bounded startup window expires.
    }
}

if (-not $ready) {
    $tail = if (Test-Path -LiteralPath $stderrPath) {
        (Get-Content -Tail 20 -LiteralPath $stderrPath) -join [Environment]::NewLine
    }
    else {
        "No stderr log was created."
    }
    $dependencyText = if ($null -ne $lastReadiness) {
        $lastReadiness.dependencies | ConvertTo-Json -Compress
    }
    else {
        "readiness endpoint did not respond"
    }
    throw "MediGen did not become ready within 180 seconds. Dependencies: $dependencyText`n$tail"
}

$processes = @(
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -eq 'python.exe' -and
            $_.CommandLine -like "*MediGen*uvicorn*--port*$Port*"
        }
)
$pidPath = Join-Path $runtimePath "uvicorn-${Port}.pids"
$processes.ProcessId | Set-Content -LiteralPath $pidPath -Encoding ascii

$readiness = Invoke-RestMethod -Uri "http://${HostAddress}:${Port}/ready" -TimeoutSec 10
Write-Output "MediGen started at http://${HostAddress}:${Port}"
Write-Output "Web UI: http://${HostAddress}:${Port}/"
Write-Output "Swagger UI: http://${HostAddress}:${Port}/docs"
Write-Output "LLM backend: $($readiness.llm_backend)"
Write-Output "DeepSeek configured: $($readiness.deepseek_configured)"
Write-Output "Recommendation store loaded: $($readiness.recommendation_store_loaded)"
Write-Output "Dependencies: $($readiness.dependencies | ConvertTo-Json -Compress)"
Write-Output "Runtime logs: $runtimePath"
