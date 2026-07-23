[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("setup", "deploy", "infra", "start", "stop", "status")]
    [string]$Action = "deploy",
    [string]$PythonCommand = "py",
    [string[]]$PythonArgs = @("-3.11"),
    [string]$HostAddress = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 8001,
    [ValidateRange(30, 900)]
    [int]$InfrastructureTimeoutSeconds = 420,
    [switch]$SkipInstall,
    [switch]$RunTests,
    [switch]$SkipModelCheck
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$pythonRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venvPath = Join-Path $pythonRoot ".venv-minionerec"
$envPath = Join-Path $pythonRoot ".env"
$envExamplePath = Join-Path $pythonRoot ".env.example"
$runtimePath = Join-Path $pythonRoot ".runtime"

function Get-VenvPython {
    $candidates = @(
        (Join-Path $venvPath "python.exe"),
        (Join-Path $venvPath "Scripts\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }
    return $candidates[-1]
}

function Get-DotEnvValue {
    param(
        [Parameter(Mandatory)]
        [string]$Name,
        [string]$Default = ""
    )

    if (-not (Test-Path -LiteralPath $envPath)) {
        return $Default
    }
    $line = Get-Content -LiteralPath $envPath |
        Where-Object { $_ -match "^\s*$([regex]::Escape($Name))\s*=" } |
        Select-Object -First 1
    if (-not $line) {
        return $Default
    }
    return ($line -split "=", 2)[1].Trim().Trim('"').Trim("'")
}

function Assert-PythonEnvironment {
    $venvPython = Get-VenvPython
    if (-not (Test-Path -LiteralPath $venvPython)) {
        throw "Missing .venv-minionerec. Run '.\scripts\local.ps1 setup' first."
    }
    $pythonVersion = (& $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
    if ($pythonVersion -ne "3.11") {
        throw ".venv-minionerec uses Python $pythonVersion; Python 3.11 is required."
    }
    return $venvPython
}

function Initialize-Environment {
    if (-not (Test-Path -LiteralPath $envPath)) {
        Copy-Item -LiteralPath $envExamplePath -Destination $envPath
        Write-Output "Created .env from .env.example."
    }

    $venvPython = Get-VenvPython
    if (-not (Test-Path -LiteralPath $venvPython)) {
        Write-Output "Creating Python 3.11 environment at $venvPath"
        & $PythonCommand @PythonArgs -m venv $venvPath
        if ($LASTEXITCODE -ne 0) {
            throw "Python 3.11 virtual-environment creation failed."
        }
        $venvPython = Assert-PythonEnvironment
    }
    else {
        $venvPython = Assert-PythonEnvironment
    }

    $torchLock = Join-Path $pythonRoot "requirements-torch-cu128.lock.txt"
    $requirementsLock = Join-Path $pythonRoot "requirements.lock.txt"
    foreach ($lock in @($torchLock, $requirementsLock)) {
        if (-not (Test-Path -LiteralPath $lock)) {
            throw "Missing dependency lock: $lock"
        }
    }

    Write-Output "Installing the hashed CUDA 12.8 PyTorch wheel."
    & $venvPython -m pip install `
        --disable-pip-version-check `
        --no-deps `
        --require-hashes `
        -r $torchLock
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA PyTorch installation failed."
    }

    Write-Output "Installing the unified application, model, training, and verification lock."
    & $venvPython -m pip install `
        --disable-pip-version-check `
        --no-deps `
        --require-hashes `
        -r $requirementsLock
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed."
    }

    & $venvPython -m pip check
    if ($LASTEXITCODE -ne 0) {
        throw "Installed dependencies are inconsistent."
    }

    & $venvPython -c "import spacy; spacy.load('en_core_web_sm')" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Output "Installing the Presidio English language model."
        & $venvPython -m spacy download en_core_web_sm
        if ($LASTEXITCODE -ne 0) {
            throw "Presidio language-model installation failed."
        }
    }

    if ($RunTests) {
        Push-Location $pythonRoot
        try {
            & $venvPython -m pytest -q
            if ($LASTEXITCODE -ne 0) {
                throw "Test suite failed."
            }
        }
        finally {
            Pop-Location
        }
    }

    $backend = Get-DotEnvValue -Name "LLM_BACKEND" -Default "deepseek"
    $keyConfigured = [bool](Get-DotEnvValue -Name "DEEPSEEK_API_KEY")
    Write-Output "Environment ready: Python 3.11, unified MiniOneRec profile."
    Write-Output "LLM backend: $backend"
    Write-Output "DeepSeek key configured: $keyConfigured"
}

function Test-ModelArtifact {
    $modelEnabled = (Get-DotEnvValue -Name "MINIONEREC_ENABLED" -Default "true") -ne "false"
    $ranker = Get-DotEnvValue -Name "RECOMMENDATION_RANKER" -Default "auto"
    if (-not $modelEnabled -or $ranker -eq "rule_v1") {
        Write-Output "Model artifact check skipped for the rule rollback configuration."
        return
    }

    $basePath = Join-Path $pythonRoot "artifacts\base-models\qwen2.5-0.5b"
    $artifactPath = Join-Path $pythonRoot "artifacts\minionerec-mvp\v1"
    if (-not (Test-Path -LiteralPath (Join-Path $basePath "base_model_manifest.json"))) {
        throw "MiniOneRec base-model snapshot is missing."
    }
    if (-not (Test-Path -LiteralPath (Join-Path $artifactPath "model_manifest.json"))) {
        throw "MiniOneRec v1 artifact is missing."
    }
    if ($SkipModelCheck) {
        Write-Output "Model files found; offline smoke was skipped."
        return
    }

    $venvPython = Assert-PythonEnvironment
    Push-Location $pythonRoot
    try {
        $env:HF_HUB_OFFLINE = "1"
        $env:TRANSFORMERS_OFFLINE = "1"
        & $venvPython recommendation_model\smoke_inference.py `
            --config recommendation_model\config.yaml `
            --artifact artifacts\minionerec-mvp\v1 `
            --device auto
        if ($LASTEXITCODE -ne 0) {
            throw "MiniOneRec offline model check failed."
        }
    }
    finally {
        Pop-Location
    }
}

function Start-Infrastructure {
    Push-Location $pythonRoot
    try {
        docker info *> $null
        if ($LASTEXITCODE -ne 0) {
            throw "Docker is unavailable. Start Docker Desktop and retry."
        }

        & (Join-Path $PSScriptRoot "build-hapi-local.ps1")
        docker compose up -d postgres neo4j redis fhir
        if ($LASTEXITCODE -ne 0) {
            throw "Local data services could not be started."
        }

        $deadline = [DateTime]::UtcNow.AddSeconds($InfrastructureTimeoutSeconds)
        do {
            $postgres = docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' medigen-postgres 2>$null
            $neo4j = docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' medigen-neo4j 2>$null
            $redis = docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' medigen-redis 2>$null
            $fhirReady = $false
            try {
                $metadata = Invoke-RestMethod -Uri "http://127.0.0.1:8080/fhir/metadata" -TimeoutSec 5
                $fhirReady = $metadata.resourceType -eq "CapabilityStatement"
            }
            catch {
                $fhirReady = $false
            }
            $ready = (
                $postgres -eq "healthy" -and
                $neo4j -eq "healthy" -and
                $redis -eq "healthy" -and
                $fhirReady
            )
            if (-not $ready) {
                Start-Sleep -Seconds 2
            }
        } while (-not $ready -and [DateTime]::UtcNow -lt $deadline)

        if (-not $ready) {
            docker compose ps
            throw "Local services did not become ready within $InfrastructureTimeoutSeconds seconds."
        }
        Write-Output "PostgreSQL, Neo4j, Redis, and HAPI FHIR are ready."
    }
    finally {
        Pop-Location
    }
}

function Assert-DeepSeekConfiguration {
    if (-not (Test-Path -LiteralPath $envPath)) {
        throw "Missing .env. Run '.\scripts\local.ps1 setup' first."
    }
    $backend = Get-DotEnvValue -Name "LLM_BACKEND" -Default "deepseek"
    if ($backend -eq "deepseek" -and -not (Get-DotEnvValue -Name "DEEPSEEK_API_KEY")) {
        throw "DEEPSEEK_API_KEY is empty in .env."
    }
}

function Start-Api {
    $venvPython = Assert-PythonEnvironment
    Assert-DeepSeekConfiguration

    $baseUrl = "http://${HostAddress}:${Port}"
    $existingHealth = $null
    try {
        $existingHealth = Invoke-RestMethod -Uri "$baseUrl/health" -TimeoutSec 2
    }
    catch {
        $existingHealth = $null
    }
    if ($null -ne $existingHealth -and $existingHealth.service -eq "medigen-mvp") {
        $existingReadiness = Invoke-RestMethod -Uri "$baseUrl/ready" -TimeoutSec 10
        if ($existingReadiness.status -eq "ready") {
            Write-Output "MediGen is already ready at $baseUrl"
            return
        }
        throw "MediGen is running on port $Port with incomplete dependencies."
    }

    $listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    if ($listeners.Count -gt 0) {
        throw "Port $Port is already in use by another process."
    }

    New-Item -ItemType Directory -Path $runtimePath -Force | Out-Null
    $stdoutPath = Join-Path $runtimePath "uvicorn-${Port}.stdout.log"
    $stderrPath = Join-Path $runtimePath "uvicorn-${Port}.stderr.log"
    $apiProcess = Start-Process `
        -FilePath $venvPython `
        -ArgumentList '-m', 'uvicorn', 'src.api.main:app', '--host', $HostAddress, '--port', $Port `
        -WorkingDirectory $pythonRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru

    $apiProcess.Id |
        Set-Content -LiteralPath (Join-Path $runtimePath "uvicorn-${Port}.pid") -Encoding ascii

    $ready = $false
    $lastReadiness = $null
    for ($attempt = 0; $attempt -lt 180; $attempt++) {
        Start-Sleep -Seconds 1
        try {
            $lastReadiness = Invoke-RestMethod -Uri "$baseUrl/ready" -TimeoutSec 5
            if ($lastReadiness.status -eq "ready") {
                $ready = $true
                break
            }
        }
        catch {
            $lastReadiness = $null
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

    Write-Output "MediGen started at $baseUrl"
    Show-Status
}

function Stop-Api {
    $pidPath = Join-Path $runtimePath "uvicorn-${Port}.pid"
    $targetIds = @()
    if (Test-Path -LiteralPath $pidPath) {
        $savedPid = (Get-Content -LiteralPath $pidPath -Raw).Trim()
        if ($savedPid -match "^\d+$") {
            $targetIds += [int]$savedPid
        }
    }

    $listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    $targetIds += $listeners.OwningProcess
    $targetIds = @($targetIds | Where-Object { $_ } | Sort-Object -Unique)

    $targets = @()
    foreach ($targetId in $targetIds) {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId = $targetId" -ErrorAction SilentlyContinue
        if (
            $null -ne $process -and
            $process.Name -eq "python.exe" -and
            $process.CommandLine -like "*uvicorn*src.api.main:app*--port*$Port*"
        ) {
            $targets += $process
        }
    }
    if ($targets.Count -eq 0) {
        Write-Output "No MediGen API process is running on port $Port."
        if (Test-Path -LiteralPath $pidPath) {
            Remove-Item -LiteralPath $pidPath -Force
        }
        return
    }
    foreach ($target in $targets) {
        Stop-Process -Id $target.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Milliseconds 500
    $remaining = @($targets | Where-Object {
        $null -ne (Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue)
    })
    if ($remaining.Count -gt 0) {
        throw "Some MediGen processes could not be stopped: $($remaining.ProcessId -join ', ')"
    }
    if (Test-Path -LiteralPath $pidPath) {
        Remove-Item -LiteralPath $pidPath -Force
    }
    Write-Output "MediGen stopped on port $Port."
}

function Show-Status {
    $baseUrl = "http://${HostAddress}:${Port}"
    try {
        $readiness = Invoke-RestMethod -Uri "$baseUrl/ready" -TimeoutSec 10
    }
    catch {
        throw "MediGen readiness is unavailable at $baseUrl/ready."
    }

    $model = $readiness.recommendation_model
    Write-Output "Status: $($readiness.status)"
    Write-Output "Web UI: $baseUrl/"
    Write-Output "Swagger UI: $baseUrl/docs"
    Write-Output "LLM backend: $($readiness.llm_backend)"
    Write-Output "DeepSeek configured: $($readiness.deepseek_configured)"
    Write-Output "MiniOneRec artifact valid: $($model.artifact_valid)"
    Write-Output "MiniOneRec loaded: $($model.loaded)"
    Write-Output "Ranking mode: $($model.configured_strategy)"
    Write-Output "Model version: $($model.model_version)"
    Write-Output "Fallback available: $($model.fallback_available)"
    Write-Output "Dependencies: $($readiness.dependencies | ConvertTo-Json -Compress)"
}

switch ($Action) {
    "setup" {
        Initialize-Environment
    }
    "deploy" {
        if (-not $SkipInstall) {
            Initialize-Environment
        }
        else {
            [void](Assert-PythonEnvironment)
            Assert-DeepSeekConfiguration
        }
        Test-ModelArtifact
        Start-Infrastructure
        Start-Api
    }
    "infra" {
        Start-Infrastructure
    }
    "start" {
        Start-Api
    }
    "stop" {
        Stop-Api
    }
    "status" {
        Show-Status
    }
}
