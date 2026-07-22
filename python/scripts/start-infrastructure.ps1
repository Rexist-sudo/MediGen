[CmdletBinding()]
param(
    [ValidateRange(30, 900)]
    [int]$TimeoutSeconds = 420
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$pythonRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $pythonRoot
try {
    docker info *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker is not available. Start Docker Desktop and retry."
    }

    & (Join-Path $PSScriptRoot "build-hapi-local.ps1")

    docker compose up -d postgres neo4j redis fhir
    if ($LASTEXITCODE -ne 0) {
        throw "Local data services could not be started."
    }

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    $ready = $false
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
        throw "Local data services did not become ready within $TimeoutSeconds seconds."
    }

    Write-Output "PostgreSQL, Neo4j, Redis, and HAPI FHIR are ready."
}
finally {
    Pop-Location
}
