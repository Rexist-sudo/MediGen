[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$pythonRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$projectRoot = (Resolve-Path (Join-Path $pythonRoot "..")).Path
$runtimeRoot = Join-Path $pythonRoot ".runtime"
$mavenVersion = "3.9.12"
$mavenRoot = Join-Path $runtimeRoot "apache-maven-$mavenVersion"
$mavenCommand = Join-Path $mavenRoot "bin\mvn.cmd"
$mavenArchive = Join-Path $runtimeRoot "apache-maven-$mavenVersion-bin.zip"
$mavenBaseUrl = "https://archive.apache.org/dist/maven/maven-3/$mavenVersion/binaries/apache-maven-$mavenVersion-bin.zip"
$sourceRoot = Join-Path $runtimeRoot "hapi-fhir-jpaserver-starter"
$sourceTag = "image/v8.10.0-3"
$sourceCommit = "4d12a3fe9e17fd19b7b446cf8c99c2f1632602eb"
$imageName = "medigen/hapi-fhir:v8.10.0-3"
$dockerfile = Join-Path $projectRoot "docker\hapi-local.Dockerfile"

$imagePresent = @(
    docker image ls --format '{{.Repository}}:{{.Tag}}' |
        Where-Object { $_ -eq $imageName }
).Count -gt 0
if ($imagePresent) {
    Write-Output "Local HAPI FHIR image is already available: $imageName"
    return
}

New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null
if (-not (Test-Path -LiteralPath $mavenCommand)) {
    Write-Output "Downloading Apache Maven $mavenVersion."
    & curl.exe -L --fail --retry 5 --retry-delay 2 -C - -o $mavenArchive $mavenBaseUrl
    if ($LASTEXITCODE -ne 0) {
        throw "Apache Maven download failed."
    }
    & curl.exe -L --fail --retry 5 --retry-delay 2 -o "$mavenArchive.sha512" "$mavenBaseUrl.sha512"
    if ($LASTEXITCODE -ne 0) {
        throw "Apache Maven checksum download failed."
    }
    $expectedHash = ((Get-Content -LiteralPath "$mavenArchive.sha512" -Raw).Trim() -split '\s+')[0]
    $actualHash = (Get-FileHash -LiteralPath $mavenArchive -Algorithm SHA512).Hash
    if ($actualHash -ne $expectedHash) {
        throw "Apache Maven archive checksum mismatch."
    }
    Expand-Archive -LiteralPath $mavenArchive -DestinationPath $runtimeRoot -Force
}

if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot ".git"))) {
    Write-Output "Cloning HAPI FHIR JPA starter $sourceTag."
    git clone --depth 1 --branch $sourceTag `
        https://github.com/hapifhir/hapi-fhir-jpaserver-starter.git `
        $sourceRoot
    if ($LASTEXITCODE -ne 0) {
        throw "HAPI FHIR source checkout failed."
    }
}
$actualCommit = (git -C $sourceRoot rev-parse HEAD).Trim()
if ($actualCommit -ne $sourceCommit) {
    throw "HAPI FHIR source commit mismatch: $actualCommit"
}

Write-Output "Building the official HAPI FHIR starter source."
Push-Location $sourceRoot
try {
    & $mavenCommand -ntp clean install "-DskipTests" "-Djdk.lang.Process.launchMechanism=vfork"
    if ($LASTEXITCODE -ne 0) {
        throw "HAPI FHIR Maven build failed."
    }
    & $mavenCommand -ntp package "-DskipTests" "spring-boot:repackage" "-Pboot"
    if ($LASTEXITCODE -ne 0) {
        throw "HAPI FHIR Spring Boot package build failed."
    }
}
finally {
    Pop-Location
}

docker build --file $dockerfile --tag $imageName $sourceRoot
if ($LASTEXITCODE -ne 0) {
    throw "HAPI FHIR local image build failed."
}
Write-Output "Built local HAPI FHIR image: $imageName"
