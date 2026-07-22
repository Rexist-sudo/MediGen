[CmdletBinding()]
param(
    [string]$PythonCommand = "python",
    [string]$HostAddress = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 8001,
    [switch]$WithTests,
    [switch]$SkipInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $SkipInstall) {
    & (Join-Path $PSScriptRoot "setup-local.ps1") `
        -PythonCommand $PythonCommand `
        -WithTests:$WithTests
}

& (Join-Path $PSScriptRoot "start-infrastructure.ps1")

& (Join-Path $PSScriptRoot "start-local.ps1") `
    -HostAddress $HostAddress `
    -Port $Port

Write-Output "Local deployment command completed."
