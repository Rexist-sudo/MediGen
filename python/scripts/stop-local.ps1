[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8001
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$pythonRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runtimePath = Join-Path $pythonRoot ".runtime"
$targets = @(
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -eq 'python.exe' -and
            $_.CommandLine -like "*MediGen*uvicorn*--port*$Port*"
        }
)

if ($targets.Count -eq 0) {
    Write-Host "No MediGen Uvicorn process is running on port $Port."
    return
}

foreach ($target in $targets) {
    Stop-Process -Id $target.ProcessId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Milliseconds 500

$remaining = @(
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -eq 'python.exe' -and
            $_.CommandLine -like "*MediGen*uvicorn*--port*$Port*"
        }
)
if ($remaining.Count -gt 0) {
    throw "Some MediGen processes could not be stopped: $($remaining.ProcessId -join ', ')"
}

$pidPath = Join-Path $runtimePath "uvicorn-${Port}.pids"
if (Test-Path -LiteralPath $pidPath) {
    Remove-Item -LiteralPath $pidPath -Force
}
Write-Host "MediGen stopped on port $Port."
