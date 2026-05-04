$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Exe = Join-Path $Root "target\release\bidking-analyzer.exe"

if (-not (Test-Path -LiteralPath $Exe)) {
    Push-Location $Root
    try {
        cargo build --release --bin bidking-analyzer
    } finally {
        Pop-Location
    }
}

Start-Process -FilePath $Exe -WorkingDirectory $Root
