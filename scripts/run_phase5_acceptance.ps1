param(
    [string]$AdapterPath = "",
    [string]$AdapterRepoId = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv was not found. Install uv as documented in the Phase 5 README section."
}

if ($AdapterPath -and $AdapterRepoId) {
    throw "Choose AdapterPath or AdapterRepoId, not both."
}
if ($AdapterPath) {
    $resolvedAdapter = Resolve-Path -LiteralPath $AdapterPath -ErrorAction Stop
    $env:HF_ADAPTER_PATH = $resolvedAdapter.Path
    $env:HF_ADAPTER_REPO_ID = ""
} elseif ($AdapterRepoId) {
    $env:HF_ADAPTER_REPO_ID = $AdapterRepoId
    $env:HF_ADAPTER_PATH = ""
}

Write-Host "[1/3] Installing the locked Windows inference wheels"
uv sync --locked --group inference
if ($LASTEXITCODE -ne 0) { throw "uv sync failed; stopping." }

Write-Host "[2/3] Checking RTX 4090, CUDA, and BF16"
uv run tw-med-local-infer --preflight-only
if ($LASTEXITCODE -ne 0) { throw "4090 preflight failed; model loading was skipped." }

Write-Host "[3/3] Loading the pinned base and adapter, then running the A-D probe"
uv run tw-med-local-infer --acceptance
if ($LASTEXITCODE -ne 0) { throw "Acceptance failed; keep outputs for diagnosis." }

Write-Host "Phase 5 acceptance passed. Return the latest outputs/local_inference JSON."
