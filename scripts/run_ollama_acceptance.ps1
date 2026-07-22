param(
    [Parameter(Mandatory = $true)]
    [string]$ExportDirectory,
    [string]$ModelName = "tw-med-taide-12b"
)

$ErrorActionPreference = "Stop"
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "ollama was not found. Install the official Windows build first."
}

$exportPath = (Resolve-Path -LiteralPath $ExportDirectory -ErrorAction Stop).Path
$modelfile = Join-Path $exportPath "Modelfile"
if (-not (Test-Path -LiteralPath $modelfile -PathType Leaf)) {
    throw "Modelfile was not found in the export directory."
}
$ggufFiles = @(Get-ChildItem -LiteralPath $exportPath -Filter "*.gguf" -File)
if ($ggufFiles.Count -ne 1) {
    throw "Expected exactly one GGUF file in the export directory."
}
$fromLine = "FROM ./" + $ggufFiles[0].Name
if (-not (Select-String -LiteralPath $modelfile -SimpleMatch $fromLine -Quiet)) {
    throw "Modelfile does not reference the only GGUF file."
}

Write-Host "[1/2] Creating the local Ollama model"
& ollama create $ModelName -f $modelfile
if ($LASTEXITCODE -ne 0) { throw "ollama create failed." }

Write-Host "[2/2] Running the fixed A-D acceptance probe"
$prompt = "Answer with only one uppercase letter. Which is the Celsius unit symbol? A. kg B. mL C. degrees C D. mmHg"
$started = Get-Date
$lines = @(& ollama run $ModelName $prompt)
$elapsed = ((Get-Date) - $started).TotalSeconds
if ($LASTEXITCODE -ne 0) { throw "ollama run failed." }
$response = ($lines -join "`n").Trim()
$passed = $response -match "^\s*C\s*[.]?\s*$"

$sha = [System.Security.Cryptography.SHA256]::Create()
try {
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($response)
    $digest = $sha.ComputeHash($bytes)
    $outputHash = ([System.BitConverter]::ToString($digest)).Replace("-", "").ToLowerInvariant()
} finally {
    $sha.Dispose()
}

$receipt = [ordered]@{
    schema_version = 1
    phase = 5
    optional_runtime = "ollama_gguf"
    model_name = $ModelName
    gguf_file = $ggufFiles[0].Name
    gguf_sha256 = (Get-FileHash -LiteralPath $ggufFiles[0].FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    probe = "synthetic_unit_mcq_v1"
    expected_answer = "C"
    output_sha256 = $outputHash
    total_seconds = $elapsed
    passed = $passed
    raw_output_recorded = $false
}
$receiptPath = Join-Path $exportPath "ollama-acceptance.json"
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
Write-Host ($receipt | ConvertTo-Json -Depth 5)
if (-not $passed) { throw "Ollama acceptance answer was not the expected standalone C." }
Write-Host "Optional Ollama acceptance passed: $receiptPath"
