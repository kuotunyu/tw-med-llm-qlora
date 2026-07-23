param(
    [Parameter(Mandatory = $true)]
    [string]$ExportDirectory,
    [string]$ModelName = "tw-med-taide-12b-q4-k-m",
    [string]$ExpectedBaseModelId = "taide/Gemma-3-TAIDE-12b-Chat-2602",
    [string]$ExpectedBaseModelRevision = "4de0b93b99f8b61b59c40d019fd593bdd1c42249",
    [string]$ExpectedPhase3ArchiveSha256 = "2c537dfd3049319286c678a3ca3aa72e3f20baa7e0f44bde93ff7ee4dc47e43e",
    [int]$ExpectedAdapterCheckpoint = 700,
    [double]$ApprovedComputeUnitLimit = 6.36
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-TextSha256 {
    param([Parameter(Mandatory = $true)][string]$Text)

    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
        $digest = $sha.ComputeHash($bytes)
        return ([System.BitConverter]::ToString($digest)).Replace("-", "").ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
}

function Get-FileSha256 {
    param([Parameter(Mandatory = $true)][string]$Path)

    $resolvedPath = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).Path
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $stream = [System.IO.File]::OpenRead($resolvedPath)
    try {
        $digest = $sha.ComputeHash($stream)
        return ([System.BitConverter]::ToString($digest)).Replace("-", "").ToLowerInvariant()
    } finally {
        $stream.Dispose()
        $sha.Dispose()
    }
}

function Get-ReceiptFileRecord {
    param(
        [Parameter(Mandatory = $true)]$Receipt,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $property = @($Receipt.files.PSObject.Properties | Where-Object { $_.Name -eq $Name })
    if ($property.Count -ne 1) {
        throw "Export receipt must contain exactly one record for $Name."
    }
    return $property[0].Value
}

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "ollama was not found. Install the official Windows build first."
}

$exportPath = (Resolve-Path -LiteralPath $ExportDirectory -ErrorAction Stop).Path
$exportReceiptPath = Join-Path $exportPath "gguf-export-receipt.json"
if (-not (Test-Path -LiteralPath $exportReceiptPath -PathType Leaf)) {
    throw "gguf-export-receipt.json was not found in the export directory."
}
$exportReceipt = Get-Content -LiteralPath $exportReceiptPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ([int]$exportReceipt.schema_version -lt 3) {
    throw "Export receipt schema_version must be at least 3."
}
if ($exportReceipt.optional_export -ne "gguf_q4_k_m") {
    throw "Export receipt does not describe the approved GGUF Q4_K_M workflow."
}
if ($exportReceipt.quantization_method -ne "q4_k_m") {
    throw "Export receipt quantization method is not q4_k_m."
}
if ($exportReceipt.base_model_id -ne $ExpectedBaseModelId) {
    throw "Export receipt base model ID mismatch."
}
if ($exportReceipt.base_model_revision -ne $ExpectedBaseModelRevision) {
    throw "Export receipt base model revision mismatch."
}
if ($exportReceipt.phase3_archive_sha256 -ne $ExpectedPhase3ArchiveSha256) {
    throw "Export receipt Phase 3 archive SHA-256 mismatch."
}
if ([int]$exportReceipt.adapter_checkpoint -ne $ExpectedAdapterCheckpoint) {
    throw "Export receipt adapter checkpoint mismatch."
}
if ([double]$exportReceipt.approval.approved_compute_units_with_20pct_buffer -ne $ApprovedComputeUnitLimit) {
    throw "Export receipt compute-unit approval limit mismatch."
}
if ([bool]$exportReceipt.published -or [bool]$exportReceipt.external_upload_performed) {
    throw "Export receipt indicates an external publication or upload."
}
if (-not [bool]$exportReceipt.adapter_merge.peft_detected) {
    throw "Export receipt does not attest that Unsloth detected a PEFT model."
}
if (-not [bool]$exportReceipt.adapter_merge.runtime_base_model_rebound_to_verified_snapshot) {
    throw "Export receipt does not attest the verified local base-model binding."
}
if ([int]$exportReceipt.adapter_merge.lora_parameter_tensors -le 0) {
    throw "Export receipt has no LoRA parameter evidence."
}
if ($exportReceipt.gguf.ollama_import_mode -ne "text_only_primary_gguf") {
    throw "Export receipt does not describe the approved text-only Ollama import mode."
}

$modelfile = Join-Path $exportPath "Modelfile"
if (-not (Test-Path -LiteralPath $modelfile -PathType Leaf)) {
    throw "Modelfile was not found in the export directory."
}
$ggufFiles = @(Get-ChildItem -LiteralPath $exportPath -Filter "*.gguf" -File)
$primaryGgufName = [string]$exportReceipt.gguf.primary_file
if ([System.IO.Path]::GetFileName($primaryGgufName) -ne $primaryGgufName) {
    throw "Export receipt primary GGUF must be a plain file name."
}
$projectorGgufNames = @(
    $exportReceipt.gguf.projector_files |
        ForEach-Object { [string]$_ }
)
foreach ($projectorName in $projectorGgufNames) {
    if ([System.IO.Path]::GetFileName($projectorName) -ne $projectorName) {
        throw "Export receipt projector GGUF must be a plain file name."
    }
}
$expectedGgufNames = @($primaryGgufName) + $projectorGgufNames
$actualGgufNames = @($ggufFiles | ForEach-Object { $_.Name })
$ggufNameDifference = @(
    Compare-Object -ReferenceObject $expectedGgufNames -DifferenceObject $actualGgufNames
)
if ($ggufNameDifference.Count -ne 0 -or $ggufFiles.Count -ne $expectedGgufNames.Count) {
    throw "GGUF files do not match the export receipt."
}
if ([bool]$exportReceipt.model_snapshot.vlm_processor_required) {
    if ($projectorGgufNames.Count -ne 1 -or -not [bool]$exportReceipt.gguf.vlm_projector_archived) {
        throw "VLM export receipt must include exactly one archived projector GGUF."
    }
}
$primaryGguf = @($ggufFiles | Where-Object { $_.Name -eq $primaryGgufName })
if ($primaryGguf.Count -ne 1) {
    throw "Expected exactly one primary GGUF named by the export receipt."
}
$ggufSha256 = $null
foreach ($ggufFile in $ggufFiles) {
    $ggufRecord = Get-ReceiptFileRecord -Receipt $exportReceipt -Name $ggufFile.Name
    $currentSha256 = Get-FileSha256 -Path $ggufFile.FullName
    if ([int64]$ggufRecord.bytes -ne [int64]$ggufFile.Length) {
        throw "GGUF byte size does not match the export receipt: $($ggufFile.Name)"
    }
    if ([string]$ggufRecord.sha256 -ne $currentSha256) {
        throw "GGUF SHA-256 does not match the export receipt: $($ggufFile.Name)"
    }
    if ($ggufFile.Name -eq $primaryGgufName) {
        $ggufSha256 = $currentSha256
    }
}
$modelfileRecord = Get-ReceiptFileRecord -Receipt $exportReceipt -Name "Modelfile"
$modelfileInfo = Get-Item -LiteralPath $modelfile
$modelfileSha256 = Get-FileSha256 -Path $modelfile
if ([int64]$modelfileRecord.bytes -ne [int64]$modelfileInfo.Length) {
    throw "Modelfile byte size does not match the export receipt."
}
if ([string]$modelfileRecord.sha256 -ne $modelfileSha256) {
    throw "Modelfile SHA-256 does not match the export receipt."
}
$fromLine = "FROM ./" + $primaryGguf[0].Name
if (-not (Select-String -LiteralPath $modelfile -SimpleMatch $fromLine -Quiet)) {
    throw "Modelfile does not reference the primary GGUF file."
}

$ollamaVersionLines = @(& ollama --version 2>&1)
$ollamaVersion = ($ollamaVersionLines -join "`n").Trim()
& ollama list *> $null
if ($LASTEXITCODE -ne 0) {
    throw "The Ollama service is not running. Start the Windows Ollama app first."
}

Write-Host "[1/3] Creating the local Ollama model"
& ollama create $ModelName -f $modelfile
if ($LASTEXITCODE -ne 0) { throw "ollama create failed." }

Write-Host "[2/3] Inspecting the imported model"
$ollamaModelfileLines = @(& ollama show $ModelName --modelfile)
if ($LASTEXITCODE -ne 0) { throw "ollama show --modelfile failed." }
$ollamaModelfile = ($ollamaModelfileLines -join "`n").Trim()

Write-Host "[3/3] Running the fixed A-D acceptance probe"
$prompt = "Answer with only one uppercase letter. Which is the Celsius unit symbol? A. kg B. mL C. degrees C D. mmHg"
$started = Get-Date
$lines = @(& ollama run $ModelName $prompt)
$elapsed = ((Get-Date) - $started).TotalSeconds
if ($LASTEXITCODE -ne 0) { throw "ollama run failed." }
$response = ($lines -join "`n").Trim()
$passed = $response -match "^\s*C\s*[.]?\s*$"

$processLines = @(& ollama ps)
if ($LASTEXITCODE -ne 0) { throw "ollama ps failed." }
$processText = ($processLines -join "`n").Trim()
$gpuFullyLoaded = ($processText -match [regex]::Escape($ModelName) -and $processText -match "(?i)100%\s+GPU")
if (-not $gpuFullyLoaded) {
    throw "Ollama did not report the accepted model as 100% GPU."
}

$receipt = [ordered]@{
    schema_version = 2
    phase = 5
    optional_runtime = "ollama_gguf"
    created_at_utc = [DateTime]::UtcNow.ToString("o")
    model_name = $ModelName
    ollama_version = $ollamaVersion
    base_model_id = $ExpectedBaseModelId
    base_model_revision = $ExpectedBaseModelRevision
    adapter_checkpoint = $ExpectedAdapterCheckpoint
    phase3_archive_sha256 = $ExpectedPhase3ArchiveSha256
    quantization_method = "q4_k_m"
    export_receipt_sha256 = Get-FileSha256 -Path $exportReceiptPath
    export_receipt_schema_version = [int]$exportReceipt.schema_version
    gguf_file = $primaryGguf[0].Name
    gguf_bytes = [int64]$primaryGguf[0].Length
    gguf_sha256 = $ggufSha256
    projector_files = $projectorGgufNames
    projector_count = $projectorGgufNames.Count
    ollama_import_mode = "text_only_primary_gguf"
    modelfile_sha256 = $modelfileSha256
    imported_modelfile_sha256 = Get-TextSha256 -Text $ollamaModelfile
    ollama_ps_sha256 = Get-TextSha256 -Text $processText
    gpu_fully_loaded = $gpuFullyLoaded
    probe = "synthetic_unit_mcq_v1"
    expected_answer = "C"
    output_sha256 = Get-TextSha256 -Text $response
    total_seconds = $elapsed
    passed = $passed
    raw_output_recorded = $false
    imported_modelfile_recorded = $false
    ollama_ps_recorded = $false
    external_upload_performed = $false
}
$receiptPath = Join-Path $exportPath "ollama-acceptance.json"
$receiptJson = ($receipt | ConvertTo-Json -Depth 5).Replace("`r`n", "`n")
$utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($receiptPath, $receiptJson + "`n", $utf8WithoutBom)
Write-Host $receiptJson
if (-not $passed) { throw "Ollama acceptance answer was not the expected standalone C." }
Write-Host "Optional Ollama acceptance passed: $receiptPath"
