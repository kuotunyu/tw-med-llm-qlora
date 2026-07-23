param(
    [Parameter(Mandatory = $true)]
    [string]$ExportDirectory,
    [string]$ModelName = "tw-med-taide-12b-q4-k-m-vlm",
    [string]$ExpectedBaseModelId = "taide/Gemma-3-TAIDE-12b-Chat-2602",
    [string]$ExpectedBaseModelRevision = "4de0b93b99f8b61b59c40d019fd593bdd1c42249",
    [int]$ExpectedAdapterCheckpoint = 700,
    [string]$OllamaApiBaseUrl = "http://127.0.0.1:11434"
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

function Assert-ReceiptFile {
    param(
        [Parameter(Mandatory = $true)]$Receipt,
        [Parameter(Mandatory = $true)][string]$Directory,
        [Parameter(Mandatory = $true)][string]$Name
    )

    if ([System.IO.Path]::GetFileName($Name) -ne $Name) {
        throw "Receipt file names must not contain a path: $Name"
    }
    $path = Join-Path $Directory $Name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Receipt file is missing: $Name"
    }
    $record = Get-ReceiptFileRecord -Receipt $Receipt -Name $Name
    $item = Get-Item -LiteralPath $path
    if ([int64]$record.bytes -ne [int64]$item.Length) {
        throw "Receipt byte size mismatch: $Name"
    }
    $sha256 = Get-FileSha256 -Path $path
    if ([string]$record.sha256 -ne $sha256) {
        throw "Receipt SHA-256 mismatch: $Name"
    }
    return [ordered]@{
        path = $path
        bytes = [int64]$item.Length
        sha256 = $sha256
    }
}

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "ollama was not found. Install the official Windows build first."
}

$ollamaApiUri = [Uri]$OllamaApiBaseUrl
$allowedLoopbackHosts = @("127.0.0.1", "localhost", "::1")
if (
    $ollamaApiUri.Scheme -ne "http" -or
    $allowedLoopbackHosts -notcontains $ollamaApiUri.Host
) {
    throw "OllamaApiBaseUrl must use HTTP on a loopback host."
}
$ollamaApiRoot = $OllamaApiBaseUrl.TrimEnd("/")

$exportPath = (Resolve-Path -LiteralPath $ExportDirectory -ErrorAction Stop).Path
$exportReceiptPath = Join-Path $exportPath "gguf-export-receipt.json"
$textAcceptancePath = Join-Path $exportPath "ollama-acceptance.json"
if (-not (Test-Path -LiteralPath $exportReceiptPath -PathType Leaf)) {
    throw "gguf-export-receipt.json was not found in the export directory."
}
if (-not (Test-Path -LiteralPath $textAcceptancePath -PathType Leaf)) {
    throw "The text-only Ollama acceptance receipt must pass before VLM acceptance."
}

$exportReceipt = Get-Content -LiteralPath $exportReceiptPath -Raw -Encoding UTF8 | ConvertFrom-Json
$textAcceptance = Get-Content -LiteralPath $textAcceptancePath -Raw -Encoding UTF8 | ConvertFrom-Json
if ([int]$exportReceipt.schema_version -lt 3) {
    throw "Export receipt schema_version must be at least 3."
}
if ($exportReceipt.optional_export -ne "gguf_q4_k_m") {
    throw "Export receipt does not describe the approved GGUF Q4_K_M workflow."
}
if ($exportReceipt.base_model_id -ne $ExpectedBaseModelId) {
    throw "Export receipt base model ID mismatch."
}
if ($exportReceipt.base_model_revision -ne $ExpectedBaseModelRevision) {
    throw "Export receipt base model revision mismatch."
}
if ([int]$exportReceipt.adapter_checkpoint -ne $ExpectedAdapterCheckpoint) {
    throw "Export receipt adapter checkpoint mismatch."
}
if ([bool]$exportReceipt.published -or [bool]$exportReceipt.external_upload_performed) {
    throw "Export receipt indicates an external publication or upload."
}
if (-not [bool]$exportReceipt.adapter_merge.peft_detected) {
    throw "Export receipt does not attest that Unsloth detected a PEFT model."
}
if ([int]$exportReceipt.adapter_merge.lora_parameter_tensors -le 0) {
    throw "Export receipt has no LoRA parameter evidence."
}
if (-not [bool]$textAcceptance.passed -or -not [bool]$textAcceptance.gpu_fully_loaded) {
    throw "The text-only Ollama acceptance receipt did not pass."
}
if ([bool]$textAcceptance.external_upload_performed) {
    throw "The text-only Ollama acceptance receipt indicates an external upload."
}

$primaryName = [string]$exportReceipt.gguf.primary_file
$projectorNames = @($exportReceipt.gguf.projector_files | ForEach-Object { [string]$_ })
if ($projectorNames.Count -ne 1 -or -not [bool]$exportReceipt.gguf.vlm_projector_archived) {
    throw "VLM acceptance requires exactly one archived projector GGUF."
}
$primary = Assert-ReceiptFile -Receipt $exportReceipt -Directory $exportPath -Name $primaryName
$projector = Assert-ReceiptFile -Receipt $exportReceipt -Directory $exportPath -Name $projectorNames[0]
if ([string]$textAcceptance.gguf_sha256 -ne [string]$primary.sha256) {
    throw "Text acceptance and export receipt refer to different primary GGUF files."
}

$ollamaVersionLines = @(& ollama --version 2>&1)
$ollamaVersion = ($ollamaVersionLines -join "`n").Trim()
& ollama list *> $null
if ($LASTEXITCODE -ne 0) {
    throw "The Ollama service is not running. Start the Windows Ollama app first."
}

$temporaryStem = ".vlm-acceptance-" + [Guid]::NewGuid().ToString("N")
$temporaryModelfile = Join-Path $exportPath ($temporaryStem + ".Modelfile")
$fixturePath = Join-Path $exportPath ($temporaryStem + ".png")
$utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
$response = $null
$show = $null
$elapsed = $null
$fixtureSha256 = $null
$answer = $null
$processText = $null

try {
    $modelfileText = @(
        "FROM ./$primaryName"
        "FROM ./$($projectorNames[0])"
        "PARAMETER temperature 0"
        "PARAMETER seed 3407"
        "PARAMETER num_ctx 2048"
        "PARAMETER num_predict 16"
        ""
    ) -join "`n"
    [System.IO.File]::WriteAllText($temporaryModelfile, $modelfileText, $utf8WithoutBom)

    Write-Host "[1/4] Creating the local VLM model from the primary and projector GGUF files"
    & ollama create $ModelName -f $temporaryModelfile
    if ($LASTEXITCODE -ne 0) { throw "ollama create failed." }

    Write-Host "[2/4] Verifying Ollama vision capability and projector metadata"
    $showRequest = @{ model = $ModelName; verbose = $false } | ConvertTo-Json
    $show = Invoke-RestMethod `
        -Uri "$ollamaApiRoot/api/show" `
        -Method Post `
        -ContentType "application/json" `
        -Body $showRequest `
        -TimeoutSec 120
    $capabilities = @($show.capabilities | ForEach-Object { [string]$_ })
    if ($capabilities -notcontains "completion") {
        throw "Ollama did not report the imported model as completion-capable."
    }
    if ($capabilities -notcontains "vision") {
        throw "Ollama did not report the imported model as vision-capable."
    }
    if ([string]$show.projector_info."general.architecture" -ne "clip") {
        throw "Ollama did not report the expected CLIP projector architecture."
    }
    if (([regex]::Matches([string]$show.modelfile, "(?m)^FROM ")).Count -ne 2) {
        throw "Imported VLM Modelfile does not contain exactly two FROM records."
    }

    Write-Host "[3/4] Generating the deterministic red-square fixture"
    Add-Type -AssemblyName System.Drawing
    $bitmap = [System.Drawing.Bitmap]::new(512, 512)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $white = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::White)
    $red = [System.Drawing.SolidBrush]::new(
        [System.Drawing.Color]::FromArgb(255, 220, 20, 60)
    )
    try {
        $graphics.FillRectangle($white, 0, 0, 512, 512)
        $graphics.FillRectangle($red, 96, 96, 320, 320)
        $bitmap.Save($fixturePath, [System.Drawing.Imaging.ImageFormat]::Png)
    } finally {
        $red.Dispose()
        $white.Dispose()
        $graphics.Dispose()
        $bitmap.Dispose()
    }
    $fixtureSha256 = Get-FileSha256 -Path $fixturePath

    Write-Host "[4/4] Running the fixed local vision probe"
    $imageBase64 = [Convert]::ToBase64String([System.IO.File]::ReadAllBytes($fixturePath))
    $chatRequest = @{
        model = $ModelName
        messages = @(
            @{
                role = "user"
                content = (
                    "Inspect the image. What is the color of the large central square? " +
                    "Reply with exactly one uppercase English word and nothing else."
                )
                images = @($imageBase64)
            }
        )
        stream = $false
        options = @{
            temperature = 0
            seed = 3407
            num_predict = 16
        }
        keep_alive = "10m"
    } | ConvertTo-Json -Depth 8 -Compress
    $started = Get-Date
    $response = Invoke-RestMethod `
        -Uri "$ollamaApiRoot/api/chat" `
        -Method Post `
        -ContentType "application/json" `
        -Body $chatRequest `
        -TimeoutSec 600
    $elapsed = ((Get-Date) - $started).TotalSeconds
    $answer = ([string]$response.message.content).Trim()
    $passed = $answer -ceq "RED"

    $processLines = @(& ollama ps)
    if ($LASTEXITCODE -ne 0) { throw "ollama ps failed." }
    $processText = ($processLines -join "`n").Trim()
    $gpuFullyLoaded = (
        $processText -match [regex]::Escape($ModelName) -and
        $processText -match "(?i)100%\s+GPU"
    )
    if (-not $gpuFullyLoaded) {
        throw "Ollama did not report the VLM model as 100% GPU."
    }

    $receipt = [ordered]@{
        schema_version = 1
        phase = 5
        optional_runtime = "ollama_vlm_gguf"
        created_at_utc = [DateTime]::UtcNow.ToString("o")
        model_name = $ModelName
        ollama_version = $ollamaVersion
        base_model_id = $ExpectedBaseModelId
        base_model_revision = $ExpectedBaseModelRevision
        adapter_checkpoint = $ExpectedAdapterCheckpoint
        export_receipt_sha256 = Get-FileSha256 -Path $exportReceiptPath
        text_acceptance_receipt_sha256 = Get-FileSha256 -Path $textAcceptancePath
        gguf_file = $primaryName
        gguf_bytes = [int64]$primary.bytes
        gguf_sha256 = [string]$primary.sha256
        projector_file = $projectorNames[0]
        projector_bytes = [int64]$projector.bytes
        projector_sha256 = [string]$projector.sha256
        capabilities = $capabilities
        projector_architecture = [string]$show.projector_info."general.architecture"
        imported_from_records = 2
        gpu_fully_loaded = $gpuFullyLoaded
        probe = "synthetic_red_square_v1"
        fixture_sha256 = $fixtureSha256
        fixture_recorded = $false
        expected_answer = "RED"
        output_sha256 = Get-TextSha256 -Text $answer
        total_seconds = $elapsed
        api_total_seconds = [double]$response.total_duration / 1e9
        api_load_seconds = [double]$response.load_duration / 1e9
        prompt_eval_count = [int]$response.prompt_eval_count
        eval_count = [int]$response.eval_count
        passed = $passed
        raw_output_recorded = $false
        imported_modelfile_recorded = $false
        ollama_ps_recorded = $false
        external_upload_performed = $false
    }
    $receiptPath = Join-Path $exportPath "ollama-vlm-acceptance.json"
    $receiptJson = ($receipt | ConvertTo-Json -Depth 5).Replace("`r`n", "`n")
    [System.IO.File]::WriteAllText($receiptPath, $receiptJson + "`n", $utf8WithoutBom)
    Write-Host $receiptJson
    if (-not $passed) {
        throw "Ollama VLM acceptance answer was not the expected standalone RED."
    }
    Write-Host "Optional Ollama VLM acceptance passed: $receiptPath"
} finally {
    if (Test-Path -LiteralPath $temporaryModelfile) {
        Remove-Item -LiteralPath $temporaryModelfile -Force
    }
    if (Test-Path -LiteralPath $fixturePath) {
        Remove-Item -LiteralPath $fixturePath -Force
    }
}
