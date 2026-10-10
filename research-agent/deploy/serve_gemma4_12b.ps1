# ============================================================================
# serve_gemma4_12b.ps1 — GPU-resident 12B-class LLM serving inside 8 GiB VRAM
# Target: NVIDIA RTX 4060 Laptop GPU (Windows, CUDA build of llama.cpp)
# Model : Gemma 4 12B QAT — Unsloth UD-Q4_K_XL (6405 MiB) + MTP drafter
# Goal  : >=30 tok/s, 32K context, ZERO shared-memory (sysmem) fallback.
#
# ONE-TIME HOST SETUP (this is what kills the silent 4-5 tok/s PCIe spill):
#   NVIDIA Control Panel -> Manage 3D Settings -> Global Settings ->
#   "CUDA - Sysmem Fallback Policy" -> "Prefer No Sysmem Fallback"
#   (Allocations now fail loudly instead of spilling into Shared System RAM.)
#   Also recommended: displays driven by iGPU (hybrid mode) to reclaim
#   300-500 MiB, and Settings -> Display -> Graphics -> llama-server.exe =
#   High Performance.
#
# Usage (PowerShell):
#   .\serve_gemma4_12b.ps1                          # balanced (32K, MTP on)
#   $env:FIT_MODE="safe";  .\serve_gemma4_12b.ps1   # 16K, max headroom
#   $env:FIT_MODE="max";   .\serve_gemma4_12b.ps1   # 64K (iGPU-display only)
#   $env:ENABLE_MTP="0";   .\serve_gemma4_12b.ps1   # no speculation
# ============================================================================
$ErrorActionPreference = "Stop"

# ---------------------------- configuration -------------------------------
$FitMode        = if ($env:FIT_MODE)       { $env:FIT_MODE }       else { "balanced" }
$EnableMtp      = if ($env:ENABLE_MTP)     { $env:ENABLE_MTP }     else { "1" }
$MtpQuant       = if ($env:MTP_QUANT)      { $env:MTP_QUANT }      else { "Q4_0" }
$EnableThinking = if ($env:ENABLE_THINKING){ $env:ENABLE_THINKING }else { "true" }
$HostAddr       = if ($env:HOST_ADDR)      { $env:HOST_ADDR }      else { "127.0.0.1" }
$Port           = if ($env:PORT)           { $env:PORT }           else { "8080" }
$ModelDir       = if ($env:MODEL_DIR)      { $env:MODEL_DIR }      else { "$HOME\models\gemma-4-12b-qat" }
$LlamaServer    = if ($env:LLAMA_SERVER)   { $env:LLAMA_SERVER }   else { "llama-server.exe" }
$HfRepo         = "unsloth/gemma-4-12b-it-qat-GGUF"
$ModelFile      = "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"
$MtpRel         = "MTP/mtp-gemma-4-12B-it-$MtpQuant.gguf"

switch ($FitMode) {
  "safe"     { $Ctx = 16384; $Nb = 256 }
  "balanced" { $Ctx = 32768; $Nb = 512 }
  "max"      { $Ctx = 65536; $Nb = 512 }
  default    { throw "FIT_MODE must be safe|balanced|max" }
}

# ------------------- zero-system-RAM enforcement (env) --------------------
$env:CUDA_VISIBLE_DEVICES   = if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { "0" }
$env:GGML_CUDA_NO_PINNED    = "1"      # no large pinned host allocations
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"

# --------------------------- fetch model files ----------------------------
New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
$ModelPath = Join-Path $ModelDir $ModelFile
$MtpPath   = Join-Path $ModelDir ($MtpRel -replace '/', '\')
if (-not (Test-Path $ModelPath) -or ($EnableMtp -eq "1" -and -not (Test-Path $MtpPath))) {
  Write-Host "[setup] downloading model + MTP drafter from $HfRepo (~6.7 GB) ..."
  if (-not (Get-Command hf -ErrorAction SilentlyContinue)) {
    pip install -q -U "huggingface_hub[cli]" hf_transfer
  }
  hf download $HfRepo $ModelFile "MTP/mtp-gemma-4-12B-it-$MtpQuant.gguf" `
              "MTP/mtp-gemma-4-12B-it-Q4_0.gguf" --local-dir $ModelDir
}

# ------------------------------- preflight --------------------------------
Write-Host "[preflight] engine : $LlamaServer"
Write-Host "[preflight] profile: FIT_MODE=$FitMode ctx=$Ctx n-batch=$Nb MTP=$EnableMtp($MtpQuant)"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
  $freeMiB = [int](nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | Select-Object -First 1)
  $needMiB = 6405 + 242 + 900 + [int]($Ctx / 128)
  if ($freeMiB -lt $needMiB) {
    Write-Warning "Free VRAM $freeMiB MiB < estimated need $needMiB MiB. Close GPU apps or use FIT_MODE=safe."
    Write-Warning "With 'Prefer No Sysmem Fallback' set, overflow = hard OOM (good), NOT 4 tok/s spill (bad)."
  }
} else { Write-Warning "nvidia-smi not found; cannot verify VRAM headroom." }

# ---------------------------- assemble command ----------------------------
$Args = @(
  "--model",   $ModelPath,
  "-ngl",      "99",                          # ALL tensors on GPU - never split layers
  "--flash-attn", "on",
  "--ctx-size", "$Ctx",
  "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",  # -47% KV, ~0% decode cost <=32K
  "--n-batch", "$Nb", "--ubatch-size", "$Nb",
  "--parallel", "1",
  "--no-mmap",                                # full VRAM residency
  "--cache-ram", "0",                         # disable host-RAM prompt cache
  "--no-warmup",
  "--jinja",
  "--temp", "1.0", "--top-p", "0.95", "--top-k", "64",   # official Gemma 4 QAT settings
  "--chat-template-kwargs", ('{\"enable_thinking\":' + $EnableThinking + '}'),
  "--alias", "gemma-4-12b-qat-gpuonly",
  "--host", $HostAddr, "--port", "$Port"
)
if ($EnableMtp -eq "1" -and (Test-Path $MtpPath)) {
  # Lossless MTP speculation (llama.cpp PR #23398): 1.4-2.2x decode speed.
  # SPEC_MIX=1 adds draftless ngram-mod — A/B via test harness first (issue #24266
  # showed ngram+MTP conflicts on some older builds).
  $SpecTypes = "draft-mtp"
  if ($env:SPEC_MIX -eq "1") { $SpecTypes = "draft-mtp,ngram-mod" }
  $DraftN = if ($env:SPEC_DRAFT_N_MAX) { $env:SPEC_DRAFT_N_MAX } else { "2" }
  $Args += @("-md", $MtpPath, "--spec-type", $SpecTypes, "--spec-draft-n-max", "$DraftN",
             "--spec-draft-ngl", "all",       # drafter 100% in VRAM (isolation)
             "-ctkd", "q8_0", "-ctvd", "q8_0") # drafter KV quantized too
}
$Args += @("--metrics")   # Prometheus telemetry: KV usage + speculative acceptance

Write-Host "[launch] $LlamaServer $($Args -join ' ')"
& $LlamaServer @Args
