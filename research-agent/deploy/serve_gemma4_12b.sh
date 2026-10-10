#!/usr/bin/env bash
# ============================================================================
# serve_gemma4_12b.sh — GPU-resident 12B-class LLM serving inside 8 GiB VRAM
# Target: NVIDIA RTX 4060 Laptop GPU (Ada sm_89, 256 GB/s, 8192 MiB)
# Model : Gemma 4 12B QAT — Unsloth UD-Q4_K_XL (6405 MiB) + MTP drafter (242 MiB)
# Goal  : >=30 tok/s, 32K context, ZERO system-RAM fallback.
#
# Usage:
#   ./serve_gemma4_12b.sh                  # balanced profile (32K ctx, MTP on)
#   FIT_MODE=safe  ./serve_gemma4_12b.sh   # 16K ctx — max VRAM headroom
#   FIT_MODE=max   ./serve_gemma4_12b.sh   # 64K ctx — Linux/iGPU-display only
#   ENABLE_MTP=0   ./serve_gemma4_12b.sh   # disable speculative decoding
#   MTP_QUANT=Q8_0 ./serve_gemma4_12b.sh   # higher-acceptance drafter (+200 MiB)
# ============================================================================
set -euo pipefail

# ---------------------------- configuration -------------------------------
FIT_MODE="${FIT_MODE:-balanced}"            # safe | balanced | max
ENABLE_MTP="${ENABLE_MTP:-1}"               # 1 = draft-mtp speculative decoding (lossless)
MTP_QUANT="${MTP_QUANT:-Q4_0}"              # Q4_0 (242 MiB) | Q8_0 (444 MiB)
ENABLE_THINKING="${ENABLE_THINKING:-true}"  # Gemma 4 hybrid thinking mode
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/gemma-4-12b-qat}"
LLAMA_SERVER="${LLAMA_SERVER:-}"            # path to llama-server; empty = auto-detect
HF_REPO="unsloth/gemma-4-12b-it-qat-GGUF"
MODEL_FILE="gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"

# VRAM budget (MiB): weights 6405 + MTP 242 + KV + buffers ~600-900 => <= 7.6 GiB
case "$FIT_MODE" in
  safe)     CTX=16384; NB=256 ;;
  balanced) CTX=32768; NB=512 ;;
  max)      CTX=65536; NB=512 ;;
  *) echo "FIT_MODE must be safe|balanced|max" >&2; exit 2 ;;
esac

# ------------------- zero-system-RAM enforcement (env) --------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"  # single GPU, no surprises
export GGML_CUDA_NO_PINNED=1    # do not allocate large pinned HOST buffers
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# ---------------------------- locate engine -------------------------------
if [[ -z "$LLAMA_SERVER" ]]; then
  for c in ./llama.cpp/build/bin/llama-server ./llama-server llama-server; do
    if command -v "$c" >/dev/null 2>&1 || [[ -x "$c" ]]; then LLAMA_SERVER="$c"; break; fi
  done
fi
if [[ -z "$LLAMA_SERVER" ]]; then
  cat >&2 <<'EOF'
[ERROR] llama-server not found. Build it (CUDA, sm_89) with:
  sudo apt-get install -y build-essential cmake git curl
  git clone https://github.com/ggml-org/llama.cpp
  cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=89
  cmake --build llama.cpp/build --config Release -j --target llama-server llama-bench
The build MUST postdate PR #23398 (MTP support, June 2026). Verify:
  llama-server --help | grep -A2 spec-type     # must list 'draft-mtp'
EOF
  exit 3
fi

# --------------------------- fetch model files ----------------------------
mkdir -p "$MODEL_DIR"
MTP_SUBDIR="MTP"; MTP_FILE_REL="mtp-gemma-4-12B-it-${MTP_QUANT}.gguf"
if [[ ! -f "$MODEL_DIR/$MODEL_FILE" || ! -f "$MODEL_DIR/$MTP_SUBDIR/$MTP_FILE_REL" ]]; then
  echo "[setup] downloading model + MTP drafter from $HF_REPO (~6.7 GB) ..."
  if ! command -v hf >/dev/null 2>&1; then pip install -q -U "huggingface_hub[cli]" hf_transfer; fi
  hf download "$HF_REPO" \
      "$MODEL_FILE" \
      "$MTP_SUBDIR/$MTP_FILE_REL" \
      "$MTP_SUBDIR/mtp-gemma-4-12B-it-Q4_0.gguf" \
      --local-dir "$MODEL_DIR"
fi

# ------------------------------- preflight --------------------------------
echo "[preflight] engine : $LLAMA_SERVER"
echo "[preflight] profile: FIT_MODE=$FIT_MODE ctx=$CTX n-batch=$NB MTP=$ENABLE_MTP($MTP_QUANT)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
  FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1)
  NEED_MIB=$((6405 + 242 + 900 + (CTX / 128)))   # weights+MTP+buffers+KV estimate
  if (( FREE_MIB < NEED_MIB )); then
    echo "[preflight][WARN] free VRAM ${FREE_MIB} MiB < estimated need ${NEED_MIB} MiB."
    echo "                  Close GPU apps, or use FIT_MODE=safe. (No sysmem fallback => hard OOM instead of 4 tok/s.)"
  fi
else
  echo "[preflight][WARN] nvidia-smi not found; cannot verify VRAM headroom."
fi

# ---------------------------- assemble command ----------------------------
ARGS=(
  --model   "$MODEL_DIR/$MODEL_FILE"
  -ngl 99                                  # ALL tensors on GPU — never split layers
  --flash-attn on                          # required for quantized KV on Ada
  --ctx-size "$CTX"
  --cache-type-k q8_0 --cache-type-v q8_0  # -47% KV vs f16, ~0% decode cost <=32K
  --n-batch "$NB" --ubatch-size "$NB"      # bounded compute buffers
  --parallel 1                             # single KV slab, no fragmentation
  --no-mmap                                # full VRAM residency; no host page-cache tensors
  --cache-ram 0                            # disable host-RAM prompt cache
  --no-warmup                              # avoid startup allocation spike
  --jinja                                  # proper chat/tool template rendering
  --temp 1.0 --top-p 0.95 --top-k 64       # Google's official Gemma 4 QAT settings
  --chat-template-kwargs "{\"enable_thinking\":$ENABLE_THINKING}"
  --alias "gemma-4-12b-qat-gpuonly"
  --host "$HOST" --port "$PORT"
)
if [[ "$ENABLE_MTP" == "1" && -f "$MODEL_DIR/$MTP_SUBDIR/$MTP_FILE_REL" ]]; then
  # MTP: lossless multi-token speculation (llama.cpp PR #23398). 1.4-2.2x decode.
  # SPEC_MIX=1 adds draftless ngram-mod (good for code-repetition workloads).
  # NOTE: issue #24266 showed ngram+MTP conflicts on some older builds — A/B via
  #       deploy/test_mandates.sh T1 on YOUR build before enabling SPEC_MIX=1.
  SPEC_TYPES="draft-mtp"; [[ "${SPEC_MIX:-0}" == "1" ]] && SPEC_TYPES="draft-mtp,ngram-mod"
  ARGS+=( -md "$MODEL_DIR/$MTP_SUBDIR/$MTP_FILE_REL"
          --spec-type "$SPEC_TYPES"
          --spec-draft-n-max "${SPEC_DRAFT_N_MAX:-2}"   # sweep 1..4, keep fastest
          --spec-draft-ngl all                          # drafter 100% in VRAM (isolation)
          -ctkd q8_0 -ctvd q8_0 )                       # drafter KV quantized too
fi
ARGS+=( --metrics )   # Prometheus telemetry: KV usage + speculative acceptance

echo "[launch] $LLAMA_SERVER ${ARGS[*]}"
echo "[launch] validate afterwards:  ./benchmark_validate.sh"
exec "$LLAMA_SERVER" "${ARGS[@]}"
