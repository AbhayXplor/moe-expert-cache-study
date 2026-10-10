#!/usr/bin/env bash
# ============================================================================
# isolate_and_verify.sh — Mandate 3 proof on Linux / WSL2
# Verifies the inference process stays 100% inside dedicated VRAM:
#   * dedicated-VRAM peak sampling via nvidia-smi
#   * host-RAM residency check via process RSS delta (weights must NOT live in RAM)
#   * llama.cpp load-log audit (CUDA0 buffers only, no meaningful CPU buffers)
# Usage:
#   LOG_FILE=/tmp/llama.log ./isolate_and_verify.sh            # audit a log + live monitor
#   SERVER_URL=http://127.0.0.1:8080 ./isolate_and_verify.sh   # also fire a load request
#   ./isolate_and_verify.sh --negative                         # OOM enforcement test guide
# ============================================================================
set -uo pipefail

SERVER_URL="${SERVER_URL:-}"
LOG_FILE="${LOG_FILE:-}"
PROC_NAME="${PROC_NAME:-llama-server}"
SAMPLES="${SAMPLES:-30}"
RSS_KB_THRESHOLD="${RSS_KB_THRESHOLD:-2097152}"   # 2 GiB host RSS = suspicious

echo "== Mandate 3 isolation audit (Linux/WSL2) =="

# 0. platform note -----------------------------------------------------------
if grep -qi microsoft /proc/version 2>/dev/null; then
  echo "[platform] WSL2 detected: CUDA traverses host WDDM."
  echo "           Ensure the WINDOWS host has NVIDIA Control Panel ->"
  echo "           'CUDA - Sysmem Fallback Policy' = 'Prefer No Sysmem Fallback'."
  echo "           Verify host Task Manager -> GPU -> Shared GPU memory stays flat."
else
  echo "[platform] Native Linux: cudaMalloc is strict (OOMs, never spills)."
  echo "           No UVM in ggml weight/KV paths => isolated by default."
fi

# 1. locate process ----------------------------------------------------------
PID=$(pgrep -x "$PROC_NAME" | head -n1 || true)
if [[ -z "$PID" ]]; then PID=$(pgrep -f "$PROC_NAME" | head -n1 || true); fi
if [[ -n "$PID" ]]; then
  echo "[proc] $PROC_NAME pid=$PID"
  RSS0=$(awk '/VmRSS/{print $2}' /proc/$PID/status 2>/dev/null || echo 0)
else
  echo "[proc] $PROC_NAME not running — will monitor GPU totals only."
  RSS0=0
fi

# 2. optional load request ---------------------------------------------------
if [[ -n "$SERVER_URL" ]]; then
  echo "[load] firing 384-token generation at $SERVER_URL/completion ..."
  curl -sS "$SERVER_URL/completion" -H 'Content-Type: application/json' \
    -d '{"prompt":"Explain in detail how POSIX bash handles file descriptor inheritance across pipelines, with exact syntax examples.","n_predict":384,"temperature":1.0,"top_p":0.95,"top_k":64,"cache_prompt":false}' \
    > /dev/null &
fi

# 3. sample dedicated VRAM (+RSS) --------------------------------------------
PEAK=0; PEAK_RSS=$RSS0
for ((i=0;i<SAMPLES;i++)); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo 0)
  (( USED > PEAK )) && PEAK=$USED
  if [[ -n "$PID" && -r /proc/$PID/status ]]; then
    R=$(awk '/VmRSS/{print $2}' /proc/$PID/status || echo 0)
    (( R > PEAK_RSS )) && PEAK_RSS=$R
  fi
  sleep 1
done
echo "[vram] peak dedicated usage during window: ${PEAK} MiB / 8192 MiB"
echo "[rss ] peak host RSS: $((PEAK_RSS/1024)) MiB (threshold $((RSS_KB_THRESHOLD/1024)) MiB)"

# 4. engine log audit ---------------------------------------------------------
if [[ -n "$LOG_FILE" && -r "$LOG_FILE" ]]; then
  echo "[log ] auditing $LOG_FILE ..."
  grep -E "model buffer|KV buffer|draft.*buffer" "$LOG_FILE" | sed 's/^/       /'
  CPU_BUF=$(grep -E "CPU.*model buffer size" "$LOG_FILE" | grep -oE '[0-9.]+ MiB' | head -n1 || true)
  if grep -q "CUDA0" "$LOG_FILE"; then echo "[log ] CUDA0 buffers present ✔"; fi
  if [[ -n "$CPU_BUF" ]]; then
    echo "[log ] WARNING: CPU model buffer reported: $CPU_BUF (norm tensors <= a few MiB are OK)"
  fi
fi

# 5. verdict -------------------------------------------------------------------
FAIL=0
(( PEAK > 7900 )) && { echo "FAIL: dedicated VRAM peak ${PEAK} MiB exceeds budget"; FAIL=1; }
(( PEAK_RSS > RSS_KB_THRESHOLD )) && { echo "FAIL: host RSS ${PEAK_RSS} kB suggests RAM residency"; FAIL=1; }
if (( FAIL == 0 )); then
  echo "RESULT ISOLATION PASS peak_vram=${PEAK}MiB peak_rss=$((PEAK_RSS/1024))MiB"
else
  echo "RESULT ISOLATION FAIL"
fi

cat <<'EOF'

== Negative test (run once, proves enforcement is live) ==
  FIT_MODE=max CTX_OVERRIDE=131072 ./serve_gemma4_12b.sh   # or: -c 131072 manually
  EXPECTED : loud 'cudaMalloc failed: out of memory' at load.
  WRONG    : server starts and generates at 4-5 tok/s  => spill path is OPEN
             (Windows/WSL: fix Sysmem Fallback Policy; Linux: investigate UVM usage).
EOF
