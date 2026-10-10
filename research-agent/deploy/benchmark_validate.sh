#!/usr/bin/env bash
# ============================================================================
# benchmark_validate.sh — prove the deployment meets BOTH hard targets:
#   (1) >= 30 tok/s sustained generation (with MTP lossless speculation)
#   (2) ZERO system-RAM fallback (weights+KV+drafter fully inside 8 GiB VRAM)
# Run AFTER serve_gemma4_12b.sh is up (server on $HOST:$PORT), or use --bench-only.
# ============================================================================
set -uo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/gemma-4-12b-qat}"
MODEL_FILE="gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"
LLAMA_BENCH="${LLAMA_BENCH:-./llama.cpp/build/bin/llama-bench}"
BENCH_ONLY=0; [[ "${1:-}" == "--bench-only" ]] && BENCH_ONLY=1

echo "=============================================================="
echo " STEP 1 — VRAM residency proof (dedicated vs shared)"
echo "=============================================================="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
  # Sample dedicated-VRAM usage for 5 s while idle/serving
  for i in 1 2 3 4 5; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader; sleep 1
  done
  cat <<'EOF'
[check] Linux/WSL2: 'memory.used' above must stay <= ~7800 MiB and the
        llama-server process must appear in 'nvidia-smi' process list.
[check] Windows (hard proof): Task Manager -> Performance -> GPU ->
        'Shared GPU memory' must show ZERO increase during generation.
        (Set NVIDIA Control Panel -> CUDA Sysmem Fallback -> 'Prefer No
         Sysmem Fallback' so overflow OOMs instead of silently spilling.)
EOF
else
  echo "[warn] nvidia-smi missing — cannot verify residency automatically."
fi

echo
echo "=============================================================="
echo " STEP 2 — raw engine throughput (llama-bench, MTP off vs on)"
echo "=============================================================="
if [[ -x "$LLAMA_BENCH" && -f "$MODEL_DIR/$MODEL_FILE" ]]; then
  echo "--- baseline (no MTP): pp512 / tg128 / long-context decode (pg 8192+128)"
  "$LLAMA_BENCH" -m "$MODEL_DIR/$MODEL_FILE" -ngl 99 -fa 1 \
      -p 512 -n 128 -pg 8192,128 -r 2
  if [[ -f "$MODEL_DIR/MTP/mtp-gemma-4-12B-it-Q4_0.gguf" ]]; then
    echo "--- with MTP drafter (expect 1.4-2.2x on tg and pg columns)"
    "$LLAMA_BENCH" -m "$MODEL_DIR/$MODEL_FILE" -ngl 99 -fa 1 \
        -md "$MODEL_DIR/MTP/mtp-gemma-4-12B-it-Q4_0.gguf" \
        --spec-type draft-mtp --spec-draft-n-max 2 \
        -n 128 -pg 8192,128 -r 2 || \
    echo "[note] this llama-bench build may not accept spec flags; measure via server (STEP 3)."
  fi
else
  echo "[skip] llama-bench or model not found at expected paths."
fi

if [[ "$BENCH_ONLY" == "1" ]]; then exit 0; fi

echo
echo "=============================================================="
echo " STEP 3 — live server: long-context generation speed"
echo "=============================================================="
# Build a ~4-6K token prompt (long-horizon context simulation)
PROMPT_FILE=$(mktemp); trap 'rm -f "$PROMPT_FILE" "$RESP_FILE"' EXIT
for i in $(seq 1 60); do
  echo "Module $i: the kernel subsystem handles scheduling, memory mapping, and IRQ routing with strict invariants; subsystem logs must remain byte-exact for audit replay." >> "$PROMPT_FILE"
done
echo "Now write a single POSIX bash function 'rotate_logs' that keeps the newest 5 *.log files in /var/log/app and compresses the rest with gzip -9. Output ONLY the code block." >> "$PROMPT_FILE"
PROMPT=$(python3 - "$PROMPT_FILE" <<'PY'
import sys,json
print(json.dumps(open(sys.argv[1],encoding="utf-8",errors="ignore").read()))
PY
)
RESP_FILE=$(mktemp)
BODY="{\"prompt\": $PROMPT, \"n_predict\": 384, \"temperature\": 1.0, \"top_p\": 0.95, \"top_k\": 64, \"cache_prompt\": false}"
echo "[gen] posting long prompt to http://$HOST:$PORT/completion ..."
T0=$(date +%s.%N)
curl -sS http://$HOST:$PORT/completion -H 'Content-Type: application/json' -d "$BODY" > "$RESP_FILE"
T1=$(date +%s.%N)
WALL=$(python3 -c "print(f'{$T1-$T0:.2f}')")
python3 - "$RESP_FILE" "$WALL" <<'PY'
import sys, json
d = json.load(open(sys.argv[1], encoding="utf-8", errors="ignore"))
n = d.get("tokens_evaluated", 0)
pt = d.get("timings", {}).get("predicted_n"); pms = d.get("timings", {}).get("predicted_ms")
wall = float(sys.argv[2])
tps = (pt / (pms / 1000.0)) if pt and pms else (n / wall if wall > 0 else 0)
print(f"[result] generated tokens : {pt or n}")
print(f"[result] decode speed     : {tps:.1f} tok/s   (TARGET >= 30)")
print("[result] VERDICT         :", "PASS" if tps >= 30 else "FAIL — sweep --spec-draft-n-max 1..4, check MTP active in server log")
txt = d.get("content", "")
print("--------- model output (coding fidelity sample) ---------")
print(txt[:1200])
# extract first code block for syntax check
import re
m = re.search(r"```(?:bash|sh)?\n(.*?)```", txt, re.S)
if m:
    open("/tmp/rotate_logs_test.sh", "w").write("#!/usr/bin/env bash\n" + m.group(1))
    print("[fidelity] extracted code block -> /tmp/rotate_logs_test.sh")
PY
if [[ -f /tmp/rotate_logs_test.sh ]]; then
  echo "[fidelity] bash -n syntax check:"
  bash -n /tmp/rotate_logs_test.sh && echo "[fidelity] PASS — generated script parses as valid bash" \
      || echo "[fidelity] FAIL — inspect /tmp/rotate_logs_test.sh"
fi

echo
echo "=============================================================="
echo " STEP 4 — server-side log verification checklist"
echo "=============================================================="
cat <<'EOF'
In the llama-server stderr log, confirm ALL of:
  [ ] "load_tensors: offloading ... to GPU" for every layer + output tensor
      (line 'n_gpu_layers = 99', 48/48 layers on CUDA0)
  [ ] "CUDA0 model buffer size = ... MiB" and "CUDA0 KV buffer size = ... MiB"
      — NO 'CPU model buffer' / 'CPU KV buffer' lines beyond a few MiB of norm tensors
  [ ] "draft-mtp" / MTP drafter loaded (spec type line present)
  [ ] n_ctx = your configured context; cache type k/v = q8_0
If any tensor shows a CPU buffer > ~1 MiB or Windows 'Shared GPU memory'
moves during generation -> the envelope was violated; lower FIT_MODE.
EOF
