#!/usr/bin/env bash
# run_benchmark.sh — HiCache benchmark
#
# Runs accumulating-cache bench scenarios across page sizes.
# The KV cache is flushed once at start; phases run back-to-back without
# flushing so the cache fills naturally across fill → hit rounds.
#
# Usage
#   bash run_benchmark.sh [OPTIONS]
#
# Options
#   --skip-server           assume server already running on $PORT
#   --model MODEL           HuggingFace model path
#   --port PORT             server port (default: 30000)
#   --context-len N         model context length (default: 32768)
#   --total-tokens N        tokens per request (default: 16384)
#   --page-sizes "A B ..."  space-separated page sizes (default: "1 8 64")
#   --quick                 fewer prompts for a fast sanity run
#
# Env-var overrides
#   MODEL, PORT, HF_CACHE, DOCKER_IMAGE
#   CONTEXT_LEN, TOTAL_TOKENS
#   PAGE_SIZES
#   NUM_PROMPTS, OUTPUT_LEN, MAX_CONCURRENCY

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${PORT:-30000}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
DOCKER_IMAGE="${DOCKER_IMAGE:-lmsysorg/sglang:latest}"
CONTEXT_LEN="${CONTEXT_LEN:-32768}"
TOTAL_TOKENS="${TOTAL_TOKENS:-16384}"
PAGE_SIZES="${PAGE_SIZES:-1 64}"

NUM_PROMPTS="${NUM_PROMPTS:-32}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR_IN_CONTAINER="/sgl-workspace/sglang/benchmark/hicache"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
SKIP_SERVER=false

usage() { grep '^#' "$0" | grep -v '#!/' | sed 's/^# \?//'; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-server)    SKIP_SERVER=true;             shift ;;
    --model)          MODEL="$2";                   shift 2 ;;
    --port)           PORT="$2";                    shift 2 ;;
    --context-len)    CONTEXT_LEN="$2";             shift 2 ;;
    --total-tokens)   TOTAL_TOKENS="$2";            shift 2 ;;
    --page-sizes)     PAGE_SIZES="$2";              shift 2 ;;
    --quick)          NUM_PROMPTS=8; MAX_CONCURRENCY=8; shift ;;
    --help|-h)        usage ;;
    *) echo "Unknown flag: $1"; usage ;;
  esac
done

# ---------------------------------------------------------------------------
# Output + logging
# ---------------------------------------------------------------------------
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$DIR/results_bench_${TIMESTAMP}"
mkdir -p "$OUT"
LOG="$OUT/run.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG" >&2; }
die() { log "ERROR: $*"; exit 1; }
hr()  { log "$(printf '─%.0s' {1..60})"; }

log "Model        : $MODEL"
log "Port         : $PORT"
log "Context len  : $CONTEXT_LEN"
log "Total tokens : $TOTAL_TOKENS"
log "Page sizes   : $PAGE_SIZES"
log "Output       : $OUT"
hr

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
docker info > /dev/null 2>&1 || die "Cannot reach Docker daemon. Try: newgrp docker"

# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------
free_port() {
  local existing
  existing=$(docker ps --format '{{.Names}} {{.Ports}}' 2>/dev/null \
    | awk -v p="$PORT" '$0 ~ ":"p"->" {print $1}' || true)
  if [[ -n "$existing" ]]; then
    log "Stopping existing container on port $PORT: $existing"
    docker stop "$existing" >> "$LOG" 2>&1 || true
    sleep 3
  fi
}

wait_for_server() {
  log "Waiting for server on port $PORT ..."
  local i=0
  until curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
    i=$((i+1))
    [[ $i -gt 600 ]] && die "Server not healthy after 600s"
    [[ $((i % 15)) -eq 0 ]] && log "  still waiting... (${i}s)"
    sleep 1
  done
  log "Server ready."
}

start_server() {
  local name="$1"
  local mem_frac="$2"
  local extra_args="${3:-}"

  free_port
  sync && sudo bash -c 'echo 1 > /proc/sys/vm/drop_caches' 2>/dev/null || true
  log "Starting server: $name  (mem_fraction=$mem_frac)"
  [[ -n "$extra_args" ]] && log "  extra args: $extra_args"

  # shellcheck disable=SC2086
  docker run -d \
    --name "$name" \
    --gpus all \
    -p "${PORT}:30000" \
    -v "${HF_CACHE}:/root/.cache/huggingface" \
    "$DOCKER_IMAGE" \
    python3 -m sglang.launch_server \
      --model-path "$MODEL" \
      --host 0.0.0.0 \
      --port 30000 \
      --context-length "$CONTEXT_LEN" \
      --mem-fraction-static "$mem_frac" \
      --enable-metrics \
      $extra_args \
    >> "$LOG" 2>&1

  wait_for_server

  log "Waiting for server to settle (GPU util → idle) ..."
  local i=0
  while true; do
    gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
               | awk '{s+=$1;n++} END{print (n?int(s/n):100)}')
    [[ "${gpu_util:-100}" -le 5 ]] && break
    i=$((i+1))
    [[ $i -gt 30 ]] && { log "  GPU still busy after 30s — proceeding anyway"; break; }
    sleep 1
  done
  log "Server settled (GPU util=${gpu_util:-?}%)."
}

stop_server() {
  local name="$1"
  log "Stopping $name"
  docker stop "$name" >> "$LOG" 2>&1 || true
  sleep 3
}

# ---------------------------------------------------------------------------
# Run bench_warm_cache.py inside the container
# ---------------------------------------------------------------------------
run_bench() {
  local label="$1"
  local container="$2"
  local bench_name="$3"
  local out_file_host="$OUT/bench_${label}.jsonl"
  local out_file_container="/tmp/bench_${label}.jsonl"
  hr; log "bench_warm_cache.py [$label]"

  docker cp "$DIR/bench_warm_cache.py" \
    "$container:$BENCH_DIR_IN_CONTAINER/bench_warm_cache.py" >> "$LOG" 2>&1 || true

  docker exec "$container" \
    python3 "$BENCH_DIR_IN_CONTAINER/bench_warm_cache.py" \
      --model           "$MODEL" \
      --port            30000 \
      --total-tokens    "$TOTAL_TOKENS" \
      --num-prompts     "$NUM_PROMPTS" \
      --output-len      "$OUTPUT_LEN" \
      --max-concurrency "$MAX_CONCURRENCY" \
      --output-file     "$out_file_container" \
      --bench           "$bench_name" \
    2>&1 | tee -a "$LOG" | tee "$OUT/bench_${label}.txt" >&2 \
    || log "WARNING: bench_warm_cache.py [$label] exited non-zero"

  docker cp "$container:$out_file_container" "$out_file_host" 2>/dev/null \
    && log "Saved → $out_file_host" \
    || log "WARNING: could not copy $out_file_container from container"
}

# ---------------------------------------------------------------------------
# Bench scenarios: "bench_name|mem_fraction|extra_server_args"
# mem_fraction=0.10 keeps the GPU KV pool small (~100K tokens) so fill/eviction
# are visible within a short run. hicache-ratio=4.0 gives ~400K host tokens.
# ---------------------------------------------------------------------------
BENCH_SCENARIOS=(
  "bench|0.10|--enable-hierarchical-cache --hicache-ratio 4.0"
)

for scen_def in "${BENCH_SCENARIOS[@]}"; do
  IFS='|' read -r bench_name mem_frac scen_extra <<< "$scen_def"
  for ps in $PAGE_SIZES; do
    ps_suffix=$([ "$ps" = "1" ] && echo "" || echo "_ps${ps}")
    label="${bench_name}${ps_suffix}"
    container="sglang_${label}_${TIMESTAMP}"
    ps_arg=$([ "$ps" != "1" ] && echo " --page-size $ps" || echo "")
    full_extra="${scen_extra}${ps_arg}"

    hr; log "=== BENCH: $bench_name  PAGE SIZE: $ps ==="
    [[ "$SKIP_SERVER" == false ]] && start_server "$container" "$mem_frac" "$full_extra"

    mon_csv="$OUT/resources_${label}.csv"
    bash "$DIR/monitor_resources.sh" "$mon_csv" &
    MON_PID=$!
    log "Resource monitor (PID $MON_PID) → $mon_csv"

    run_bench "$label" "$container" "$bench_name"

    kill "$MON_PID" 2>/dev/null || true
    log "Resource monitor stopped."

    [[ "$SKIP_SERVER" == false ]] && stop_server "$container"
  done
done

hr; log "Generating plots..."
python3.12 "$DIR/plot_results.py" "$OUT" || log "WARNING: plot_results.py failed"

hr
log "All done. Results in: $OUT"
