#!/usr/bin/env bash
# monitor_resources.sh — sample host/GPU hardware metrics + SGLang KV cache occupancy every 1s
# Usage: ./monitor_resources.sh <output.csv> [sglang_port]
# Columns:
#   timestamp_s, mem_available_mb, mem_used_mb, cpu_util_pct,
#   gpu_util_pct, gpu_mem_used_mb, gpu_mem_free_mb, gpu_mem_total_mb,
#   disk_read_mb_s, disk_write_mb_s, disk_util_pct,
#   kv_used_tokens, kv_evictable_tokens, kv_available_tokens, kv_total_tokens, kv_occupancy_pct,
#   host_used_tokens, host_total_tokens, host_occupancy_pct,
#   evicted_tokens_total, load_back_tokens_total, cache_hit_rate,
#   num_running_reqs, num_queue_reqs, gen_throughput,
#   e2e_latency_mean_ms

set -uo pipefail
OUT="${1:-resources.csv}"
SGLANG_PORT="${2:-30000}"
METRICS_URL="http://localhost:${SGLANG_PORT}/metrics"

# ---------------------------------------------------------------------------
# Disk device detection
# ---------------------------------------------------------------------------
DISK_DEV=$(basename "$(findmnt -n -o SOURCE / 2>/dev/null)" 2>/dev/null)
if [[ -z "$DISK_DEV" ]]; then
    DISK_DEV=$(awk '$2=="/" {print $1}' /proc/mounts | tail -1 | xargs basename 2>/dev/null)
fi
DISK_DEV="${DISK_DEV:-nvme0n1p1}"
if ! awk -v dev="$DISK_DEV" '$3==dev{found=1} END{exit !found}' /proc/diskstats 2>/dev/null; then
    DISK_DEV=$(echo "$DISK_DEV" | sed 's/p[0-9]*$//')
fi

# ---------------------------------------------------------------------------
# CSV header
# ---------------------------------------------------------------------------
echo "timestamp_s,mem_available_mb,mem_used_mb,cpu_util_pct,\
gpu_util_pct,gpu_mem_used_mb,gpu_mem_free_mb,gpu_mem_total_mb,\
disk_read_mb_s,disk_write_mb_s,disk_util_pct,\
kv_used_tokens,kv_evictable_tokens,kv_available_tokens,kv_total_tokens,kv_occupancy_pct,\
host_used_tokens,host_total_tokens,host_occupancy_pct,\
evicted_tokens_total,load_back_tokens_total,cache_hit_rate,\
num_running_reqs,num_queue_reqs,gen_throughput,\
e2e_latency_mean_ms" > "$OUT"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
read_cpu_ticks() {
    awk '/^cpu /{print $2+$3+$4+$5+$6+$7+$8, $5}' /proc/stat
}
read_disk_stats() {
    awk -v dev="$DISK_DEV" '$3==dev{print $6, $10, $13}' /proc/diskstats
}

# Parse a single metric value from Prometheus text: "metricname{...} VALUE"
# Usage: parse_metric <metric_name> <prometheus_text>
# Prints the numeric value, or 0 if not found.
parse_metric() {
    local name="$1"
    local text="$2"
    echo "$text" | awk -v m="^${name}[{ ]" '$0 ~ m { gsub(/.*[} ]/, ""); printf "%.0f\n", $0; found=1; exit }
                                             END { if (!found) print 0 }'
}

prev_cpu=( $(read_cpu_ticks) )
prev_disk=( $(read_disk_stats) )
prev_disk=( ${prev_disk[0]:-0} ${prev_disk[1]:-0} ${prev_disk[2]:-0} )
prev_e2e_sum=0
prev_e2e_count=0
prev_gen_tok=0

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
while true; do
    ts=$(date +%s.%N)

    # Host memory
    mem_total=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
    mem_avail=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
    mem_used_mb=$(( (mem_total - mem_avail) / 1024 ))
    mem_avail_mb=$(( mem_avail / 1024 ))

    # CPU utilization
    curr_cpu=( $(read_cpu_ticks) )
    total_diff=$(( curr_cpu[0] - prev_cpu[0] ))
    idle_diff=$(( curr_cpu[1]  - prev_cpu[1] ))
    cpu_util=$(( total_diff > 0 ? 100 * (total_diff - idle_diff) / total_diff : 0 ))
    prev_cpu=( "${curr_cpu[@]}" )

    # GPU hardware metrics
    read gpu_util gpu_mem_used gpu_mem_free gpu_mem_total < <(
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,memory.total \
            --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' '{ u+=$1; um+=$2; uf+=$3; ut+=$4; n++ }
                     END { if(n>0) printf "%d %d %d %d\n", u/n, um/n, uf/n, ut/n;
                           else print "0 0 0 0" }' \
        || echo "0 0 0 0"
    )

    # Disk I/O
    curr_disk=( $(read_disk_stats) )
    curr_disk=( ${curr_disk[0]:-0} ${curr_disk[1]:-0} ${curr_disk[2]:-0} )
    read_diff=$(( curr_disk[0]  - prev_disk[0] ))
    write_diff=$(( curr_disk[1] - prev_disk[1] ))
    ioticks_diff=$(( curr_disk[2] - prev_disk[2] ))
    disk_read_mb=$(echo "scale=2; $read_diff  * 512 / 1000000" | bc 2>/dev/null || echo "0")
    disk_write_mb=$(echo "scale=2; $write_diff * 512 / 1000000" | bc 2>/dev/null || echo "0")
    disk_util=$(( ioticks_diff > 1000 ? 100 : ioticks_diff / 10 ))
    prev_disk=( "${curr_disk[@]}" )

    # SGLang KV cache metrics from /metrics (Prometheus text)
    prom=$(curl -s --max-time 1 "$METRICS_URL" 2>/dev/null || true)

    kv_used=$(      parse_metric "sglang:kv_used_tokens"         "$prom")
    kv_evictable=$( parse_metric "sglang:kv_evictable_tokens"    "$prom")
    kv_available=$( parse_metric "sglang:kv_available_tokens"    "$prom")
    host_used=$(    parse_metric "sglang:hicache_host_used_tokens"  "$prom")
    host_total=$(   parse_metric "sglang:hicache_host_total_tokens" "$prom")
    evicted=$(      parse_metric "sglang:evicted_tokens_total"    "$prom")
    load_back=$(    parse_metric "sglang:load_back_tokens_total"  "$prom")

    # cache_hit_rate is a float — keep 4 decimals
    cache_hit=$(echo "$prom" | awk '/^sglang:cache_hit_rate[{ ]/ { gsub(/.*[} ]/, ""); printf "%.4f\n", $0; found=1; exit }
                                    END { if (!found) print "0.0000" }')

    # num_pages = total KV token capacity
    kv_total=$(parse_metric "sglang:num_pages" "$prom")

    # Scheduler queue depth
    num_running=$(parse_metric "sglang:num_running_reqs" "$prom")
    num_queue=$(  parse_metric "sglang:num_queue_reqs"   "$prom")

    # Generation throughput: delta of cumulative counter per second (stable, vs flaky gauge)
    gen_tok_total=$(echo "$prom" | awk '/^sglang:generation_tokens_total[{ ]/ { gsub(/.*[} ]/, ""); printf "%.0f\n", $0; found=1; exit } END { if (!found) print 0 }')
    gen_tput=$(awk -v c="$gen_tok_total" -v p="$prev_gen_tok" \
        'BEGIN { delta = c - p; if (delta >= 0) printf "%.1f", delta; else print "0.0" }')

    # E2E latency mean: derive from histogram _sum/_count delta per second
    e2e_sum=$(echo "$prom"   | awk '/^sglang:e2e_request_latency_seconds_sum[{ \n]/ { gsub(/.*[} ]/, ""); printf "%.6f\n", $0; found=1; exit } END { if (!found) print "0" }')
    e2e_count=$(echo "$prom" | awk '/^sglang:e2e_request_latency_seconds_count[{ \n]/ { gsub(/.*[} ]/, ""); printf "%.0f\n", $0; found=1; exit } END { if (!found) print "0" }')
    e2e_mean_ms=$(awk -v s="$e2e_sum" -v c="$e2e_count" \
        -v ps="$prev_e2e_sum" -v pc="$prev_e2e_count" \
        'BEGIN {
            dc = c - pc; ds = s - ps
            if (dc > 0) printf "%.1f", 1000 * ds / dc
            else print "0.0"
        }')

    # Occupancy: (used + evictable) / total  [evictable = cached but not locked]
    kv_occ=$(awk -v u="$kv_used" -v e="$kv_evictable" -v t="$kv_total" \
        'BEGIN { if (t>0) printf "%.2f", 100*(u+e)/t; else print "0.00" }')
    host_occ=$(awk -v u="$host_used" -v t="$host_total" \
        'BEGIN { if (t>0) printf "%.2f", 100*u/t; else print "0.00" }')

    echo "${ts},${mem_avail_mb},${mem_used_mb},${cpu_util},\
${gpu_util:-0},${gpu_mem_used:-0},${gpu_mem_free:-0},${gpu_mem_total:-0},\
${disk_read_mb},${disk_write_mb},${disk_util},\
${kv_used},${kv_evictable},${kv_available},${kv_total},${kv_occ},\
${host_used},${host_total},${host_occ},\
${evicted:-0},${load_back:-0},${cache_hit},\
${num_running:-0},${num_queue:-0},${gen_tput:-0.00},\
${e2e_mean_ms}" >> "$OUT"

    prev_e2e_sum="$e2e_sum"
    prev_e2e_count="$e2e_count"
    prev_gen_tok="$gen_tok_total"

    sleep 1
done
