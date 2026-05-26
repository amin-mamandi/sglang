#!/usr/bin/env python3
"""
Plot HiCache benchmark results: warm-cache TTFT, throughput, and resource utilization.

Supports two modes:
  1) Baseline/HiCache comparison (--compare mode):
     Expects: warm_cache_baseline.jsonl, warm_cache_hicache.jsonl
  2) Sweep mode (--sweep mode):
     Expects: warm_cache_*.jsonl files for scenarios (gpu-fit, gpu-evict, 2tier-fit, 2tier-overflow)
              with optional page-size suffixes (_ps2, _ps4, _ps8, etc.)

Usage:
  python3 plot_results.py <results_dir>
  python3 plot_results.py <results_dir> --output plots_custom/
  python3 plot_results.py <results_dir> --show  # Display plots interactively
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import seaborn as sns
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

_RCPARAMS = {
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 200,
    "font.family": "serif",
}

# ---------------------------------------------------------------------------
# Plotting Utilities
# ---------------------------------------------------------------------------
def _style(ax) -> None:
    ax.grid(True, axis="both", linestyle="--", alpha=0.35, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def _tag(ax, text: str) -> None:
    ax.text(0.97, 0.97, text, transform=ax.transAxes,
            fontsize=7, fontweight="bold", va="top", ha="right", color="#444444")


def _legend(fig, handles, labels: List[str], ncol: Optional[int] = None) -> None:
    ncol = ncol or len(handles)
    fig.legend(handles, labels, loc="lower center", ncol=ncol,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02),
               handlelength=1.2, handleheight=0.8,
               handletextpad=0.5, borderpad=0.35)

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def load_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({k: float(v) if v not in (None, "") else 0.0
                             for k, v in r.items()})
            except (ValueError, TypeError):
                continue
    return rows

def warm_cache_summary(rows):
    out = []
    for r in rows:
        pct = r.get("shared_prefix_pct")
        if pct is None:
            continue
        out.append({
            "pct":        pct,
            "ttft_mean":  r.get("mean_ttft_ms", 0),
            "ttft_p90":   r.get("p90_ttft_ms",  0),
            "throughput": r.get("input_throughput", 0),
        })
    return sorted(out, key=lambda x: x["pct"])

# ---------------------------------------------------------------------------
# Sweep mode detection & grouping
# ---------------------------------------------------------------------------
def detect_sweep_files(results_dir):
    """Detect if results_dir contains sweep mode files.
    
    Returns dict: {scenario_name: {page_size: path_to_jsonl}}
    """
    sweep_files = {}
    
    # Pattern: bench_SCENARIO(_psN)?(.jsonl) or warm_cache_SCENARIO(_psN)?(.jsonl)
    pattern = re.compile(r"(?:bench|warm_cache)_([a-z0-9\-]+)(?:_ps(\d+))?\.jsonl")

    for fpath in sorted(list(results_dir.glob("bench_*.jsonl")) +
                        list(results_dir.glob("warm_cache_*.jsonl"))):
        match = pattern.match(fpath.name)
        if not match:
            continue
        
        scenario = match.group(1)
        page_size = int(match.group(2)) if match.group(2) else 1
        
        if scenario not in sweep_files:
            sweep_files[scenario] = {}
        sweep_files[scenario][page_size] = fpath
    
    return sweep_files if sweep_files else None

# ---------------------------------------------------------------------------
# Warm-cache plots
# ---------------------------------------------------------------------------
def plot_warm_cache(results_dir, out_dir):
    base_file = results_dir / "warm_cache_baseline.jsonl"
    hi_file   = results_dir / "warm_cache_hicache.jsonl"
    if not base_file.exists() or not hi_file.exists():
        print("  skip warm_cache plots (missing files)")
        return

    base = warm_cache_summary(load_jsonl(base_file))
    hi   = warm_cache_summary(load_jsonl(hi_file))
    if not base or not hi:
        print("  skip warm_cache plots (empty data)")
        return

    sns.set_style("whitegrid")
    plt.rcParams.update(_RCPARAMS)

    pcts_b = [r["pct"] for r in base]
    pcts_h = [r["pct"] for r in hi]

    # --- TTFT ---
    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    _style(ax)
    h1, = ax.plot(pcts_b, [r["ttft_mean"] for r in base],
                  linestyle=LS_BASELINE, marker=MK_BASELINE,
                  color=C_BASELINE, linewidth=1.4, markersize=4, label="Baseline")
    h2, = ax.plot(pcts_h, [r["ttft_mean"] for r in hi],
                  linestyle=LS_HICACHE, marker=MK_HICACHE,
                  color=C_HICACHE,  linewidth=1.4, markersize=4, label="HiCache")
    ax.fill_between(pcts_b,
                    [r["ttft_mean"] for r in base],
                    [r["ttft_p90"]  for r in base],
                    color=C_BASELINE, alpha=0.12)
    ax.fill_between(pcts_h,
                    [r["ttft_mean"] for r in hi],
                    [r["ttft_p90"]  for r in hi],
                    color=C_HICACHE,  alpha=0.12)
    ax.set_xlabel("Shared prefix (%)")
    ax.set_ylabel("TTFT (ms)")
    _tag(ax, "mean ± P90 band")
    _legend(fig, [h1, h2], ["Baseline", "HiCache"])
    fig.savefig(out_dir / "warm_cache_ttft.png", dpi=450, bbox_inches="tight")
    plt.close(fig)
    print("  saved warm_cache_ttft.png")

    # --- Throughput ---
    fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    _style(ax)
    h1, = ax.plot(pcts_b, [r["throughput"] for r in base],
                  linestyle=LS_BASELINE, marker=MK_BASELINE,
                  color=C_BASELINE, linewidth=1.4, markersize=4, label="Baseline")
    h2, = ax.plot(pcts_h, [r["throughput"] for r in hi],
                  linestyle=LS_HICACHE, marker=MK_HICACHE,
                  color=C_HICACHE,  linewidth=1.4, markersize=4, label="HiCache")
    ax.set_xlabel("Shared prefix (%)")
    ax.set_ylabel("Input throughput (tok/s)")
    _legend(fig, [h1, h2], ["Baseline", "HiCache"])
    fig.savefig(out_dir / "warm_cache_throughput.png", dpi=450, bbox_inches="tight")
    plt.close(fig)
    print("  saved warm_cache_throughput.png")

    # --- Speedup bar ---
    common = sorted(set(r["pct"] for r in base) & set(r["pct"] for r in hi))
    if common:
        base_map = {r["pct"]: r["ttft_mean"] for r in base}
        hi_map   = {r["pct"]: r["ttft_mean"] for r in hi}
        speedup  = [base_map[p] / hi_map[p] if hi_map[p] > 0 else 1.0 for p in common]

        fig, ax = plt.subplots(figsize=(8, 3.0), constrained_layout=True)
        _style(ax)
        ax.grid(False, axis="x")
        colors  = ["#029e73" if s >= 1 else "#d55e00" for s in speedup]
        bars = ax.bar([f"{p}%" for p in common], speedup,
                      color=colors, edgecolor="#000000",
                      linewidth=0.6, width=0.6, zorder=3)
        ax.axhline(1.0, color="#888888", linewidth=0.8, linestyle="--", zorder=2)
        ax.set_xlabel("Shared prefix (%)")
        ax.set_ylabel("Speedup  (Baseline / HiCache)")
        _tag(ax, ">1 = HiCache faster")
        fig.savefig(out_dir / "warm_cache_speedup.png", dpi=450, bbox_inches="tight")
        plt.close(fig)
        print("  saved warm_cache_speedup.png")

# ---------------------------------------------------------------------------
# Sweep mode plots (consolidated: all scenarios × page sizes on one chart)
# ---------------------------------------------------------------------------
def plot_sweep(results_dir, out_dir, sweep_files):
    """Plot sweep results on consolidated charts.
    
    Creates two plots:
    - sweep_ttft_consolidated.png: TTFT for all scenarios & page sizes
    - sweep_throughput_consolidated.png: Throughput for all scenarios & page sizes
    
    X-axis: shared prefix (%)
    Colors: page sizes
    Markers: scenarios
    """
    
    # Color palette for different page sizes (colorblind-safe)
    colors_ps = ["#0173b2", "#d55e00", "#029e73", "#cc78bc", "#ca9161", "#949494"]
    
    # Markers for different scenarios
    markers_scenario = {
        "gpu-fit": "o",
        "gpu-evict": "^",
        "2tier-fit": "s",
        "2tier-overflow": "D"
    }
    
    # Scenario display names
    scenario_names = {
        "gpu-fit": "GPU Fit",
        "gpu-evict": "GPU Evict",
        "2tier-fit": "2-Tier Fit",
        "2tier-overflow": "2-Tier Overflow"
    }
    
    # Collect all data
    all_data = {}  # {scenario: {page_size: data}}
    all_page_sizes = set()
    
    for scenario in sweep_files.keys():
        all_data[scenario] = {}
        page_sizes = sorted(sweep_files[scenario].keys())
        all_page_sizes.update(page_sizes)
        
        for ps in page_sizes:
            fpath = sweep_files[scenario][ps]
            rows = load_jsonl(fpath)
            all_data[scenario][ps] = warm_cache_summary(rows)
    
    if not all_data:
        return

    sns.set_style("whitegrid")
    plt.rcParams.update(_RCPARAMS)

    all_page_sizes = sorted(all_page_sizes)

    def _sweep_plot(ylabel, metric_key, out_name):
        fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
        _style(ax)
        handles = []
        for ps_idx, ps in enumerate(all_page_sizes):
            ps_color = colors_ps[ps_idx % len(colors_ps)]
            for scenario in sorted(all_data.keys()):
                if ps not in all_data[scenario]:
                    continue
                data = all_data[scenario][ps]
                if not data:
                    continue
                pcts = [r["pct"] for r in data]
                vals = [r[metric_key] for r in data]
                marker = markers_scenario.get(scenario, "o")
                h, = ax.plot(pcts, vals, marker=marker, color=ps_color,
                             linewidth=1.4, markersize=4, linestyle="-",
                             label=f"PS {ps} - {scenario_names.get(scenario, scenario)}")
                handles.append(h)
        ax.set_xlabel("Shared prefix (%)")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        _tag(ax, f"colors=page size  markers=scenario")
        if handles:
            ax.legend(handles, [h.get_label() for h in handles],
                      loc="upper left", frameon=False,
                      handlelength=1.2, handleheight=0.8,
                      handletextpad=0.5, borderpad=0.35)
        fig.savefig(out_dir / out_name, dpi=450, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_name}")

    _sweep_plot("TTFT (ms)", "ttft_mean", "sweep_ttft_consolidated.png")
    _sweep_plot("Input throughput (tok/s)", "throughput", "sweep_throughput_consolidated.png")

# ---------------------------------------------------------------------------
# Multiturn plots
# ---------------------------------------------------------------------------
def plot_multiturn(results_dir, out_dir):
    base_file = results_dir / "multiturn_baseline.jsonl"
    hi_file   = results_dir / "multiturn_hicache.jsonl"
    if not base_file.exists() or not hi_file.exists():
        print("  skip multiturn plots (missing files)")
        return

    def parse(path):
        out = []
        for r in load_jsonl(path):
            s = r.get("summary", {})
            rate = s.get("request_rate")
            ttft = s.get("average_ttft")
            hit  = s.get("cache_hit_rate", 0)
            if rate is not None and ttft is not None:
                out.append({"rate": rate, "ttft": ttft, "hit": hit})
        return sorted(out, key=lambda x: x["rate"])

    base = parse(base_file)
    hi   = parse(hi_file)
    if not base or not hi:
        print("  skip multiturn plots (parse failed)")
        return

    sns.set_style("whitegrid")
    plt.rcParams.update(_RCPARAMS)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)

    ax = axes[0]
    _style(ax)
    h1, = ax.plot([r["rate"] for r in base], [r["ttft"] * 1000 for r in base],
                  linestyle=LS_BASELINE, marker=MK_BASELINE,
                  color=C_BASELINE, linewidth=1.4, markersize=4, label="Baseline")
    h2, = ax.plot([r["rate"] for r in hi],   [r["ttft"] * 1000 for r in hi],
                  linestyle=LS_HICACHE, marker=MK_HICACHE,
                  color=C_HICACHE,  linewidth=1.4, markersize=4, label="HiCache")
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel("Mean TTFT (ms)")
    _tag(ax, "multiturn TTFT")

    ax = axes[1]
    _style(ax)
    ax.plot([r["rate"] for r in base], [r["hit"] for r in base],
            linestyle=LS_BASELINE, marker=MK_BASELINE,
            color=C_BASELINE, linewidth=1.4, markersize=4)
    ax.plot([r["rate"] for r in hi],   [r["hit"] for r in hi],
            linestyle=LS_HICACHE, marker=MK_HICACHE,
            color=C_HICACHE,  linewidth=1.4, markersize=4)
    ax.set_xlabel("Request rate (req/s)")
    ax.set_ylabel("Cache hit rate")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    _tag(ax, "cache hit rate")

    _legend(fig, [h1, h2], ["Baseline", "HiCache"])
    fig.savefig(out_dir / "multiturn.png", dpi=450, bbox_inches="tight")
    plt.close(fig)
    print("  saved multiturn.png")

# ---------------------------------------------------------------------------
# Phase annotation helpers
# ---------------------------------------------------------------------------

# Phase colours (light fills, drawn behind data)
_PHASE_COLORS = {
    "gpu":       ("#d0e8ff", None),           # blue  — no legend entry
    "dram":      ("#d6f0d6", "DRAM tier"),    # green — GPU evicts → DRAM
    "disk":      ("#f0e6ff", None),           # purple — no legend entry
    "recompute": ("#fde8d0", None),           # orange — no legend entry
}

def _compute_phases(results_dir, scenario_base):
    """Return list of (t_start, t_end, phase_key) for a scenario.

    Phase is determined by comparing the active KV token demand for each
    benchmark pct step against the GPU and combined GPU+DRAM pool sizes.

    Pool sizes are read from the Docker container log embedded in run.log
    (line: 'KV Cache is allocated. #tokens: N').  If unavailable we fall
    back to the hicache-ratio embedded in the server-args line.
    """
    jsonl = results_dir / f"bench_{scenario_base}.jsonl"
    if not jsonl.exists():
        jsonl = results_dir / f"warm_cache_{scenario_base}.jsonl"
    if not jsonl.exists():
        return [], []

    rows = load_jsonl(jsonl)
    if not rows:
        return [], []

    # --- parse pool sizes from run.log ---
    gpu_tokens = None
    hicache_ratio = None
    log_path = results_dir / "run.log"
    if log_path.exists():
        log_text = log_path.read_text()
        # GPU pool size
        m = re.search(r"KV Cache is allocated\. #tokens:\s*(\d+)", log_text)
        if m:
            gpu_tokens = int(m.group(1))
        # hicache ratio for this scenario
        # look for the server-args block nearest to this scenario label
        pattern = re.compile(
            rf"=== SCENARIO: {re.escape(scenario_base)}\b.*?hicache_ratio=([\d.]+)",
            re.S
        )
        m2 = pattern.search(log_text)
        if m2:
            hicache_ratio = float(m2.group(1))

    if gpu_tokens is None:
        gpu_tokens = 100_000  # safe fallback
    dram_tokens = int(gpu_tokens * hicache_ratio) if hicache_ratio else 0
    total_2tier = gpu_tokens + dram_tokens

    # Detect 3-tier: storage backend present in server args in the log
    has_storage = False
    if log_path and log_path.exists():
        scen_block_pat = re.compile(
            rf"=== SCENARIO: {re.escape(scenario_base)}\b(.*?)(?====|$)", re.S)
        m3 = scen_block_pat.search(log_path.read_text())
        if m3 and "hicache-storage-backend" in m3.group(1):
            has_storage = True

    # For 3-tier, disk is effectively unbounded for our workload — the "overflow"
    # scenario overflows GPU+DRAM but disk capacity > total peak, so we never hit
    # a 4th "recompute" tier unless explicitly modelled.
    # We treat disk tier as unbounded here; adjust if disk size is known.
    DISK_TOKENS = float("inf")

    num_prompts = rows[0].get("num_prompts", 32)

    # Use wall-clock timestamps if available (exact alignment with resource CSV).
    # Fall back to accumulated benchmark_duration if not present.
    has_wall_time = all("round_start_wall_time" in r for r in rows)

    # CSV t0 for relative-time alignment
    csv_path = results_dir / f"resources_{scenario_base}.csv"
    csv_t0 = None
    if has_wall_time and csv_path.exists():
        with open(csv_path) as f:
            first = next(iter(csv.DictReader(f)), None)
            if first:
                try:
                    csv_t0 = float(first["timestamp_s"])
                except (KeyError, ValueError):
                    pass

    phases = []
    pct_labels = []  # (t_start, shared_prefix_pct)

    if has_wall_time and csv_t0 is not None:
        # Anchor each round at its real wall-clock start relative to CSV t0
        for i, r in enumerate(rows):
            t_start = r["round_start_wall_time"] - csv_t0
            # t_end: next round's start, or current start + benchmark_duration
            if i + 1 < len(rows):
                t_end = rows[i + 1]["round_start_wall_time"] - csv_t0
            else:
                t_end = t_start + r["benchmark_duration"]

            pct    = r["shared_prefix_pct"]
            pre    = r["prefix_len"]
            suf    = r["suffix_len"]
            active = pre + num_prompts * suf

            pct_labels.append((t_start, pct))

            if active <= gpu_tokens:
                key = "gpu"
            elif dram_tokens > 0 and active <= total_2tier:
                key = "dram"
            elif has_storage and active <= total_2tier + DISK_TOKENS:
                key = "disk"
            else:
                key = "recompute"

            phases.append((t_start, t_end, key))
    else:
        # Fallback: accumulate benchmark_duration (may drift due to setup overhead)
        t = 0.0
        for r in rows:
            dur    = r["benchmark_duration"]
            pre    = r["prefix_len"]
            suf    = r["suffix_len"]
            pct    = r["shared_prefix_pct"]
            active = pre + num_prompts * suf

            pct_labels.append((t, pct))

            if active <= gpu_tokens:
                key = "gpu"
            elif dram_tokens > 0 and active <= total_2tier:
                key = "dram"
            elif has_storage and active <= total_2tier + DISK_TOKENS:
                key = "disk"
            else:
                key = "recompute"

            phases.append((t, t + dur, key))
            t += dur

    return phases, pct_labels


def _annotate_phases(axes, phases, pct_labels=None):
    """Draw coloured axvspan bands on every axis.

    Two layers are drawn:
      1. Tier-phase background (gpu/dram/recompute) at alpha=0.25.
      2. Per-pct foreground strip at alpha=0.18 so each round's prefix-share
         percentage has its own distinct colour — no text labels needed.

    pct_labels: list of (t_start, pct_value) per benchmark round, in order.
    The end time of each round is inferred from the next round's t_start
    (last round ends at the rightmost phase boundary).
    """
    if not phases:
        return
    from matplotlib.patches import Patch
    import matplotlib.cm as cm

    # --- layer 1: tier-phase background ---
    for ax in axes:
        for t0, t1, key in phases:
            color, _ = _PHASE_COLORS[key]
            ax.axvspan(t0, t1, color=color, alpha=0.25, linewidth=0, zorder=0)

    # --- layer 2: per-pct foreground colour strips ---
    pct_legend_handles = []
    if pct_labels:
        unique_pcts = sorted(set(p for _, p in pct_labels))
        # Qualitative palette — enough distinct colours for typical pct counts
        _qual_colors = [
            "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
            "#ff7f00", "#a65628", "#f781bf", "#999999",
            "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
        ]
        pct_color = {pct: _qual_colors[i % len(_qual_colors)]
                     for i, pct in enumerate(unique_pcts)}

        # Build (t_start, t_end, pct) intervals
        t_end_overall = phases[-1][1]
        pct_intervals = []
        for i, (t_start, pct) in enumerate(pct_labels):
            t_end = pct_labels[i + 1][0] if i + 1 < len(pct_labels) else t_end_overall
            pct_intervals.append((t_start, t_end, pct))

        for ax in axes:
            for t0, t1, pct in pct_intervals:
                ax.axvspan(t0, t1, color=pct_color[pct], alpha=0.18,
                           linewidth=0, zorder=1)

        # Boundary lines between rounds
        for i, (t_start, _) in enumerate(pct_labels):
            if i == 0:
                continue
            for ax in axes:
                ax.axvline(t_start, color="#888888", linewidth=0.5,
                           linestyle=":", alpha=0.5, zorder=2)

        # Legend patches for pct values
        for pct in unique_pcts:
            pct_legend_handles.append(
                Patch(facecolor=pct_color[pct], alpha=0.7, edgecolor="none",
                      label=f"{int(pct)}% prefix")
            )

    # Tier-phase legend patches — only for phases with a non-None label
    seen = {}
    for _, _, key in phases:
        if key not in seen:
            color, label = _PHASE_COLORS[key]
            if label is not None:
                seen[key] = Patch(facecolor=color, alpha=0.55, edgecolor="none", label=label)

    return list(seen.values()) + pct_legend_handles


# ---------------------------------------------------------------------------
# Resource plots: KV cache occupancy + GPU util + disk
# ---------------------------------------------------------------------------
def plot_host_memory(results_dir, out_dir):
    """Plot resource metrics grouped by base scenario.

    Panel layout (top → bottom):
      1. GPU KV cache occupancy %   (kv_used + kv_evictable) / kv_total
      2. Host DRAM KV occupancy %   host_used / host_total        [only if hicache]
      3. Eviction rate (tokens/s)   delta of evicted_tokens_total + load_back_tokens_total
      4. GPU compute utilization %
      5. Disk I/O MB/s              [only if columns present]
      6. Disk utilization %         [only if columns present]

    Falls back to raw GPU memory (GB, log scale) when KV occupancy columns are absent.
    """

    # Discover CSV files
    found = []
    for label, fname in [("Baseline", "resources_baseline.csv"),
                         ("HiCache",  "resources_hicache.csv")]:
        p = results_dir / fname
        if p.exists():
            found.append((label, load_csv(p)))

    if not found:
        pattern = re.compile(r"resources_([a-z0-9\-_]+)\.csv")
        for fpath in sorted(results_dir.glob("resources_*.csv")):
            m = pattern.match(fpath.name)
            if m:
                found.append((m.group(1), load_csv(fpath)))

    if not found:
        print("  skip resource plots (no resources CSV)")
        return

    sns.set_style("whitegrid")
    plt.rcParams.update(_RCPARAMS)

    def _base_scenario(label):
        return re.sub(r"_ps\d+$", "", label)

    groups: Dict[str, List] = {}
    for label, rows in found:
        groups.setdefault(_base_scenario(label), []).append((label, rows))

    color_palette = ["#0173b2", "#d55e00", "#029e73", "#cc78bc", "#ca9161", "#949494"]
    # Marker per page-size tag — "ps1" → 'o', "ps8" → 's', "ps64" → '^', others cycle
    _PS_MARKERS = {"ps1": "o", "ps8": "s", "ps64": "^"}
    _MARKER_CYCLE = ["o", "s", "^", "D", "v", "P"]

    sample = found[0][1][0]
    has_kv_occ   = "kv_occupancy_pct"   in sample and "kv_total_tokens" in sample
    has_host_occ = "host_occupancy_pct" in sample and "host_total_tokens" in sample
    has_eviction  = "evicted_tokens_total" in sample
    has_hit_rate  = "cache_hit_rate" in sample
    has_scheduler = "num_running_reqs" in sample
    has_gen_tput  = "gen_throughput" in sample
    has_e2e_lat   = "e2e_latency_mean_ms" in sample
    has_gpu_mem   = (not has_kv_occ) and "gpu_mem_used_mb" in sample
    has_cpu       = "cpu_util_pct" in sample
    has_disk      = False  # disk panels removed

    # Panel count
    nrows = (1 if has_kv_occ    else 0) + \
            (1 if has_host_occ  else 0) + \
            (1 if has_eviction  else 0) + \
            (1 if has_hit_rate  else 0) + \
            (1 if has_scheduler else 0) + \
            (1 if has_gen_tput  else 0) + \
            (1 if has_e2e_lat   else 0) + \
            (1 if has_gpu_mem   else 0) + \
            (1 if has_cpu       else 0) + \
            1 +                           \
            (2 if has_disk      else 0)   # always: GPU util
    nrows = max(nrows, 1)

    saved = []
    for grp_name, members in groups.items():
        lengths = [len(r) for _, r in members]
        cap = sorted(lengths)[len(lengths) // 2] * 2

        COLORS = {lbl: color_palette[i % len(color_palette)]
                  for i, (lbl, _) in enumerate(members)}

        phases, pct_labels = _compute_phases(results_dir, grp_name)

        fig, axes = plt.subplots(nrows, 1,
                                 figsize=(10, 2.4 * nrows),
                                 constrained_layout=True)
        if nrows == 1:
            axes = [axes]

        ax_idx = 0
        handles, labels_legend = [], []
        first_label = members[0][0]

        for label, rows in members:
            rows = rows[:cap]
            t0  = rows[0]["timestamp_s"]
            ts  = [r["timestamp_s"] - t0 for r in rows]
            c   = COLORS[label]
            ps_tag = label[len(grp_name):].lstrip("_") or "ps1"
            marker = _PS_MARKERS.get(ps_tag, _MARKER_CYCLE[list(COLORS).index(label) % len(_MARKER_CYCLE)])
            mkw = dict(marker=marker, markevery=max(1, len(ts)//20), markersize=4)

            if has_kv_occ:
                # GPU KV occupancy: (used + evictable) / total
                kv_occ = []
                for r in rows:
                    total = float(r.get("kv_total_tokens") or 0)
                    if total > 0:
                        occ = 100.0 * (float(r.get("kv_used_tokens", 0)) +
                                       float(r.get("kv_evictable_tokens", 0))) / total
                    else:
                        occ = float(r.get("kv_occupancy_pct", 0))
                    kv_occ.append(min(occ, 100.0))
                h, = axes[ax_idx].plot(ts, kv_occ, color=c, linewidth=1.4,
                                       label=ps_tag, zorder=3, **mkw)
                handles.append(h); labels_legend.append(ps_tag)

            if has_host_occ:
                host_occ = []
                for r in rows:
                    total = float(r.get("host_total_tokens") or 0)
                    if total > 0:
                        occ = 100.0 * float(r.get("host_used_tokens", 0)) / total
                    else:
                        occ = float(r.get("host_occupancy_pct", 0))
                    host_occ.append(min(occ, 100.0))
                hi_ax = ax_idx + (1 if has_kv_occ else 0)
                axes[hi_ax].plot(ts, host_occ, color=c, linewidth=1.4, zorder=3, **mkw)
                if not has_kv_occ:
                    h, = axes[hi_ax].plot([], [], color=c, linewidth=1.4, label=ps_tag, **mkw)
                    handles.append(h); labels_legend.append(ps_tag)

            if has_eviction:
                ev_vals = [float(r.get("evicted_tokens_total", 0)) for r in rows]

                def _rate(vals, ts_list):
                    out = [0.0]
                    for i in range(1, len(vals)):
                        dt = ts_list[i] - ts_list[i-1]
                        delta = vals[i] - vals[i-1]
                        out.append(max(delta, 0) / dt if dt > 0 else 0.0)
                    return out

                base_ev_ax = ax_idx + (1 if has_kv_occ else 0) + (1 if has_host_occ else 0)
                axes[base_ev_ax].plot(ts, _rate(ev_vals, ts),
                                      color=c, linewidth=1.2, zorder=3, **mkw)
                if not has_kv_occ and not has_host_occ:
                    h, = axes[base_ev_ax].plot([], [], color=c, linewidth=1.4, label=ps_tag, **mkw)
                    handles.append(h); labels_legend.append(ps_tag)

            if has_hit_rate:
                hr_vals = [float(r.get("cache_hit_rate", 0)) for r in rows]
                hr_ax = ax_idx + (1 if has_kv_occ else 0) + (1 if has_host_occ else 0) + \
                        (1 if has_eviction else 0)
                axes[hr_ax].plot(ts, [v * 100.0 for v in hr_vals],
                                 color=c, linewidth=1.2, zorder=3, **mkw)
                if not has_kv_occ and not has_host_occ and not has_eviction:
                    h, = axes[hr_ax].plot([], [], color=c, linewidth=1.4, label=ps_tag, **mkw)
                    handles.append(h); labels_legend.append(ps_tag)

            _sched_off = (1 if has_kv_occ else 0) + (1 if has_host_occ else 0) + \
                         (1 if has_eviction else 0) + (1 if has_hit_rate else 0)
            if has_scheduler:
                sched_ax = ax_idx + _sched_off
                running = [float(r.get("num_running_reqs", 0)) for r in rows]
                queued  = [float(r.get("num_queue_reqs",   0)) for r in rows]
                axes[sched_ax].plot(ts, running, color=c, linewidth=1.2, zorder=3, **mkw)
                axes[sched_ax].plot(ts, queued,  color=c, linewidth=1.0, zorder=3,
                                    linestyle="--", alpha=0.7, **mkw)

            if has_gen_tput:
                gt_ax = ax_idx + _sched_off + (1 if has_scheduler else 0)
                axes[gt_ax].plot(ts,
                    [float(r.get("gen_throughput", 0)) for r in rows],
                    color=c, linewidth=1.2, zorder=3, **mkw)

            if has_e2e_lat:
                el_ax = ax_idx + _sched_off + (1 if has_scheduler else 0) + (1 if has_gen_tput else 0)
                axes[el_ax].plot(ts,
                    [float(r.get("e2e_latency_mean_ms", 0)) for r in rows],
                    color=c, linewidth=1.2, zorder=3, **mkw)

            if has_gpu_mem:
                gmu = [max(r["gpu_mem_used_mb"],  1) / 1024 for r in rows]
                gmt = [max(r["gpu_mem_total_mb"], 1) / 1024 for r in rows]
                h, = axes[ax_idx].plot(ts, gmu, color=c, linewidth=1.4,
                                       label=ps_tag, zorder=3, **mkw)
                handles.append(h); labels_legend.append(ps_tag)
                if label == first_label:
                    axes[ax_idx].plot(ts, gmt, color="#888888",
                                      linewidth=0.8, linestyle="--", zorder=3)

            # GPU util axis index
            gpu_util_off = (1 if has_kv_occ    else 0) + \
                           (1 if has_host_occ   else 0) + \
                           (1 if has_eviction   else 0) + \
                           (1 if has_hit_rate   else 0) + \
                           (1 if has_scheduler  else 0) + \
                           (1 if has_gen_tput   else 0) + \
                           (1 if has_e2e_lat    else 0) + \
                           (1 if has_gpu_mem    else 0) + \
                           (1 if has_cpu        else 0)
            gpu_util_ax = ax_idx + gpu_util_off

            if has_cpu:
                cpu_ax = ax_idx + (1 if has_kv_occ   else 0) + \
                                  (1 if has_host_occ  else 0) + \
                                  (1 if has_eviction  else 0) + \
                                  (1 if has_hit_rate  else 0) + \
                                  (1 if has_scheduler else 0) + \
                                  (1 if has_gen_tput  else 0) + \
                                  (1 if has_e2e_lat   else 0) + \
                                  (1 if has_gpu_mem   else 0)
                axes[cpu_ax].plot(ts, [r["cpu_util_pct"] for r in rows],
                                  color=c, linewidth=1.4, zorder=3, **mkw)

            axes[gpu_util_ax].plot(ts, [r["gpu_util_pct"] for r in rows],
                                   color=c, linewidth=1.4, zorder=3, **mkw)

            if has_disk:
                disk_ax = gpu_util_ax + 1
                axes[disk_ax].plot(ts,
                    [float(r.get("disk_read_mb_s",  0)) for r in rows],
                    color=c, linewidth=1.4, linestyle="-",  zorder=3)
                axes[disk_ax].plot(ts,
                    [float(r.get("disk_write_mb_s", 0)) for r in rows],
                    color=c, linewidth=1.0, linestyle="--", zorder=3)
                axes[disk_ax + 1].plot(ts,
                    [float(r.get("disk_util_pct", 0)) for r in rows],
                    color=c, linewidth=1.4, zorder=3)

        # Phase + pct annotations
        phase_handles = _annotate_phases(list(axes), phases, pct_labels)

        # Label axes
        if has_kv_occ:
            _style(axes[ax_idx])
            axes[ax_idx].set_ylabel("GPU KV occupancy (%)")
            axes[ax_idx].set_ylim(0, 105)
            _tag(axes[ax_idx], f"{grp_name} — GPU KV cache")
            ax_idx += 1

        if has_host_occ:
            _style(axes[ax_idx])
            axes[ax_idx].set_ylabel("Host KV occupancy (%)")
            axes[ax_idx].set_ylim(0, 105)
            _tag(axes[ax_idx], "host DRAM KV cache")
            ax_idx += 1

        if has_eviction:
            _style(axes[ax_idx])
            axes[ax_idx].set_ylabel("Eviction rate (tok/s)")
            _tag(axes[ax_idx], "GPU → DRAM evictions")
            ax_idx += 1

        if has_hit_rate:
            _style(axes[ax_idx])
            axes[ax_idx].set_ylabel("Cache hit rate (%)")
            axes[ax_idx].set_ylim(0, 105)
            _tag(axes[ax_idx], "prefix cache hit rate")
            ax_idx += 1

        if has_scheduler:
            _style(axes[ax_idx])
            axes[ax_idx].set_ylabel("Requests")
            _tag(axes[ax_idx], "running (—)  queued (--)")
            ax_idx += 1

        if has_gen_tput:
            _style(axes[ax_idx])
            axes[ax_idx].set_ylabel("Gen throughput (tok/s)")
            _tag(axes[ax_idx], "server-side generation throughput")
            ax_idx += 1

        if has_e2e_lat:
            _style(axes[ax_idx])
            axes[ax_idx].set_ylabel("E2E latency (ms)")
            _tag(axes[ax_idx], "mean e2e request latency")
            ax_idx += 1

        if has_gpu_mem:
            _style(axes[ax_idx])
            axes[ax_idx].set_yscale("log")
            axes[ax_idx].set_ylabel("GPU memory (GB, log)")
            axes[ax_idx].yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _: f"{v:.3g}"))
            _tag(axes[ax_idx], f"{grp_name} — GPU memory")
            ax_idx += 1

        if has_cpu:
            _style(axes[ax_idx])
            axes[ax_idx].set_ylabel("CPU util (%)")
            axes[ax_idx].set_ylim(0, 100)
            _tag(axes[ax_idx], "CPU util")
            ax_idx += 1

        _style(axes[ax_idx])
        axes[ax_idx].set_ylabel("GPU util (%)")
        axes[ax_idx].set_ylim(0, 100)
        _tag(axes[ax_idx], "GPU util")
        ax_idx += 1

        if has_disk:
            _style(axes[ax_idx])
            axes[ax_idx].set_ylabel("Disk I/O (MB/s)")
            _tag(axes[ax_idx], "disk read (—) write (--)")
            ax_idx += 1
            _style(axes[ax_idx])
            axes[ax_idx].set_ylabel("Disk util (%)")
            axes[ax_idx].set_ylim(0, 100)
            _tag(axes[ax_idx], "disk util")
            ax_idx += 1

        axes[-1].set_xlabel("Time (s)")

        all_handles = handles + (phase_handles or [])
        all_labels  = labels_legend + [h.get_label() for h in (phase_handles or [])]
        _legend(fig, all_handles, all_labels)

        fname = f"resources_{grp_name}.png"
        fig.savefig(out_dir / fname, dpi=450, bbox_inches="tight")
        plt.close(fig)
        saved.append(fname)

    for f in saved:
        print(f"  saved {f}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: plot_results.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.is_dir():
        print(f"Not a directory: {results_dir}")
        sys.exit(1)

    if not HAS_MPL:
        print("matplotlib not installed — install with: pip install matplotlib")
        sys.exit(1)

    out_dir = results_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    print(f"Plotting {results_dir} → {out_dir}")

    # Detect mode (sweep vs baseline/hicache comparison)
    sweep_files = detect_sweep_files(results_dir)
    
    if sweep_files:
        print("  Detected sweep mode (multiple scenarios & page sizes)")
        plot_sweep(results_dir, out_dir, sweep_files)
    else:
        print("  Detected baseline/hicache mode")
        plot_warm_cache(results_dir, out_dir)
        plot_multiturn(results_dir, out_dir)
    
    # Plot host memory/resources (works for both modes if data exists)
    plot_host_memory(results_dir, out_dir)

    print("Done.")

if __name__ == "__main__":
    main()
