"""
MemCentric-Sim: Publication-Grade Visualization Generator
==========================================================
Generates high-resolution performance and energy charts:
- Speedup Comparison across Policies
- Data Movement Reduction (%)
- Energy-Delay Product (EDP) Improvement
- Architectural Roofline Model (Host CPU vs Near-Memory PIM)
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def generate_all_plots(results_path: str = None, output_dir: str = None):
    if results_path is None:
        results_path = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "results.json")
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")

    os.makedirs(output_dir, exist_ok=True)

    with open(results_path, "r") as f:
        data = json.load(f)

    workloads = list(data.keys())
    policies = ["STATIC_HOST", "STATIC_PIM", "ROOFLINE_OI", "DYNAMIC_ANALYTICAL", "ORACLE"]
    policy_labels = {
        "STATIC_HOST": "Host Baseline",
        "STATIC_PIM": "PIM Only",
        "ROOFLINE_OI": "Roofline-OI",
        "DYNAMIC_ANALYTICAL": "MemCentric (Dynamic)",
        "ORACLE": "Oracle (upper bound)"
    }
    colors = {
        "STATIC_HOST": "#4A5568",
        "STATIC_PIM": "#ED8936",
        "ROOFLINE_OI": "#3182CE",
        "DYNAMIC_ANALYTICAL": "#38A169",
        "ORACLE": "#A0AEC0"
    }

    # 1. SPEEDUP COMPARISON
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    x = np.arange(len(workloads))
    width = 0.16
    multiplier = 0

    for pol in policies:
        speedups = [data[wl][pol]["speedup"] for wl in workloads]
        offset = width * multiplier
        rects = ax.bar(x + offset, speedups, width, label=policy_labels[pol], color=colors[pol], alpha=0.9, edgecolor="black", linewidth=0.5)
        ax.bar_label(rects, padding=3, fmt="%.2fx", fontsize=6, rotation=90)
        multiplier += 1

    ax.set_ylabel("Speedup vs Baseline Host (x, log scale)", fontweight="bold")
    ax.set_yscale("log")
    ax.set_title("Execution Speedup: Host Baseline vs Static PIM vs MemCentric Dynamic", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(workloads, rotation=15, ha="right", fontweight="bold")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    p1 = os.path.join(output_dir, "speedup_comparison.png")
    fig.savefig(p1)
    plt.close(fig)
    print(f"Generated: {p1}")

    # 2. DATA MOVEMENT REDUCTION
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    multiplier = 0
    for pol in policies:
        data_kb = [data[wl][pol]["data_moved_kb"] for wl in workloads]
        offset = width * multiplier
        rects = ax.bar(x + offset, data_kb, width, label=policy_labels[pol], color=colors[pol], alpha=0.9, edgecolor="black", linewidth=0.5)
        multiplier += 1

    ax.set_ylabel("Data Moved Over Interconnect (KB)", fontweight="bold")
    ax.set_yscale("log")
    ax.set_title("Interconnect Data Movement Volume (Lower is Better - Log Scale)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(workloads, rotation=15, ha="right", fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    p2 = os.path.join(output_dir, "data_movement.png")
    fig.savefig(p2)
    plt.close(fig)
    print(f"Generated: {p2}")

    # 3. EDP IMPROVEMENT
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    for pol in ["STATIC_PIM", "ROOFLINE_OI", "DYNAMIC_ANALYTICAL"]:
        edp_gains = [data[wl][pol]["edp_reduction_pct"] for wl in workloads]
        offset = width * multiplier
        ax.plot(workloads, edp_gains, marker="o", linewidth=2.2, label=policy_labels[pol], color=colors[pol])

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("EDP Reduction / Gain (%) vs Baseline", fontweight="bold")
    ax.set_title("Energy-Delay Product (EDP) Improvement (%)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(range(len(workloads)))
    ax.set_xticklabels(workloads, rotation=15, ha="right", fontweight="bold")
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    p3 = os.path.join(output_dir, "edp_improvement.png")
    fig.savefig(p3)
    plt.close(fig)
    print(f"Generated: {p3}")

    # 4. ROOFLINE MODEL
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    oi_range = np.logspace(-2, 2, 200) # 0.01 to 100 FLOPs/Byte

    # Host: Peak Compute = 3.2 GHz * 64 ops = 204.8 GFLOPs
    # Host: Peak Bus BW = 64 GB/s
    host_peak_flops = 204.8
    host_bus_bw = 64.0
    host_roofline = np.minimum(host_peak_flops, oi_range * host_bus_bw)

    # PIM: Peak Compute = 1.0 GHz * 16 ops = 16.0 GFLOPs
    # PIM: Peak Internal Bank BW = 1024 GB/s
    pim_peak_flops = 16.0
    pim_internal_bw = 1024.0
    pim_roofline = np.minimum(pim_peak_flops, oi_range * pim_internal_bw)

    ax.plot(oi_range, host_roofline, label="Host CPU Roofline (64 GB/s CXL, 204 GFLOPs)", color="#3182CE", linewidth=2.5)
    ax.plot(oi_range, pim_roofline, label="Near-Memory PIM Roofline (1024 GB/s TSV, 16 GFLOPs)", color="#ED8936", linewidth=2.5, linestyle="--")

    # Annotate workload operational intensities
    ax.axvspan(0.01, 1.0, alpha=0.15, color="orange", label="Memory-Bound Region (PIM Favored)")
    ax.axvspan(2.0, 100.0, alpha=0.15, color="blue", label="Compute-Bound Region (Host Favored)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Operational Intensity (FLOPs / Byte)", fontweight="bold")
    ax.set_ylabel("Attainable Performance (GFLOPs/s)", fontweight="bold")
    ax.set_title("Architectural Roofline Model: Host Compute vs Near-Memory Computing", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    p4 = os.path.join(output_dir, "roofline_model.png")
    fig.savefig(p4)
    plt.close(fig)
    print(f"Generated: {p4}")

    # 5. DECISION-OVERHEAD FEASIBILITY STUDY
    study_path = os.path.join(os.path.dirname(results_path), "overhead_study.json")
    if os.path.exists(study_path):
        with open(study_path, "r") as f:
            study = json.load(f)
        plot_overhead_study(study, output_dir)


def plot_overhead_study(study, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    sweep = study["sample_rate_sweep"]
    rates = [r["sample_rate"] for r in sweep]
    ax1.plot(rates, [r["oracle_efficiency_pct"] for r in sweep], marker="o", color="#38A169",
             linewidth=2.2, label="Performance vs oracle (%)")
    ax1.set_xscale("log")
    ax1.set_ylim(90, 100.5)
    ax1.set_xlabel("Profiler sample rate (fraction of address stream)", fontweight="bold")
    ax1.set_ylabel("Performance vs oracle (%)", fontweight="bold", color="#38A169")
    ax1b = ax1.twinx()
    ax1b.plot(rates, [r["overhead_pct"] for r in sweep], marker="s", color="#E53E3E",
              linewidth=2.2, linestyle="--", label="Decision overhead (% of runtime)")
    ax1b.set_ylabel("Decision overhead (% of runtime)", fontweight="bold", color="#E53E3E")
    for r in sweep:
        ax1.annotate(f"{r['oracle_agreement']}/{r['tasks']}", (r["sample_rate"], r["oracle_efficiency_pct"]),
                     textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
    ax1.set_title("Sampling: overhead vs decision quality\n(labels = phases matching oracle)",
                  fontsize=11, fontweight="bold")

    cost = study["profiling_cost_sweep"]
    cps = [r["cycles_per_sample"] for r in cost]
    host = study["host_cycles"]
    ax2.plot(cps, [r["speedup"] for r in cost], marker="o", color="#38A169", linewidth=2.2,
             label="MemCentric dynamic (full-trace profiling)")
    ax2.axhline(host / study["best_static_or_roofline_cycles"], color="#3182CE", linestyle="--",
                label="Best of static / roofline")
    ax2.axhline(host / study["oracle_cycles"], color="#A0AEC0", linestyle=":", label="Oracle (zero overhead)")
    ax2.axhline(1.0, color="#4A5568", linewidth=0.8, label="Host baseline")
    ax2.set_xscale("log")
    ax2.set_xlabel("Profiling cost (cycles per sampled access)", fontweight="bold")
    ax2.set_ylabel("Speedup vs host (x)", fontweight="bold")
    ax2.set_title("Break-even: how costly can decisions get?", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower left", fontsize=8, frameon=True)

    plt.tight_layout()
    p5 = os.path.join(output_dir, "overhead_study.png")
    fig.savefig(p5)
    plt.close(fig)
    print(f"Generated: {p5}")


if __name__ == "__main__":
    generate_all_plots()
