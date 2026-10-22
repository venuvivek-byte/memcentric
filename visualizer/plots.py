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
    policies = ["STATIC_HOST", "STATIC_PIM", "ROOFLINE_OI", "DYNAMIC_ANALYTICAL"]
    policy_labels = {
        "STATIC_HOST": "Host Baseline",
        "STATIC_PIM": "PIM Only",
        "ROOFLINE_OI": "Roofline-OI",
        "DYNAMIC_ANALYTICAL": "MemCentric (Dynamic)"
    }
    colors = {
        "STATIC_HOST": "#4A5568",
        "STATIC_PIM": "#ED8936",
        "ROOFLINE_OI": "#3182CE",
        "DYNAMIC_ANALYTICAL": "#38A169"
    }

    # 1. SPEEDUP COMPARISON
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(workloads))
    width = 0.2
    multiplier = 0

    for pol in policies:
        speedups = [data[wl][pol]["speedup"] for wl in workloads]
        offset = width * multiplier
        rects = ax.bar(x + offset, speedups, width, label=policy_labels[pol], color=colors[pol], alpha=0.9, edgecolor="black", linewidth=0.5)
        ax.bar_label(rects, padding=3, fmt="%.2fx", fontsize=8)
        multiplier += 1

    ax.set_ylabel("Speedup vs Baseline Host (x)", fontweight="bold")
    ax.set_title("Execution Speedup: Host Baseline vs Static PIM vs MemCentric Dynamic", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(workloads, rotation=15, ha="right", fontweight="bold")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    p1 = os.path.join(output_dir, "speedup_comparison.png")
    fig.savefig(p1)
    plt.close(fig)
    print(f"Generated: {p1}")

    # 2. DATA MOVEMENT REDUCTION
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    multiplier = 0
    for pol in policies:
        data_kb = [data[wl][pol]["data_moved_kb"] for wl in workloads]
        offset = width * multiplier
        rects = ax.bar(x + offset, data_kb, width, label=policy_labels[pol], color=colors[pol], alpha=0.9, edgecolor="black", linewidth=0.5)
        multiplier += 1

    ax.set_ylabel("Data Moved Over Interconnect (KB)", fontweight="bold")
    ax.set_yscale("log")
    ax.set_title("Interconnect Data Movement Volume (Lower is Better - Log Scale)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x + width * 1.5)
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


if __name__ == "__main__":
    generate_all_plots()
