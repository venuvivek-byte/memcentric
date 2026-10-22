"""
MemCentric-Sim: Comprehensive Benchmark & Experiment Suite
===========================================================
Runs rigorous evaluations across workload types and placement policies:
  Workloads:
    - High-Reuse (GEMM 64x64)
    - Streaming Bandwidth-Bound (Vector Reduction & Database Scan)
    - Random Access / Latency-Bound (Graph BFS & Sparse Embedding Lookup)
    - Heterogeneous End-to-End Pipeline
  Policies:
    - STATIC_HOST (Baseline Conventional CPU)
    - STATIC_PIM (Static PIM Offload)
    - ROOFLINE_OI (Operational Intensity Cutoff)
    - DYNAMIC_ANALYTICAL (MemCentric Intelligent Placement)
    - ADAPTIVE_CONGESTION (Congestion-Aware Intelligent Placement)
"""

import os
import sys
import json
from typing import Dict, List, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memcentric.system import MemoryCentricSystem
from memcentric.runtime.policy import PlacementPolicy
from memcentric.workloads.gemm import generate_gemm_task
from memcentric.workloads.streaming import generate_streaming_vector_task, generate_db_filter_scan_task
from memcentric.workloads.graph import generate_graph_bfs_task, generate_sparse_embedding_lookup_task
from memcentric.workloads.pipeline import generate_heterogeneous_pipeline


def get_benchmark_workloads():
    return {
        "High-Reuse (GEMM)": [generate_gemm_task(matrix_dim=64)],
        "Streaming (Vector Sum)": [generate_streaming_vector_task(num_elements=32768)],
        "Streaming (DB Scan)": [generate_db_filter_scan_task(num_records=50000)],
        "Random Access (Graph BFS)": [generate_graph_bfs_task(num_nodes=8000, edges_per_node=6, frontier_size=400)],
        "Random Access (Embedding)": [generate_sparse_embedding_lookup_task(table_entries=15000, lookup_count=1000)],
        "Heterogeneous Pipeline": generate_heterogeneous_pipeline()
    }


def run_all_experiments() -> Dict[str, Any]:
    policies = [
        PlacementPolicy.STATIC_HOST,
        PlacementPolicy.STATIC_PIM,
        PlacementPolicy.ROOFLINE_OI,
        PlacementPolicy.DYNAMIC_ANALYTICAL,
        PlacementPolicy.ADAPTIVE_CONGESTION
    ]

    workload_suite = get_benchmark_workloads()
    results = {}

    print("=" * 80)
    print("RUNNING MEMCENTRIC-SIM ARCHITECTURE EVALUATION")
    print("=" * 80)

    for wl_name, tasks in workload_suite.items():
        print(f"\n--> Benchmark: {wl_name} ({len(tasks)} tasks)")
        results[wl_name] = {}

        # First run baseline (STATIC_HOST) for normalization
        system_baseline = MemoryCentricSystem(policy=PlacementPolicy.STATIC_HOST)
        baseline_res = system_baseline.run_workload(tasks)
        baseline_cycles = baseline_res.total_cycles
        baseline_bytes = baseline_res.total_bytes_transferred
        baseline_edp = baseline_res.edp_joule_sec
        baseline_energy = baseline_res.total_energy_uj

        for pol in policies:
            # Recreate clean system per run
            system = MemoryCentricSystem(policy=pol)
            res = system.run_workload(tasks)

            speedup = baseline_cycles / max(1, res.total_cycles)
            data_reduction_pct = (
                ((baseline_bytes - res.total_bytes_transferred) / max(1, baseline_bytes)) * 100.0
                if baseline_bytes > 0 else 0.0
            )
            energy_saving_pct = (
                ((baseline_energy - res.total_energy_uj) / max(1, baseline_energy)) * 100.0
            )
            edp_reduction_pct = (
                ((baseline_edp - res.edp_joule_sec) / max(1e-12, baseline_edp)) * 100.0
            )

            results[wl_name][pol.value] = {
                "cycles": res.total_cycles,
                "speedup": round(speedup, 3),
                "data_moved_kb": round(res.total_bytes_transferred / 1024.0, 2),
                "data_reduction_pct": round(data_reduction_pct, 2),
                "energy_uj": round(res.total_energy_uj, 2),
                "energy_saving_pct": round(energy_saving_pct, 2),
                "edp_joule_sec": res.edp_joule_sec,
                "edp_reduction_pct": round(edp_reduction_pct, 2),
                "bus_utilization_pct": round(res.bus_utilization * 100.0, 2),
                "decisions": res.decisions_summary,
                "tasks": [
                    {
                        "name": t.task_name,
                        "target": t.target,
                        "cycles": t.elapsed_cycles,
                        "bytes": t.bytes_transferred,
                        "reason": t.decision_reason
                    }
                    for t in res.task_records
                ]
            }

            print(
                f"    [{pol.value:<20}] Speedup: {speedup:5.2f}x | "
                f"Data Saved: {data_reduction_pct:5.1f}% | "
                f"EDP Gain: {edp_reduction_pct:5.1f}% | "
                f"Placements: H={res.decisions_summary['placed_on_host']} P={res.decisions_summary['placed_on_pim']}"
            )

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll benchmark results exported to: {output_file}")
    return results


if __name__ == "__main__":
    run_all_experiments()
