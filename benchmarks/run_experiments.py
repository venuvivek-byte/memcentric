"""
MemCentric-Sim: Comprehensive Benchmark & Experiment Suite
===========================================================
Runs rigorous evaluations across workload types and placement policies:
  Workloads:
    - High-Reuse (GEMM 64x64)
    - Streaming Bandwidth-Bound (Vector Reduction & Database Scan)
    - Random Access / Latency-Bound (Graph BFS & Sparse Embedding Lookup)
    - Heterogeneous End-to-End Pipeline
    - Mixed-Dataset Session (reusable + streaming + random data on one
      system, with cache-resident data and a congested-bus phase)
  Policies:
    - STATIC_HOST (Baseline Conventional CPU)
    - STATIC_PIM (Static PIM Offload)
    - ROOFLINE_OI (Operational Intensity Cutoff)
    - DYNAMIC_ANALYTICAL (MemCentric Intelligent Placement)
    - ADAPTIVE_CONGESTION (Congestion-Aware Intelligent Placement)
    - ORACLE (zero-overhead upper bound by trial execution)
  Feasibility / overhead study (Mixed-Dataset Session):
    - profiler sample-rate sweep: overhead vs decision quality
    - profiling cost sweep: how expensive can decisions get before
      dynamic placement stops paying off
    - cost-model prediction error against simulated cycles
"""

import os
import sys
import json
from typing import Dict, List, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memcentric.system import MemoryCentricSystem
from memcentric.runtime.policy import PlacementPolicy
from memcentric.runtime.profiler import MicroProfiler
from memcentric.workloads.gemm import generate_gemm_task
from memcentric.workloads.streaming import generate_streaming_vector_task, generate_db_filter_scan_task
from memcentric.workloads.graph import generate_graph_bfs_task, generate_sparse_embedding_lookup_task
from memcentric.workloads.pipeline import generate_heterogeneous_pipeline
from memcentric.workloads.mixed import generate_mixed_dataset_session

MIXED = "Mixed-Dataset Session"

POLICIES = [
    PlacementPolicy.STATIC_HOST,
    PlacementPolicy.STATIC_PIM,
    PlacementPolicy.ROOFLINE_OI,
    PlacementPolicy.DYNAMIC_ANALYTICAL,
    PlacementPolicy.ADAPTIVE_CONGESTION,
    PlacementPolicy.ORACLE,
]

WORKLOAD_DESCRIPTIONS = {
    "High-Reuse (GEMM)": "Dense 64x64 GEMM: high temporal reuse, compute bound.",
    "Streaming (Vector Sum)": "Streaming vector reduction: each line touched once, scalar result.",
    "Streaming (DB Scan)": "Column predicate scan: pure streaming, scalar aggregate.",
    "Random Access (Graph BFS)": "BFS frontier expansion: pointer chasing, cache thrashing.",
    "Random Access (Embedding)": "Sparse embedding gather and pooling: random lines, one vector out.",
    "Heterogeneous Pipeline": "Filter -> lookup -> GEMM -> reduce on cold data.",
    MIXED: "One continuous run over reusable, streaming and random datasets, including data "
           "already dirty in the host caches and a phase with a 1 MB DMA burst on the interconnect.",
}


def get_benchmark_workloads():
    return {
        "High-Reuse (GEMM)": generate_gemm_task(matrix_dim=64),
        "Streaming (Vector Sum)": generate_streaming_vector_task(num_elements=32768),
        "Streaming (DB Scan)": generate_db_filter_scan_task(num_records=50000),
        "Random Access (Graph BFS)": generate_graph_bfs_task(num_nodes=8000, edges_per_node=6, frontier_size=400),
        "Random Access (Embedding)": generate_sparse_embedding_lookup_task(table_entries=15000, lookup_count=1000),
        "Heterogeneous Pipeline": generate_heterogeneous_pipeline(),
        MIXED: generate_mixed_dataset_session(),
    }


def _as_list(tasks):
    return tasks if isinstance(tasks, list) else [tasks]


def run_policy(tasks, policy, **system_kwargs):
    system = MemoryCentricSystem(policy=policy, **system_kwargs)
    return system.run_workload(_as_list(tasks))


def run_all_experiments() -> Dict[str, Any]:
    workload_suite = get_benchmark_workloads()
    results = {}

    print("=" * 80)
    print("RUNNING MEMCENTRIC-SIM ARCHITECTURE EVALUATION")
    print("=" * 80)

    for wl_name, tasks in workload_suite.items():
        tasks = _as_list(tasks)
        print(f"\n--> Benchmark: {wl_name} ({len(tasks)} tasks)")
        runs = {pol: run_policy(tasks, pol) for pol in POLICIES}

        base = runs[PlacementPolicy.STATIC_HOST]
        oracle = runs[PlacementPolicy.ORACLE]
        oracle_targets = [t.target for t in oracle.task_records]
        results[wl_name] = {}

        for pol, res in runs.items():
            speedup = base.total_cycles / max(1, res.total_cycles)
            data_reduction_pct = (
                ((base.total_bytes_transferred - res.total_bytes_transferred) / max(1, base.total_bytes_transferred)) * 100.0
                if base.total_bytes_transferred > 0 else 0.0
            )
            energy_saving_pct = ((base.total_energy_uj - res.total_energy_uj) / max(1, base.total_energy_uj)) * 100.0
            edp_reduction_pct = ((base.edp_joule_sec - res.edp_joule_sec) / max(1e-12, base.edp_joule_sec)) * 100.0
            overhead = res.decisions_summary["total_overhead_cycles"]
            matches = sum(1 for t, o in zip(res.task_records, oracle_targets) if t.target == o)

            results[wl_name][pol.value] = {
                "cycles": res.total_cycles,
                "speedup": round(speedup, 3),
                "oracle_efficiency_pct": round(100.0 * oracle.total_cycles / max(1, res.total_cycles), 2),
                "oracle_agreement": f"{matches}/{len(tasks)}",
                "data_moved_kb": round(res.total_bytes_transferred / 1024.0, 2),
                "data_reduction_pct": round(data_reduction_pct, 2),
                "energy_uj": round(res.total_energy_uj, 2),
                "energy_saving_pct": round(energy_saving_pct, 2),
                "edp_joule_sec": res.edp_joule_sec,
                "edp_reduction_pct": round(edp_reduction_pct, 2),
                "bus_utilization_pct": round(res.bus_utilization * 100.0, 2),
                "decision_overhead_cycles": overhead,
                "decision_overhead_pct": round(100.0 * overhead / max(1, res.total_cycles), 3),
                "events_processed": res.events_processed,
                "decisions": res.decisions_summary,
                "tasks": [
                    {
                        "name": t.task_name,
                        "category": t.category,
                        "target": t.target,
                        "cycles": t.elapsed_cycles,
                        "bytes": t.bytes_transferred,
                        "overhead_cycles": t.decision_overhead_cycles,
                        "bus_backlog_cycles": t.bus_backlog_at_arrival,
                        "coherence_writebacks": t.coherence_writebacks,
                        "predicted_host_cycles": t.predicted_host_cycles,
                        "predicted_pim_cycles": t.predicted_pim_cycles,
                        "reason": t.decision_reason
                    }
                    for t in res.task_records
                ]
            }

            print(
                f"    [{pol.value:<20}] Speedup: {speedup:5.2f}x | "
                f"Oracle: {results[wl_name][pol.value]['oracle_efficiency_pct']:5.1f}% | "
                f"Data Saved: {data_reduction_pct:5.1f}% | "
                f"EDP Gain: {edp_reduction_pct:5.1f}% | "
                f"Overhead: {overhead:>5}c | "
                f"Placements: H={res.decisions_summary['placed_on_host']} P={res.decisions_summary['placed_on_pim']}"
            )

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll benchmark results exported to: {output_file}")

    study = run_overhead_study(results)
    with open(os.path.join(output_dir, "overhead_study.json"), "w") as f:
        json.dump(study, f, indent=2)
    print(f"Overhead study exported to: {os.path.join(output_dir, 'overhead_study.json')}")

    export_dashboard_data(results, study)
    return results


def run_overhead_study(results: Dict[str, Any]) -> Dict[str, Any]:
    """Feasibility of dynamic placement on the Mixed-Dataset Session."""
    print("\n" + "=" * 80)
    print("DECISION-OVERHEAD FEASIBILITY STUDY (Mixed-Dataset Session)")
    print("=" * 80)

    mixed = results[MIXED]
    oracle_targets = [t["target"] for t in mixed["ORACLE"]["tasks"]]
    oracle_cycles = mixed["ORACLE"]["cycles"]
    host_cycles = mixed["STATIC_HOST"]["cycles"]
    best_static = min(mixed["STATIC_HOST"]["cycles"], mixed["STATIC_PIM"]["cycles"], mixed["ROOFLINE_OI"]["cycles"])
    tasks = generate_mixed_dataset_session()

    def measure(profiler, decision_cycles=35):
        res = run_policy(tasks, PlacementPolicy.DYNAMIC_ANALYTICAL, profiler=profiler, decision_cycles=decision_cycles)
        overhead = res.decisions_summary["total_overhead_cycles"]
        matches = sum(1 for t, o in zip(res.task_records, oracle_targets) if t.target == o)
        return {
            "cycles": res.total_cycles,
            "speedup": round(host_cycles / res.total_cycles, 3),
            "oracle_efficiency_pct": round(100.0 * oracle_cycles / res.total_cycles, 2),
            "overhead_cycles": overhead,
            "overhead_pct": round(100.0 * overhead / res.total_cycles, 3),
            "oracle_agreement": matches,
            "tasks": len(tasks),
        }

    print("\n  Sample-rate sweep (0.5 cycles per sample):")
    sample_sweep = []
    for rate in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005]:
        row = {"sample_rate": rate, **measure(MicroProfiler(sample_rate=rate))}
        sample_sweep.append(row)
        print(f"    rate={rate:<6} overhead={row['overhead_cycles']:>6}c ({row['overhead_pct']:.2f}%) "
              f"oracle-match={row['oracle_agreement']}/{row['tasks']} efficiency={row['oracle_efficiency_pct']:.1f}%")

    print("\n  Profiling-cost sweep (full trace, sample_rate=1.0):")
    cost_sweep = []
    for cps in [0.5, 2, 8, 32, 64, 128]:
        row = {"cycles_per_sample": cps, **measure(MicroProfiler(sample_rate=1.0, cycles_per_sample=cps))}
        row["beats_best_static"] = row["cycles"] < best_static
        cost_sweep.append(row)
        print(f"    {cps:>5} c/sample overhead={row['overhead_cycles']:>9,}c ({row['overhead_pct']:5.1f}%) "
              f"speedup={row['speedup']:.2f}x beats best static/roofline: {row['beats_best_static']}")

    # Prediction error of the cost model for the placement it actually chose
    errors = []
    for t in mixed["DYNAMIC_ANALYTICAL"]["tasks"]:
        predicted = t["predicted_pim_cycles"] if t["target"] == "PIM" else t["predicted_host_cycles"]
        actual = t["cycles"] - t["overhead_cycles"]
        errors.append({
            "task": t["name"],
            "target": t["target"],
            "predicted": predicted,
            "actual": actual,
            "error_pct": round(100.0 * (predicted - actual) / max(1, actual), 1),
        })
    print("\n  Cost-model prediction error (chosen target):")
    for e in errors:
        print(f"    {e['task']:<28} {e['target']:<4} predicted={e['predicted']:>9,} actual={e['actual']:>9,} error={e['error_pct']:+.1f}%")

    return {
        "workload": MIXED,
        "oracle_cycles": oracle_cycles,
        "host_cycles": host_cycles,
        "best_static_or_roofline_cycles": best_static,
        "sample_rate_sweep": sample_sweep,
        "profiling_cost_sweep": cost_sweep,
        "prediction_error": errors,
    }


def export_dashboard_data(results: Dict[str, Any], study: Dict[str, Any]):
    """Writes web/data.js so the dashboard shows exactly what the simulator produced."""
    dashboard = {}
    for wl, by_policy in results.items():
        dashboard[wl] = {}
        for pol, r in by_policy.items():
            dashboard[wl][pol] = {
                "speedup": r["speedup"],
                "dataSaved": r["data_reduction_pct"],
                "edpGain": r["edp_reduction_pct"],
                "oracleEfficiency": r["oracle_efficiency_pct"],
                "overheadPct": r["decision_overhead_pct"],
                "hostTasks": r["decisions"]["placed_on_host"],
                "pimTasks": r["decisions"]["placed_on_pim"],
                "busMovedKB": r["data_moved_kb"],
                "desc": WORKLOAD_DESCRIPTIONS.get(wl, ""),
                "tasks": [
                    {
                        "name": t["name"],
                        "category": t["category"],
                        "target": t["target"],
                        "cycles": t["cycles"],
                        "dataMoved": f"{t['bytes'] / 1024:.2f} KB",
                        "overhead": f"{t['overhead_cycles']}c",
                        "reason": t["reason"],
                    }
                    for t in r["tasks"]
                ],
            }
    web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")
    path = os.path.join(web_dir, "data.js")
    with open(path, "w") as f:
        f.write("// Generated by benchmarks/run_experiments.py - do not edit by hand.\n")
        f.write("window.MEMCENTRIC_RESULTS = ")
        json.dump(dashboard, f, indent=1)
        f.write(";\nwindow.MEMCENTRIC_OVERHEAD_STUDY = ")
        json.dump(study, f, indent=1)
        f.write(";\n")
    print(f"Dashboard data exported to: {os.path.abspath(path)}")


if __name__ == "__main__":
    run_all_experiments()
