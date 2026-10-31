"""
MemCentric-Sim: Main Entrypoint & Command-Line Interface
=========================================================
A comprehensive simulation framework for Memory-Centric Computing,
Near-Memory Processing (PIM), and Intelligent Runtime Computation Placement.

Usage:
  python main.py                  # Run default simulation & generate summary
  python main.py --run-all        # Run exhaustive benchmark suite across all policies
  python main.py --plot           # Generate publication-grade figures
  python main.py --test           # Run unit and integration tests
  python main.py --serve [PORT]   # Launch interactive visualizer dashboard (default: 8080)
"""

import os
import sys
import argparse
import http.server
import socketserver
import webbrowser

from memcentric.system import MemoryCentricSystem
from memcentric.runtime.policy import PlacementPolicy
from memcentric.workloads.mixed import generate_mixed_dataset_session
from benchmarks.run_experiments import run_all_experiments
from visualizer.plots import generate_all_plots


def run_pipeline_demo():
    print("=" * 80)
    print(" MEMCENTRIC-SIM: MEMORY-CENTRIC COMPUTING ARCHITECTURE DEMO")
    print("=" * 80)
    print("Evaluating Mixed-Dataset Session (reusable + streaming + random data, one system)...")

    # 1. Baseline Host Run
    host_system = MemoryCentricSystem(policy=PlacementPolicy.STATIC_HOST)
    host_tasks = generate_mixed_dataset_session()
    host_res = host_system.run_workload(host_tasks)

    # 2. Dynamic MemCentric Run
    dynamic_system = MemoryCentricSystem(policy=PlacementPolicy.DYNAMIC_ANALYTICAL)
    dyn_tasks = generate_mixed_dataset_session()
    dyn_res = dynamic_system.run_workload(dyn_tasks)

    speedup = host_res.total_cycles / max(1, dyn_res.total_cycles)
    data_reduction = (
        (host_res.total_bytes_transferred - dyn_res.total_bytes_transferred)
        / max(1, host_res.total_bytes_transferred)
    ) * 100.0
    energy_saving = (
        (host_res.total_energy_uj - dyn_res.total_energy_uj) / max(1, host_res.total_energy_uj)
    ) * 100.0
    edp_gain = (
        (host_res.edp_joule_sec - dyn_res.edp_joule_sec) / max(1e-12, host_res.edp_joule_sec)
    ) * 100.0

    print("\n" + "-" * 80)
    print(f"| {'METRIC':<30} | {'HOST BASELINE':<18} | {'MEMCENTRIC DYNAMIC':<22} |")
    print("-" * 80)
    print(f"| {'Total Execution Cycles':<30} | {host_res.total_cycles:<18,d} | {dyn_res.total_cycles:<22,d} |")
    print(f"| {'Execution Speedup':<30} | {'1.00x':<18} | {f'{speedup:.2f}x':<22} |")
    print(f"| {'Data Moved Over Interconnect':<30} | {f'{host_res.total_bytes_transferred / 1024:.1f} KB':<18} | {f'{dyn_res.total_bytes_transferred / 1024:.1f} KB':<22} |")
    print(f"| {'Interconnect Traffic Saved':<30} | {'0.0%':<18} | {f'{data_reduction:.1f}%':<22} |")
    print(f"| {'Total System Energy':<30} | {f'{host_res.total_energy_uj / 1e3:.2f} mJ':<18} | {f'{dyn_res.total_energy_uj / 1e3:.2f} mJ':<22} |")
    print(f"| {'Energy Savings':<30} | {'0.0%':<18} | {f'{energy_saving:.1f}%':<22} |")
    print(f"| {'Energy-Delay Product (EDP)':<30} | {f'{host_res.edp_joule_sec:.6e}':<18} | {f'{dyn_res.edp_joule_sec:.6e}':<22} |")
    print(f"| {'EDP Improvement':<30} | {'0.0%':<18} | {f'{edp_gain:.1f}%':<22} |")
    print("-" * 80)

    print("\nTASK-LEVEL DYNAMIC PLACEMENT BREAKDOWN:")
    for t in dyn_res.task_records:
        target_color = "PIM (Near-Memory)" if t.target == "PIM" else "HOST (Vector CPU)"
        print(f"  * {t.task_name:<32} -> Placed on {target_color:<18} (Data Moved: {t.bytes_transferred / 1024.0:.2f} KB, Cycles: {t.elapsed_cycles:,})")
        print(f"    Reason: {t.decision_reason}")
    print(f"\nDecision overhead: {dyn_res.decisions_summary['total_overhead_cycles']:,} cycles "
          f"({100.0 * dyn_res.decisions_summary['total_overhead_cycles'] / dyn_res.total_cycles:.2f}% of runtime), "
          f"{dyn_res.events_processed} simulation events")
    print("\n" + "=" * 80)


def run_tests():
    import subprocess
    print("Running test suite...")
    test_files = [
        "tests/test_cache.py",
        "tests/test_memory.py",
        "tests/test_interconnect.py",
        "tests/test_runtime.py",
        "tests/test_system.py",
        "tests/test_placement_dynamics.py",
    ]
    all_passed = True
    for t in test_files:
        res = subprocess.run([sys.executable, t])
        if res.returncode != 0:
            all_passed = False
            print(f"[FAIL] Test failed: {t}")
    if all_passed:
        print("\n[PASS] All unit and integration tests passed successfully!")
    return all_passed


def serve_dashboard(port: int = 8080):
    web_dir = os.path.join(os.path.dirname(__file__), "web")
    os.chdir(web_dir)
    handler = http.server.SimpleHTTPRequestHandler
    
    print(f"\n🚀 Launching MemCentric-Sim Interactive Dashboard on: http://localhost:{port}")
    print("Press Ctrl+C to terminate.")
    
    with socketserver.TCPServer(("", port), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


def main():
    parser = argparse.ArgumentParser(description="MemCentric-Sim: Memory-Centric Computing Architecture Simulator")
    parser.add_argument("--run-all", action="store_true", help="Run comprehensive benchmark suite across all workloads and policies")
    parser.add_argument("--plot", action="store_true", help="Generate publication-grade figures")
    parser.add_argument("--test", action="store_true", help="Run test suite")
    parser.add_argument("--serve", type=int, nargs="?", const=8080, help="Launch web visualizer server on PORT (default: 8080)")

    args = parser.parse_args()

    if args.run_all:
        run_all_experiments()
        generate_all_plots()
    elif args.plot:
        generate_all_plots()
    elif args.test:
        run_tests()
    elif args.serve is not None:
        serve_dashboard(args.serve)
    else:
        run_pipeline_demo()


if __name__ == "__main__":
    main()
