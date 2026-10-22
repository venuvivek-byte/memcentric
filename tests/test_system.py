import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memcentric.system import MemoryCentricSystem
from memcentric.runtime.policy import PlacementPolicy
from memcentric.workloads.gemm import generate_gemm_task
from memcentric.workloads.streaming import generate_streaming_vector_task
from memcentric.workloads.graph import generate_graph_bfs_task


def test_system_end_to_end_simulation():
    system = MemoryCentricSystem(policy=PlacementPolicy.DYNAMIC_ANALYTICAL)

    tasks = [
        generate_gemm_task(matrix_dim=32),
        generate_streaming_vector_task(num_elements=4096),
        generate_graph_bfs_task(num_nodes=1000, edges_per_node=4, frontier_size=100)
    ]

    result = system.run_workload(tasks)
    assert result.total_cycles > 0
    assert result.total_energy_uj > 0
    assert len(result.task_records) == 3

    # Check that GEMM went to HOST and Streaming went to PIM
    gemm_record = [r for r in result.task_records if "GEMM" in r.task_name][0]
    stream_record = [r for r in result.task_records if "Streaming" in r.task_name][0]

    assert gemm_record.target == "HOST"
    assert stream_record.target == "PIM"

    # Verify decision overhead was accounted for
    assert result.decisions_summary["total_overhead_cycles"] > 0
    assert result.decisions_summary["placed_on_host"] >= 1
    assert result.decisions_summary["placed_on_pim"] >= 1


if __name__ == "__main__":
    test_system_end_to_end_simulation()
    print("System end-to-end tests passed successfully!")
