import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memcentric.system import MemoryCentricSystem
from memcentric.runtime.policy import PlacementPolicy
from memcentric.runtime.profiler import MicroProfiler
from memcentric.hardware.cache import CacheHierarchy
from memcentric.workloads.streaming import generate_streaming_vector_task
from memcentric.workloads.mixed import (
    generate_dense_transform_task,
    generate_feature_cross_task,
    generate_mixed_dataset_session,
)


def test_execution_runs_through_event_queue():
    system = MemoryCentricSystem(policy=PlacementPolicy.DYNAMIC_ANALYTICAL)
    tasks = [generate_streaming_vector_task(num_elements=4096), generate_feature_cross_task(num_lines=128)]
    result = system.run_workload(tasks)
    # arrival + dispatch + completion per task
    assert result.events_processed == 3 * len(tasks)
    assert result.total_cycles == system.task_records[-1].end_cycle
    assert system.task_records[1].start_cycle == system.task_records[0].end_cycle


def test_cache_probe_has_no_side_effects_and_invalidate_reports_dirty():
    cache = CacheHierarchy()
    cache.access(0x1000, is_write=True, cycle=1)
    hits_before = cache.l1d.stats.hits
    assert cache.probe_line(0x1000) == (True, True)
    assert cache.l1d.stats.hits == hits_before
    assert cache.invalidate_line(0x1000) is True
    assert cache.probe_line(0x1000) == (False, False)


def test_hot_dirty_data_stays_on_host_and_offload_pays_coherence():
    produce = generate_dense_transform_task(num_elements=8192, base_addr=0x5000000)
    consume = generate_streaming_vector_task(num_elements=4096, base_addr=0x5000000)

    dyn = MemoryCentricSystem(policy=PlacementPolicy.DYNAMIC_ANALYTICAL)
    dyn.run_workload([produce, consume])
    assert dyn.task_records[1].target == "HOST"

    roof = MemoryCentricSystem(policy=PlacementPolicy.ROOFLINE_OI)
    roof.run_workload([produce, consume])
    assert roof.task_records[1].target == "PIM"
    assert roof.task_records[1].coherence_writebacks > 0
    assert roof.task_records[1].elapsed_cycles > dyn.task_records[1].elapsed_cycles


def test_same_kernel_moves_to_pim_when_bus_is_congested():
    quiet = generate_feature_cross_task(base_addr=0x9000000)
    busy = generate_feature_cross_task(base_addr=0xA000000)
    busy.interfering_dma_bytes = 1024 * 1024

    system = MemoryCentricSystem(policy=PlacementPolicy.DYNAMIC_ANALYTICAL)
    system.run_workload([quiet, busy])
    assert system.task_records[0].target == "HOST"
    assert system.task_records[0].bus_backlog_at_arrival == 0
    assert system.task_records[1].bus_backlog_at_arrival > 0
    assert system.task_records[1].target == "PIM"
    # Background DMA traffic is not attributed to the workload
    assert system.interconnect.background_bytes_transferred == 1024 * 1024


def test_decision_overhead_scales_with_profiling_and_is_zero_for_static():
    task = generate_feature_cross_task()
    overheads = []
    for rate in (1.0, 0.1):
        system = MemoryCentricSystem(profiler=MicroProfiler(sample_rate=rate))
        overheads.append(system.run_task(task).decision_overhead_cycles)
    assert overheads[0] > 3 * overheads[1] > 0

    static = MemoryCentricSystem(policy=PlacementPolicy.STATIC_PIM)
    assert static.run_task(task).decision_overhead_cycles == 0


def test_sampled_working_set_estimate():
    task = generate_feature_cross_task(num_lines=1024, passes=2)
    profile = MicroProfiler(sample_rate=0.1).profile(
        task.name, task.num_ops, task.memory_accesses, task.output_bytes
    )
    assert 0.75 * 1024 <= profile.unique_lines <= 1.25 * 1024


def test_dynamic_matches_oracle_on_mixed_session():
    tasks = generate_mixed_dataset_session()
    dyn = MemoryCentricSystem(policy=PlacementPolicy.DYNAMIC_ANALYTICAL).run_workload(tasks)
    oracle = MemoryCentricSystem(policy=PlacementPolicy.ORACLE).run_workload(tasks)
    roof = MemoryCentricSystem(policy=PlacementPolicy.ROOFLINE_OI).run_workload(tasks)
    assert [t.target for t in dyn.task_records] == [t.target for t in oracle.task_records]
    assert dyn.total_cycles < roof.total_cycles
    assert oracle.total_cycles <= dyn.total_cycles


if __name__ == "__main__":
    test_execution_runs_through_event_queue()
    test_cache_probe_has_no_side_effects_and_invalidate_reports_dirty()
    test_hot_dirty_data_stays_on_host_and_offload_pays_coherence()
    test_same_kernel_moves_to_pim_when_bus_is_congested()
    test_decision_overhead_scales_with_profiling_and_is_zero_for_static()
    test_sampled_working_set_estimate()
    test_dynamic_matches_oracle_on_mixed_session()
    print("Placement dynamics tests passed successfully!")
