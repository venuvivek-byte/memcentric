import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memcentric.runtime.profiler import MicroProfiler
from memcentric.runtime.cost_model import AnalyticalCostModel
from memcentric.runtime.policy import PlacementEngine, PlacementPolicy
from memcentric.workloads.gemm import generate_gemm_task
from memcentric.workloads.streaming import generate_streaming_vector_task


def test_profiler_and_cost_model():
    profiler = MicroProfiler()
    cost_model = AnalyticalCostModel()

    # Profile GEMM (High reuse, high operational intensity)
    gemm_task = generate_gemm_task(matrix_dim=32)
    gemm_profile = profiler.profile(
        name=gemm_task.name,
        num_ops=gemm_task.num_ops,
        memory_accesses=gemm_task.memory_accesses,
        output_bytes=gemm_task.output_bytes
    )
    assert gemm_profile.operational_intensity > 2.0
    gemm_est = cost_model.estimate(gemm_profile)
    # High reuse should favor HOST
    assert gemm_est.recommended_placement == "HOST"

    # Profile Streaming task (Low reuse, streaming, low OI)
    stream_task = generate_streaming_vector_task(num_elements=8192)
    stream_profile = profiler.profile(
        name=stream_task.name,
        num_ops=stream_task.num_ops,
        memory_accesses=stream_task.memory_accesses,
        output_bytes=stream_task.output_bytes
    )
    assert stream_profile.operational_intensity < 1.0
    stream_est = cost_model.estimate(stream_profile)
    # Streaming reduction should favor PIM
    assert stream_est.recommended_placement == "PIM"


def test_placement_engine_dynamic_decisions():
    engine = PlacementEngine(policy=PlacementPolicy.DYNAMIC_ANALYTICAL)

    gemm_task = generate_gemm_task(matrix_dim=32)
    d1 = engine.decide(
        kernel_name=gemm_task.name,
        num_ops=gemm_task.num_ops,
        memory_accesses=gemm_task.memory_accesses,
        output_bytes=gemm_task.output_bytes
    )
    assert d1.target == "HOST"

    stream_task = generate_streaming_vector_task(num_elements=8192)
    d2 = engine.decide(
        kernel_name=stream_task.name,
        num_ops=stream_task.num_ops,
        memory_accesses=stream_task.memory_accesses,
        output_bytes=stream_task.output_bytes
    )
    assert d2.target == "PIM"


if __name__ == "__main__":
    test_profiler_and_cost_model()
    test_placement_engine_dynamic_decisions()
    print("Runtime tests passed successfully!")
