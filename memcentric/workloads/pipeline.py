"""
MemCentric-Sim Workload: Dynamic Heterogeneous Pipeline
========================================================
Demonstrates runtime computation placement switching across stages of an
end-to-end data processing and machine learning pipeline:
  - Stage 1: Raw Table Filtering & Scan (Streaming) -> Best on PIM
  - Stage 2: Sparse Graph Feature Gather (Random Access) -> Best on PIM
  - Stage 3: High-Intensity Feature Cross & GEMM (High Reuse) -> Best on Host CPU
  - Stage 4: Global Metrics Accumulator (Streaming Reduction) -> Best on PIM
"""

from typing import List
from memcentric.workloads.base import KernelTask
from memcentric.workloads.gemm import generate_gemm_task
from memcentric.workloads.streaming import generate_db_filter_scan_task, generate_streaming_vector_task
from memcentric.workloads.graph import generate_graph_bfs_task, generate_sparse_embedding_lookup_task


def generate_heterogeneous_pipeline() -> List[KernelTask]:
    """
    Constructs a 4-stage pipeline simulating an end-to-end Analytics & Inference pipeline.
    """
    stage1 = generate_db_filter_scan_task(num_records=40000, base_addr=0x1000000)
    stage1.name = "Pipeline_Stage1_LogFilter"
    stage1.description = "Stage 1: Streaming log filter and record predicate selection"

    stage2 = generate_sparse_embedding_lookup_task(table_entries=15000, lookup_count=1200, base_addr=0x2000000)
    stage2.name = "Pipeline_Stage2_EntityLookup"
    stage2.description = "Stage 2: Random access sparse feature and embedding retrieval"

    stage3 = generate_gemm_task(matrix_dim=64, base_addr=0x3000000)
    stage3.name = "Pipeline_Stage3_MLInference"
    stage3.description = "Stage 3: Compute-heavy Matrix multiplication for neural feature transformation"

    stage4 = generate_streaming_vector_task(num_elements=16384, base_addr=0x4000000)
    stage4.name = "Pipeline_Stage4_GlobalReduction"
    stage4.description = "Stage 4: Final metric aggregation and summary reduction"

    return [stage1, stage2, stage3, stage4]
