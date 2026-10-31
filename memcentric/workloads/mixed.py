"""
MemCentric-Sim Workload: Mixed-Dataset Session
===============================================
One system, one continuous run, three kinds of data:
  - a highly reusable dataset (weights, a freshly produced buffer that is hot
    and dirty in the host caches),
  - streaming datasets (cold table scan, aggregation over the hot buffer),
  - randomly accessed datasets (embedding gather, graph frontier),
plus a phase that runs while another device floods the shared interconnect.

The right placement for a phase depends on runtime state, not only on the
kernel's static operational intensity:
  - Streaming over data that is already in the host caches should stay on the
    host; offloading it forces a coherence write-back of every dirty line.
  - The same moderate-intensity kernel belongs on the host when the bus is
    quiet and near memory when the bus is congested.
"""

from typing import List, Tuple
from memcentric.workloads.base import KernelTask
from memcentric.workloads.gemm import generate_gemm_task
from memcentric.workloads.streaming import generate_streaming_vector_task, generate_db_filter_scan_task
from memcentric.workloads.graph import generate_graph_bfs_task, generate_sparse_embedding_lookup_task


def generate_dense_transform_task(
    num_elements: int = 65536,
    ops_per_element: int = 128,
    base_addr: int = 0x5000000,
    weights_addr: int = 0x5F00000,
    weights_lines: int = 64
) -> KernelTask:
    """
    Producer kernel: applies a small, hot weight table to generate a new
    buffer (e.g. feature encoding). Compute-heavy and it writes every output
    line, leaving the buffer resident and dirty in the host caches.
    """
    elem_size = 4
    out_lines = (num_elements * elem_size) // 64
    accesses: List[Tuple[int, bool]] = []
    for line in range(out_lines):
        accesses.append((weights_addr + (line % weights_lines) * 64, False))
        accesses.append((base_addr + line * 64, True))

    return KernelTask(
        name=f"Dense_Transform_{num_elements}",
        category="HIGH_REUSE",
        num_ops=num_elements * ops_per_element,
        memory_accesses=accesses,
        output_bytes=0,  # result stays in the produced buffer
        description=f"Producer: encodes {num_elements} elements with a hot weight table ({ops_per_element} ops/element)"
    )


def generate_feature_cross_task(
    num_lines: int = 1024,
    ops_per_line: int = 700,
    passes: int = 2,
    base_addr: int = 0x9000000,
    output_bytes: int = 256
) -> KernelTask:
    """
    Moderate-intensity kernel (~11 FLOP/B) over cold data. On a quiet bus the
    host's SIMD units win; when the bus is congested near-memory execution wins.
    """
    accesses: List[Tuple[int, bool]] = []
    for _ in range(passes):
        for line in range(num_lines):
            accesses.append((base_addr + line * 64, False))

    return KernelTask(
        name=f"Feature_Cross_{num_lines}L",
        category="HIGH_REUSE",
        num_ops=num_lines * ops_per_line,
        memory_accesses=accesses,
        output_bytes=output_bytes,
        description=f"Feature crossing over {num_lines * 64 // 1024} KB of cold data, {passes} passes"
    )


def generate_mixed_dataset_session(dma_burst_bytes: int = 1024 * 1024) -> List[KernelTask]:
    buffer_addr = 0x5000000
    n_elems = 32768  # streaming task reads two vectors of this length = the produced buffer

    produce = generate_dense_transform_task(num_elements=2 * n_elems, base_addr=buffer_addr)
    produce.name = "S1_Ingest_Transform"
    produce.description = "Reusable: hot weights generate a 256 KB buffer (left dirty in host caches)"

    hot_stream = generate_streaming_vector_task(num_elements=n_elems, base_addr=buffer_addr)
    hot_stream.name = "S2_HotBuffer_Aggregate"
    hot_stream.description = "Streaming over the buffer S1 just produced - data is already in the host caches"

    cold_scan = generate_db_filter_scan_task(num_records=50000, base_addr=0x6000000)
    cold_scan.name = "S3_ColdTable_Scan"
    cold_scan.description = "Streaming scan of a cold table that lives only in DRAM"

    gather = generate_sparse_embedding_lookup_task(table_entries=20000, lookup_count=1000, base_addr=0x7000000)
    gather.name = "S4_Embedding_Gather"
    gather.description = "Random access gather from a cold embedding table"

    gemm = generate_gemm_task(matrix_dim=64, base_addr=0x8000000)
    gemm.name = "S5_Model_GEMM"
    gemm.description = "Reusable: dense GEMM with high cache reuse"

    cross_quiet = generate_feature_cross_task(base_addr=0x9000000)
    cross_quiet.name = "S6_FeatureCross_QuietBus"
    cross_quiet.description = "Moderate intensity kernel, interconnect idle"

    cross_busy = generate_feature_cross_task(base_addr=0xA000000)
    cross_busy.name = "S7_FeatureCross_BusyBus"
    cross_busy.description = f"Same kernel while a {dma_burst_bytes // 1024} KB storage DMA occupies the interconnect"
    cross_busy.interfering_dma_bytes = dma_burst_bytes

    bfs = generate_graph_bfs_task(num_nodes=8000, edges_per_node=6, frontier_size=400, base_addr=0xB000000)
    bfs.name = "S8_Graph_Frontier"
    bfs.description = "Random access pointer chasing over a cold graph"

    return [produce, hot_stream, cold_scan, gather, gemm, cross_quiet, cross_busy, bfs]
