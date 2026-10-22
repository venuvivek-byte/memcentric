"""
MemCentric-Sim Workload: Streaming Vector / Database Filter & Aggregate
========================================================================
Characteristics:
- Zero/Minimal Temporal Locality: Each element visited once.
- Low Operational Intensity: ~0.125 to 0.25 FLOPs/Byte.
- Memory Bandwidth Bound: CXL bus quickly becomes bottleneck if fetched to Host.
- High Reduction Ratio: Output is a small filtered table or a scalar aggregate (8 bytes).
- Near-Memory Computing Ideal Target.
"""

from typing import List, Tuple
from memcentric.workloads.base import KernelTask


def generate_streaming_vector_task(num_elements: int = 32768, base_addr: int = 0x400000) -> KernelTask:
    """
    Simulates streaming vector addition & scalar reduction:
    Result = SUM(A[i] * alpha + B[i]) for i in 0..N
    """
    elem_size = 4
    vector_bytes = num_elements * elem_size
    addr_a = base_addr
    addr_b = base_addr + vector_bytes

    accesses: List[Tuple[int, bool]] = []
    # Sample linearly across cache lines
    step = 16 # Every cache line (64 bytes = 16 floats)
    for i in range(0, num_elements, step):
        accesses.append((addr_a + (i * elem_size), False))
        accesses.append((addr_b + (i * elem_size), False))

    num_ops = 2 * num_elements
    output_bytes = 8 # Single 64-bit scalar sum returned to Host!

    return KernelTask(
        name=f"Streaming_Vector_Reduction_{num_elements}",
        category="STREAMING",
        num_ops=num_ops,
        memory_accesses=accesses,
        output_bytes=output_bytes,
        description=f"Linear Streaming Vector Reduction ({num_elements} elements, low FLOP/Byte, small reduction output)"
    )


def generate_db_filter_scan_task(num_records: int = 50000, base_addr: int = 0x800000) -> KernelTask:
    """
    Simulates relational database column scan and predicate filter:
    SELECT SUM(salary) FROM Employees WHERE dept_id == 42;
    """
    record_size = 8
    total_bytes = num_records * record_size
    accesses: List[Tuple[int, bool]] = []

    # Sequential scan across records
    step = 8 # Every 64 bytes
    for i in range(0, num_records, step):
        accesses.append((base_addr + (i * record_size), False))

    num_ops = num_records # Comparison + accumulate
    output_bytes = 8      # Aggregated scalar sum

    return KernelTask(
        name=f"DB_Scan_Filter_{num_records}",
        category="STREAMING",
        num_ops=num_ops,
        memory_accesses=accesses,
        output_bytes=output_bytes,
        description=f"Database Column Filter & Aggregate Scan ({num_records} records, pure streaming)"
    )
