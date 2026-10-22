"""
MemCentric-Sim Workload: Dense GEMM (General Matrix Multiply)
=============================================================
Characteristics:
- High Temporal Locality & Cache Reuse: Elements of A and B are reused N times.
- High Operational Intensity: O(N^3) operations / O(N^2) memory footprint.
- Compute-Bound: Benefits substantially from Host CPU wide SIMD/AVX and fast L1/L2 caches.
"""

from typing import List, Tuple
from memcentric.workloads.base import KernelTask


def generate_gemm_task(matrix_dim: int = 64, base_addr: int = 0x100000) -> KernelTask:
    """
    Generates memory access trace for C = A * B.
    For matrix_dim = 64:
      - Matrices A, B, C of size 64x64 (4 bytes per float) = 16 KB each.
      - Total compute ops = 2 * 64^3 = 524,288 FLOPs.
      - Highly reusable across inner loops.
    """
    elem_size = 4
    matrix_bytes = matrix_dim * matrix_dim * elem_size
    addr_a = base_addr
    addr_b = base_addr + matrix_bytes
    addr_c = base_addr + (2 * matrix_bytes)

    accesses: List[Tuple[int, bool]] = []
    num_ops = 2 * (matrix_dim ** 3)

    # Sample loop nest trace (tiled/blocked representation)
    # We sample a representative sequence to keep simulation responsive while capturing reuse
    step = max(1, matrix_dim // 16)
    for i in range(0, matrix_dim, step):
        for j in range(0, matrix_dim, step):
            for k in range(0, matrix_dim, step):
                # Read A[i, k]
                accesses.append((addr_a + (i * matrix_dim + k) * elem_size, False))
                # Read B[k, j]
                accesses.append((addr_b + (k * matrix_dim + j) * elem_size, False))
            # Write C[i, j]
            accesses.append((addr_c + (i * matrix_dim + j) * elem_size, True))

    output_bytes = matrix_bytes

    return KernelTask(
        name=f"GEMM_{matrix_dim}x{matrix_dim}",
        category="HIGH_REUSE",
        num_ops=num_ops,
        memory_accesses=accesses,
        output_bytes=output_bytes,
        description=f"Dense Matrix-Matrix Multiplication ({matrix_dim}x{matrix_dim}) with high temporal cache reuse"
    )
