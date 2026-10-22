"""
MemCentric-Sim: Micro-Profiler for Runtime Task Analysis
=========================================================
Extracts key data-movement and computation metrics from kernels:
- Working Set Size (Bytes)
- Operational Intensity (FLOPs / Byte)
- Memory Access Stride & Spatial Locality
- Temporal Reuse Distance Estimation
- Reduction Ratio (Output Bytes / Input Bytes)
"""

from dataclasses import dataclass
from typing import List, Tuple, Set


@dataclass
class KernelProfile:
    name: str
    num_ops: int
    input_bytes: int
    output_bytes: int
    unique_lines: int
    operational_intensity: float  # FLOPs per Byte
    temporal_reuse_factor: float  # Average accesses per unique line
    reduction_ratio: float        # output_bytes / input_bytes
    is_sequential_stride: bool


class MicroProfiler:
    """
    Lightweight hardware/runtime profiler modeling metadata sampling overhead.
    """
    def __init__(self, sample_rate: float = 1.0, profiling_latency_cycles: int = 25):
        self.sample_rate = sample_rate
        self.profiling_latency_cycles = profiling_latency_cycles

    def profile(
        self,
        name: str,
        num_ops: int,
        memory_accesses: List[Tuple[int, bool]],
        output_bytes: int,
        line_size_bytes: int = 64
    ) -> KernelProfile:
        """
        Profiles memory access trace to extract data-local characteristics.
        """
        total_accesses = len(memory_accesses)
        if total_accesses == 0:
            return KernelProfile(
                name=name,
                num_ops=num_ops,
                input_bytes=0,
                output_bytes=output_bytes,
                unique_lines=0,
                operational_intensity=float("inf"),
                temporal_reuse_factor=0.0,
                reduction_ratio=0.0,
                is_sequential_stride=True
            )

        unique_lines: Set[int] = set()
        strides: List[int] = []
        prev_addr: int = -1

        for addr, _ in memory_accesses:
            line_addr = addr // line_size_bytes
            unique_lines.add(line_addr)
            if prev_addr != -1:
                strides.append(abs(addr - prev_addr))
            prev_addr = addr

        total_unique_lines = len(unique_lines)
        input_bytes = total_unique_lines * line_size_bytes

        # Operational intensity: FLOPs / Byte of unique data footprint
        oi = num_ops / max(1, input_bytes)

        # Temporal reuse: how many accesses per unique cache line
        reuse = total_accesses / max(1, total_unique_lines)

        # Stride analysis: check if dominant stride is cache line sequential
        seq_strides = sum(1 for s in strides if 0 <= s <= line_size_bytes)
        is_sequential = (seq_strides / len(strides)) > 0.7 if strides else True

        # Reduction ratio: small ratio indicates massive bandwidth savings if computed near memory
        red_ratio = output_bytes / max(1, input_bytes)

        return KernelProfile(
            name=name,
            num_ops=num_ops,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            unique_lines=total_unique_lines,
            operational_intensity=oi,
            temporal_reuse_factor=reuse,
            reduction_ratio=red_ratio,
            is_sequential_stride=is_sequential
        )
