"""
MemCentric-Sim: Workload & Task Abstraction
============================================
Defines the structure for simulated kernels and memory access patterns.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class KernelTask:
    name: str
    category: str               # "HIGH_REUSE", "STREAMING", "RANDOM_ACCESS", "HETEROGENEOUS"
    num_ops: int                # Total ALU / floating-point operations
    memory_accesses: List[Tuple[int, bool]]  # List of (byte_address, is_write)
    output_bytes: int           # Size of final produced result
    description: str = ""

    @property
    def total_memory_accesses(self) -> int:
        return len(self.memory_accesses)
