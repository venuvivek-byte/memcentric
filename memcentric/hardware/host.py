"""
MemCentric-Sim: Host Compute Unit (CPU / GPU Host)
===================================================
Models high-performance Host Processor with:
- Multi-issue superscalar / vector compute pipeline
- High clock frequency (e.g., 3.2 GHz)
- Deep multi-level cache hierarchy (L1/L2/L3)
- Cache misses propagate through Interconnect to DRAM
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from memcentric.hardware.cache import CacheHierarchy
from memcentric.hardware.interconnect import Interconnect
from memcentric.hardware.memory import MemorySubsystem


@dataclass
class HostConfig:
    name: str = "Host-CPU"
    frequency_ghz: float = 3.2
    num_cores: int = 4            # Multi-core Host
    ipc: float = 2.0              # Peak instructions per cycle per core
    vector_width_lanes: int = 8   # SIMD width (e.g., AVX-512)
    active_power_watts: float = 65.0
    idle_power_watts: float = 10.0


class HostProcessor:
    """
    Host Processing Unit coupled with a multi-level cache hierarchy
    and connected to DRAM via the external Interconnect.
    """
    def __init__(
        self,
        config: Optional[HostConfig] = None,
        cache_hierarchy: Optional[CacheHierarchy] = None,
        interconnect: Optional[Interconnect] = None,
        memory: Optional[MemorySubsystem] = None
    ):
        self.config = config or HostConfig()
        self.cache = cache_hierarchy or CacheHierarchy()
        self.interconnect = interconnect or Interconnect()
        self.memory = memory or MemorySubsystem()

        # Stats
        self.total_compute_cycles: int = 0
        self.total_memory_stall_cycles: int = 0
        self.total_instructions: int = 0
        self.total_energy_pj: float = 0.0

    def execute_compute(self, num_ops: int, is_vector: bool = False) -> int:
        """
        Calculates compute execution cycles on Host.
        Vector instructions utilize vector_width_lanes across cores.
        """
        lanes = self.config.vector_width_lanes if is_vector else 1.0
        throughput = self.config.num_cores * self.config.ipc * lanes
        compute_cycles = max(1, int(num_ops / throughput))
        self.total_compute_cycles += compute_cycles
        self.total_instructions += num_ops

        # Dynamic energy = Power * Time = Watts * (cycles / freq_hz) * 10^12 pJ
        time_seconds = compute_cycles / (self.config.frequency_ghz * 1e9)
        energy_pj = self.config.active_power_watts * time_seconds * 1e12
        self.total_energy_pj += energy_pj

        return compute_cycles

    def access_memory(self, addr: int, is_write: bool, current_cycle: int) -> Tuple[int, int]:
        """
        Executes a memory access from Host:
        1. Query cache hierarchy (L1 -> L2 -> L3).
        2. On cache hit: Return cache latency & energy.
        3. On L3 miss:
           - Send request packet over Interconnect
           - Access DRAM bank
           - Send response data packet (64 bytes) over Interconnect
           - Install in cache
        Returns: (completion_cycle, latency_cycles)
        """
        cache_lat, cache_energy, full_miss = self.cache.access(addr, is_write, current_cycle)
        self.total_energy_pj += cache_energy

        if not full_miss:
            # Hit in L1, L2, or L3
            return current_cycle + cache_lat, cache_lat

        # Cache Miss: traverse interconnect to DRAM
        stall_start = current_cycle + cache_lat

        # 1. Transmit 8-byte request packet to memory controller
        tx_req_cycle, _ = self.interconnect.transmit(size_bytes=8, current_cycle=stall_start)

        # 2. Access DRAM
        dram_finish, dram_lat, dram_energy = self.memory.access(
            addr=addr, is_write=is_write, current_cycle=tx_req_cycle
        )

        # 3. Transmit 64-byte cache line response back to Host
        tx_resp_cycle, _ = self.interconnect.transmit(size_bytes=64, current_cycle=dram_finish)

        total_latency = (tx_resp_cycle - current_cycle)
        self.total_memory_stall_cycles += total_latency

        return tx_resp_cycle, total_latency

    def reset(self):
        self.total_compute_cycles = 0
        self.total_memory_stall_cycles = 0
        self.total_instructions = 0
        self.total_energy_pj = 0.0
        self.cache.reset_stats()

    def get_stats(self) -> Dict[str, float]:
        return {
            "compute_cycles": self.total_compute_cycles,
            "memory_stall_cycles": self.total_memory_stall_cycles,
            "total_cycles": self.total_compute_cycles + self.total_memory_stall_cycles,
            "total_energy_uj": round(self.total_energy_pj / 1e6, 3),
            "cache_stats": self.cache.get_summary(),
        }
