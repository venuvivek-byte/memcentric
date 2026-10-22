"""
MemCentric-Sim: Near-Memory Computing (PIM / NMC) Model
========================================================
Models Processing-In-Memory (PIM) units placed near DRAM banks:
- Direct, high-bandwidth internal bus to DRAM banks
- Bypasses external Host-Memory interconnect for operand streaming
- Constrained thermal envelope / lower operating frequency (e.g., 1.0 GHz)
- Only reduced / filtered final results are transmitted back to Host
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from memcentric.hardware.interconnect import Interconnect
from memcentric.hardware.memory import MemorySubsystem


@dataclass
class PIMConfig:
    name: str = "PIM-Cluster"
    num_pes: int = 8                # 8 lightweight logic-die PEs near DRAM banks
    frequency_ghz: float = 1.0       # Thermal-constrained frequency (1.0 GHz)
    simd_width: int = 2              # 2-way SIMD ALU per PE
    ipc: float = 1.0
    active_power_per_pe_watts: float = 0.8
    idle_power_watts: float = 0.5


class PIMSubsystem:
    """
    Cluster of Near-Memory Processing Elements (PEs) integrated on logic die.
    """
    def __init__(
        self,
        config: Optional[PIMConfig] = None,
        memory: Optional[MemorySubsystem] = None,
        interconnect: Optional[Interconnect] = None
    ):
        self.config = config or PIMConfig()
        self.memory = memory or MemorySubsystem()
        self.interconnect = interconnect or Interconnect()

        # Stats
        self.total_compute_cycles: int = 0
        self.total_dram_stall_cycles: int = 0
        self.total_energy_pj: float = 0.0
        self.total_offloaded_tasks: int = 0
        self.total_result_bytes_transmitted: int = 0

    def execute_kernel_in_memory(
        self,
        num_ops: int,
        memory_accesses: List[Tuple[int, bool]],  # (address, is_write)
        result_size_bytes: int,
        current_cycle: int
    ) -> Tuple[int, int]:
        """
        Executes a kernel entirely inside memory:
        1. Accesses DRAM directly through local bank logic without Host-Bus roundtrips.
        2. Executes compute on PIM PEs.
        3. Sends ONLY the aggregated/reduced result back across Interconnect to Host.
        Returns: (completion_cycle, total_latency_cycles)
        """
        self.total_offloaded_tasks += 1
        start_cycle = current_cycle

        # 1. Local DRAM accesses (parallelized across PEs/banks)
        max_bank_cycle = current_cycle
        for addr, is_write in memory_accesses:
            finish_c, _, _ = self.memory.access(addr=addr, is_write=is_write, current_cycle=current_cycle)
            if finish_c > max_bank_cycle:
                max_bank_cycle = finish_c

        dram_latency = max_bank_cycle - current_cycle
        self.total_dram_stall_cycles += dram_latency

        # 2. Local PIM compute cycles (scaled by PE parallelism and lower frequency)
        # Ratio of Host Freq (e.g. 3.2GHz) to PIM Freq (e.g. 1.0GHz)
        freq_scaling = 3.2 / self.config.frequency_ghz
        effective_throughput = self.config.num_pes * self.config.simd_width * self.config.ipc
        pim_compute_cycles = max(1, int((num_ops / effective_throughput) * freq_scaling))
        
        self.total_compute_cycles += pim_compute_cycles

        # Dynamic compute energy
        time_seconds = pim_compute_cycles / (3.2 * 1e9) # normalized to simulation time
        total_power = self.config.num_pes * self.config.active_power_per_pe_watts
        energy_pj = total_power * time_seconds * 1e12
        self.total_energy_pj += energy_pj

        compute_finish_cycle = max_bank_cycle + pim_compute_cycles

        # 3. Transmit final result back to Host via Interconnect
        final_cycle = compute_finish_cycle
        if result_size_bytes > 0:
            finish_tx, _ = self.interconnect.transmit(
                size_bytes=result_size_bytes, current_cycle=compute_finish_cycle
            )
            final_cycle = finish_tx
            self.total_result_bytes_transmitted += result_size_bytes

        total_latency = final_cycle - start_cycle
        return final_cycle, total_latency

    def reset(self):
        self.total_compute_cycles = 0
        self.total_dram_stall_cycles = 0
        self.total_energy_pj = 0.0
        self.total_offloaded_tasks = 0
        self.total_result_bytes_transmitted = 0

    def get_stats(self) -> Dict[str, float]:
        return {
            "compute_cycles": self.total_compute_cycles,
            "dram_stall_cycles": self.total_dram_stall_cycles,
            "total_cycles": self.total_compute_cycles + self.total_dram_stall_cycles,
            "total_energy_uj": round(self.total_energy_pj / 1e6, 3),
            "result_bytes_sent_kb": round(self.total_result_bytes_transmitted / 1024, 2),
            "tasks_executed": self.total_offloaded_tasks
        }
