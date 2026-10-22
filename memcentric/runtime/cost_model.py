"""
MemCentric-Sim: Analytical Cost & Performance Model
====================================================
Predicts execution latency and energy for Host execution vs. PIM offload.
Used by the dynamic runtime decision engine to make placement choices.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
from memcentric.runtime.profiler import KernelProfile


@dataclass
class CostEstimate:
    host_cycles: int
    host_energy_uj: float
    host_edp: float
    pim_cycles: int
    pim_energy_uj: float
    pim_edp: float
    recommended_placement: str # "HOST" or "PIM"
    confidence_margin: float


class AnalyticalCostModel:
    """
    Mathematical cost model formulating:
      T_host = T_compute_host + T_transfer_in + T_dram_read + T_transfer_out
      T_pim  = T_compute_pim  + T_pim_local_dram + T_transfer_reduction
    """
    def __init__(
        self,
        host_freq_ghz: float = 3.2,
        host_ipc_vector: float = 64.0,  # 4 cores * 2.0 IPC * 8 lanes
        host_power_watts: float = 65.0,
        bus_bw_gbps: float = 64.0,
        bus_latency_cycles: int = 40,
        bus_energy_per_byte_pj: float = 120.0,
        pim_freq_ghz: float = 1.0,
        pim_pes: int = 8,
        pim_simd: int = 2,
        pim_power_watts: float = 6.4,   # 8 PEs * 0.8W
        dram_latency_cycles: int = 45,
        dram_energy_per_line_pj: float = 1500.0, # 1.5 nJ
        l3_capacity_bytes: int = 16 * 1024 * 1024
    ):
        self.host_freq_ghz = host_freq_ghz
        self.host_ipc_vector = host_ipc_vector
        self.host_power_watts = host_power_watts
        self.bus_bw_gbps = bus_bw_gbps
        self.bus_latency_cycles = bus_latency_cycles
        self.bus_energy_per_byte_pj = bus_energy_per_byte_pj
        self.pim_freq_ghz = pim_freq_ghz
        self.pim_pes = pim_pes
        self.pim_simd = pim_simd
        self.pim_power_watts = pim_power_watts
        self.dram_latency_cycles = dram_latency_cycles
        self.dram_energy_per_line_pj = dram_energy_per_line_pj
        self.l3_capacity_bytes = l3_capacity_bytes

        # Normalized bus bytes per host cycle
        self.bus_bytes_per_cycle = self.bus_bw_gbps / self.host_freq_ghz

    def estimate(self, profile: KernelProfile, current_bus_queue_cycles: int = 0) -> CostEstimate:
        """
        Estimates runtime latency and energy for Host vs PIM.
        """
        # ==================== 1. HOST ESTIMATION ====================
        # Compute cycles
        host_compute_cycles = max(1, int(profile.num_ops / self.host_ipc_vector))

        # Cache behavior estimation
        # If input working set fits in L3 and reuse is high, DRAM traffic is small
        if profile.input_bytes <= self.l3_capacity_bytes and profile.temporal_reuse_factor > 2.0:
            effective_lines_fetched = profile.unique_lines
        else:
            # Thrashing or streaming
            effective_lines_fetched = int(profile.unique_lines * min(2.0, profile.temporal_reuse_factor))

        data_to_fetch_bytes = effective_lines_fetched * 64

        # Bus transfer time for host
        tx_in_cycles = (
            self.bus_latency_cycles + 
            current_bus_queue_cycles + 
            int(data_to_fetch_bytes / self.bus_bytes_per_cycle)
        )
        tx_out_cycles = (
            self.bus_latency_cycles + 
            int(profile.output_bytes / self.bus_bytes_per_cycle)
        )

        # Memory stall time (interleaved across DRAM channels/banks)
        parallel_banks = 32
        dram_cycles = int(effective_lines_fetched * self.dram_latency_cycles / parallel_banks)

        # Overlapping compute and memory (imperfect overlap factor ~0.7)
        total_host_cycles = max(host_compute_cycles, tx_in_cycles + dram_cycles) + int(0.3 * min(host_compute_cycles, tx_in_cycles)) + tx_out_cycles

        # Host Energy
        time_host_sec = total_host_cycles / (self.host_freq_ghz * 1e9)
        host_compute_energy_pj = self.host_power_watts * time_host_sec * 1e12
        host_bus_energy_pj = (data_to_fetch_bytes + profile.output_bytes) * self.bus_energy_per_byte_pj
        host_dram_energy_pj = effective_lines_fetched * self.dram_energy_per_line_pj
        total_host_energy_uj = (host_compute_energy_pj + host_bus_energy_pj + host_dram_energy_pj) / 1e6
        host_edp = (total_host_energy_uj / 1e6) * time_host_sec

        # ==================== 2. PIM ESTIMATION ====================
        # PIM compute throughput
        freq_ratio = self.host_freq_ghz / self.pim_freq_ghz
        pim_throughput = self.pim_pes * self.pim_simd
        pim_compute_cycles = max(1, int((profile.num_ops / pim_throughput) * freq_ratio))

        # Local DRAM accesses (direct wide TSV/bus: ~4x faster than host interconnect)
        pim_dram_cycles = int(profile.unique_lines * 14 / parallel_banks)

        # Interconnect transfer for PIM: ONLY the final output/reduction
        pim_tx_out_cycles = (
            self.bus_latency_cycles + 
            int(profile.output_bytes / self.bus_bytes_per_cycle)
            if profile.output_bytes > 0 else 0
        )

        total_pim_cycles = pim_compute_cycles + pim_dram_cycles + pim_tx_out_cycles

        # PIM Energy
        time_pim_sec = total_pim_cycles / (self.host_freq_ghz * 1e9)
        pim_compute_energy_pj = self.pim_power_watts * time_pim_sec * 1e12
        pim_bus_energy_pj = profile.output_bytes * self.bus_energy_per_byte_pj
        pim_dram_energy_pj = profile.unique_lines * (self.dram_energy_per_line_pj * 0.7) # local access saves I/O buffer energy
        total_pim_energy_uj = (pim_compute_energy_pj + pim_bus_energy_pj + pim_dram_energy_pj) / 1e6
        pim_edp = (total_pim_energy_uj / 1e6) * time_pim_sec

        # Recommended placement based on latency and operational intensity:
        # High Operational Intensity (>2.0) with high reuse -> Host CPU vector/cache advantage
        # Low Operational Intensity (<2.0) or latency-bound random access -> Near-Memory PIM
        if total_pim_cycles < total_host_cycles:
            recommended = "PIM"
            margin = (total_host_cycles - total_pim_cycles) / max(1, total_host_cycles)
        elif profile.operational_intensity >= 2.0 or total_host_cycles <= total_pim_cycles:
            recommended = "HOST"
            margin = (total_pim_cycles - total_host_cycles) / max(1, total_pim_cycles)
        elif pim_edp < (0.6 * host_edp):
            recommended = "PIM"
            margin = (host_edp - pim_edp) / max(1e-12, host_edp)
        else:
            recommended = "HOST"
            margin = (total_pim_cycles - total_host_cycles) / max(1, total_pim_cycles)

        return CostEstimate(
            host_cycles=total_host_cycles,
            host_energy_uj=total_host_energy_uj,
            host_edp=host_edp,
            pim_cycles=total_pim_cycles,
            pim_energy_uj=total_pim_energy_uj,
            pim_edp=pim_edp,
            recommended_placement=recommended,
            confidence_margin=margin
        )
