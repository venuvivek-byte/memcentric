"""
MemCentric-Sim: Analytical Cost & Performance Model
====================================================
Predicts execution latency and energy for Host execution vs. PIM offload.
Used by the dynamic runtime decision engine to make placement choices.

The model is parameterised from the same hardware configuration the simulator
uses (see `from_hardware`), so its predictions track the simulated machine:
  - every host cache miss holds the link for a request and a 64 B response,
    and pays a DRAM row access in between;
  - data already resident in the host caches costs no interconnect traffic;
  - offloading data that is dirty in the host caches first requires a
    coherence write-back of each dirty line;
  - PIM DRAM time is set by the busiest bank (the data's physical location);
  - traffic already queued on the link delays whichever path uses it first.
"""

from dataclasses import dataclass
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
    host_breakdown: dict = None
    pim_breakdown: dict = None


class AnalyticalCostModel:
    """
    T_host = T_compute_host + T_cache + N_miss * (t_req + t_dram + t_resp) + T_backlog
    T_pim  = T_flush + T_bank(max) + T_compute_pim + T_result + T_residual_backlog
    """
    def __init__(
        self,
        host_freq_ghz: float = 3.2,
        host_cores: int = 4,
        host_ipc: float = 2.0,
        host_vector_lanes: int = 8,
        host_power_watts: float = 65.0,
        bus_bw_gbps: float = 64.0,
        bus_latency_cycles: int = 40,
        bus_energy_per_byte_pj: float = 120.0,
        pim_freq_ghz: float = 1.0,
        pim_pes: int = 8,
        pim_simd: int = 2,
        pim_ipc: float = 1.0,
        pim_power_watts: float = 6.4,   # 8 PEs * 0.8W
        dram_access_cycles: int = 46,   # row conflict: tRP + tRCD + tCAS + tBURST
        dram_row_hit_cycles: int = 18,  # open-row hit: tCAS + tBURST
        dram_energy_per_line_pj: float = 1500.0, # 1.5 nJ
        cache_path_cycles: int = 58,    # L1 + L2 + L3 lookup before a miss
        l3_capacity_bytes: int = 16 * 1024 * 1024
    ):
        self.host_freq_ghz = host_freq_ghz
        self.host_cores = host_cores
        self.host_ipc = host_ipc
        self.host_vector_lanes = host_vector_lanes
        self.host_power_watts = host_power_watts
        self.bus_bw_gbps = bus_bw_gbps
        self.bus_latency_cycles = bus_latency_cycles
        self.bus_energy_per_byte_pj = bus_energy_per_byte_pj
        self.pim_freq_ghz = pim_freq_ghz
        self.pim_pes = pim_pes
        self.pim_simd = pim_simd
        self.pim_ipc = pim_ipc
        self.pim_power_watts = pim_power_watts
        self.dram_access_cycles = dram_access_cycles
        self.dram_row_hit_cycles = dram_row_hit_cycles
        self.dram_energy_per_line_pj = dram_energy_per_line_pj
        self.cache_path_cycles = cache_path_cycles
        self.l3_capacity_bytes = l3_capacity_bytes

        # Normalized bus bytes per host cycle
        self.bus_bytes_per_cycle = self.bus_bw_gbps / self.host_freq_ghz

    @classmethod
    def from_hardware(cls, host_cfg, pim_cfg, interconnect, memory, cache) -> "AnalyticalCostModel":
        """Builds a model whose parameters match the simulated components."""
        return cls(
            host_freq_ghz=host_cfg.frequency_ghz,
            host_cores=host_cfg.num_cores,
            host_ipc=host_cfg.ipc,
            host_vector_lanes=host_cfg.vector_width_lanes,
            host_power_watts=host_cfg.active_power_watts,
            bus_bw_gbps=interconnect.bandwidth_gbps,
            bus_latency_cycles=interconnect.base_latency_cycles,
            bus_energy_per_byte_pj=interconnect.energy_per_byte_pj,
            pim_freq_ghz=pim_cfg.frequency_ghz,
            pim_pes=pim_cfg.num_pes,
            pim_simd=pim_cfg.simd_width,
            pim_ipc=pim_cfg.ipc,
            pim_power_watts=pim_cfg.num_pes * pim_cfg.active_power_per_pe_watts,
            dram_access_cycles=memory.row_conflict_latency,
            dram_row_hit_cycles=memory.timing.tCAS + memory.timing.tBURST,
            cache_path_cycles=cache.hit_path_latency,
            l3_capacity_bytes=cache.l3.size_bytes
        )

    def _dram_cycles(self, row_hit_fraction: float) -> float:
        """Average DRAM access latency given the profiled open-row hit rate."""
        h = row_hit_fraction
        return h * self.dram_row_hit_cycles + (1.0 - h) * self.dram_access_cycles

    def _link_cycles(self, size_bytes: int) -> int:
        return self.bus_latency_cycles + max(1, int(size_bytes / self.bus_bytes_per_cycle))

    def estimate(self, profile: KernelProfile, current_bus_queue_cycles: int = 0) -> CostEstimate:
        """
        Estimates runtime latency and energy for Host vs PIM.
        """
        backlog = max(0, current_bus_queue_cycles)

        # ==================== 1. HOST ESTIMATION ====================
        lanes = self.host_vector_lanes if profile.vectorizable else 1
        host_compute_cycles = max(1, int(profile.num_ops / (self.host_cores * self.host_ipc * lanes)))

        cold = 1.0 - profile.resident_fraction
        if profile.input_bytes <= self.l3_capacity_bytes:
            misses = profile.unique_lines * cold
        else:
            # Working set overflows L3: reuse cannot be captured
            misses = profile.total_accesses * cold

        out_lines = profile.output_bytes // 64
        if profile.output_bytes > 0:
            out_lines = max(1, out_lines)
        misses += out_lines * (1.0 - profile.output_resident_fraction)

        dram_cycles = self._dram_cycles(profile.host_row_hit_fraction)
        per_miss = self._link_cycles(8) + dram_cycles + self._link_cycles(64)
        host_mem_cycles = self.cache_path_cycles + int(misses * per_miss)
        # Misses issue after compute, so backlog drains while the host computes
        host_queue_cycles = (
            max(0, backlog - host_compute_cycles - self.cache_path_cycles) if misses >= 1 else 0
        )
        total_host_cycles = host_compute_cycles + host_mem_cycles + host_queue_cycles

        time_host_sec = total_host_cycles / (self.host_freq_ghz * 1e9)
        host_bus_bytes = misses * (8 + 64)
        host_energy_pj = (
            self.host_power_watts * (host_compute_cycles / (self.host_freq_ghz * 1e9)) * 1e12
            + host_bus_bytes * self.bus_energy_per_byte_pj
            + misses * self.dram_energy_per_line_pj
        )
        total_host_energy_uj = host_energy_pj / 1e6
        host_edp = (total_host_energy_uj / 1e6) * time_host_sec

        # ==================== 2. PIM ESTIMATION ====================
        dirty_lines = profile.unique_lines * profile.dirty_fraction
        flush_cycles = int(dirty_lines * self._link_cycles(64))
        if dirty_lines >= 1:
            flush_cycles += backlog  # write-backs queue behind traffic already on the link

        pim_dram_cycles = int(profile.max_bank_accesses * self._dram_cycles(profile.pim_row_hit_fraction))

        freq_ratio = self.host_freq_ghz / self.pim_freq_ghz
        pim_throughput = self.pim_pes * self.pim_simd * self.pim_ipc
        pim_compute_cycles = max(1, int((profile.num_ops / pim_throughput) * freq_ratio))

        pim_tx_out_cycles = self._link_cycles(profile.output_bytes) if profile.output_bytes > 0 else 0
        busy = flush_cycles + pim_dram_cycles + pim_compute_cycles
        residual = max(0, backlog - busy) if (profile.output_bytes > 0 and dirty_lines < 1) else 0

        total_pim_cycles = busy + pim_tx_out_cycles + residual

        time_pim_sec = total_pim_cycles / (self.host_freq_ghz * 1e9)
        pim_energy_pj = (
            self.pim_power_watts * (pim_compute_cycles / (self.host_freq_ghz * 1e9)) * 1e12
            + (profile.output_bytes + dirty_lines * 64) * self.bus_energy_per_byte_pj
            + profile.unique_lines * self.dram_energy_per_line_pj * 0.7  # local access skips I/O buffers
        )
        total_pim_energy_uj = pim_energy_pj / 1e6
        pim_edp = (total_pim_energy_uj / 1e6) * time_pim_sec

        # Latency objective: run where the task finishes first; ties stay on host.
        if total_pim_cycles < total_host_cycles:
            recommended = "PIM"
            margin = (total_host_cycles - total_pim_cycles) / max(1, total_host_cycles)
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
            confidence_margin=margin,
            host_breakdown={
                "compute": host_compute_cycles,
                "memory": host_mem_cycles,
                "bus_queue": host_queue_cycles,
            },
            pim_breakdown={
                "coherence_flush": flush_cycles,
                "bank": pim_dram_cycles,
                "compute": pim_compute_cycles,
                "result": pim_tx_out_cycles + residual,
            }
        )
