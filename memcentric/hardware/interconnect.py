"""
MemCentric-Sim: Interconnect Model (CXL / PCIe Bus)
===================================================
Models the communication link between Host CPU and Memory/PIM subsystem.
Includes bandwidth throttling, serialization latency, queue contention,
and per-byte transmission energy tracking.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Packet:
    packet_id: int
    src: str
    dst: str
    size_bytes: int
    scheduled_cycle: int
    payload: Dict = None


class Interconnect:
    """
    Simulates high-speed serial interconnect (e.g., CXL 3.0 / PCIe 5.0).
    Key dynamics:
      - Propagation latency (base fixed delay in cycles)
      - Transmission latency = size_bytes / bandwidth_bytes_per_cycle
      - Serialization and channel queuing (busy_until cycle)
      - Dynamic transfer energy in pJ/bit (or pJ/byte)
    """
    def __init__(
        self,
        name: str = "Host-to-Mem-Bus",
        bandwidth_gbps: float = 64.0,       # e.g., 64 GB/s bi-directional
        base_latency_cycles: int = 40,      # ~20ns at 2GHz
        energy_per_byte_pj: float = 120.0,  # ~15 pJ/bit SerDes PHY link energy
        host_freq_ghz: float = 2.0
    ):
        self.name = name
        self.bandwidth_gbps = bandwidth_gbps
        self.base_latency_cycles = base_latency_cycles
        self.energy_per_byte_pj = energy_per_byte_pj
        self.host_freq_ghz = host_freq_ghz

        # Bytes per host cycle = (GB/s * 10^9) / (Freq_GHz * 10^9) = GB/s / Freq_GHz
        self.bytes_per_cycle = self.bandwidth_gbps / self.host_freq_ghz

        # Link availability tracking
        self.busy_until_cycle: int = 0
        self.active_transfers: int = 0

        # Statistics
        self.total_bytes_transferred: int = 0
        self.total_packets_transferred: int = 0
        self.total_transfer_energy_pj: float = 0.0
        self.total_queuing_delay_cycles: int = 0

        # Traffic from other agents (DMA engines, NICs) sharing the link.
        # It occupies the link but is not attributed to the simulated workload.
        self.background_bytes_transferred: int = 0
        self.background_packets_transferred: int = 0

    @property
    def request_occupancy_cycles(self) -> int:
        """Link cycles held by one 8-byte miss request."""
        return self.base_latency_cycles + max(1, int(8 / self.bytes_per_cycle))

    @property
    def line_occupancy_cycles(self) -> int:
        """Link cycles held by one 64-byte cache-line transfer."""
        return self.base_latency_cycles + max(1, int(64 / self.bytes_per_cycle))

    def queue_backlog(self, current_cycle: int) -> int:
        """Cycles a request issued now would wait before the link is free."""
        return max(0, self.busy_until_cycle - current_cycle)

    def transmit(self, size_bytes: int, current_cycle: int, account: bool = True) -> Tuple[int, int]:
        """
        Transmits data over the bus.
        Link reservations are granted in issue order (FIFO).
        account=False marks background traffic: it occupies the link but is
        kept out of the workload's byte, packet and energy statistics.
        Returns:
            (completion_cycle, total_latency_cycles)
        """
        # Transmission delay based on bandwidth
        tx_cycles = max(1, int(size_bytes / self.bytes_per_cycle))
        
        # Queuing delay if link is currently occupied
        wait_cycles = max(0, self.busy_until_cycle - current_cycle)
        start_cycle = current_cycle + wait_cycles
        finish_cycle = start_cycle + self.base_latency_cycles + tx_cycles

        self.busy_until_cycle = finish_cycle
        total_latency = finish_cycle - current_cycle

        if not account:
            self.background_bytes_transferred += size_bytes
            self.background_packets_transferred += 1
            return finish_cycle, total_latency

        # Update stats
        self.total_bytes_transferred += size_bytes
        self.total_packets_transferred += 1
        self.total_queuing_delay_cycles += wait_cycles
        energy_pj = size_bytes * self.energy_per_byte_pj
        self.total_transfer_energy_pj += energy_pj

        return finish_cycle, total_latency

    def get_utilization(self, total_simulation_cycles: int) -> float:
        """Returns link busy ratio (0.0 to 1.0)."""
        if total_simulation_cycles <= 0:
            return 0.0
        total_busy_estimate = (self.total_bytes_transferred / self.bytes_per_cycle)
        return min(1.0, total_busy_estimate / total_simulation_cycles)

    def reset(self):
        self.busy_until_cycle = 0
        self.active_transfers = 0
        self.total_bytes_transferred = 0
        self.total_packets_transferred = 0
        self.total_transfer_energy_pj = 0.0
        self.total_queuing_delay_cycles = 0
        self.background_bytes_transferred = 0
        self.background_packets_transferred = 0

    def get_stats(self) -> Dict[str, float]:
        return {
            "total_bytes_mb": round(self.total_bytes_transferred / (1024 * 1024), 3),
            "total_packets": self.total_packets_transferred,
            "total_energy_uj": round(self.total_transfer_energy_pj / 1e6, 3),
            "avg_queuing_delay_cycles": (
                round(self.total_queuing_delay_cycles / self.total_packets_transferred, 2)
                if self.total_packets_transferred > 0 else 0.0
            ),
            "background_bytes_mb": round(self.background_bytes_transferred / (1024 * 1024), 3),
        }
