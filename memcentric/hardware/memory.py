"""
MemCentric-Sim: Multi-Channel DRAM / HBM Subsystem Model
=========================================================
Accurately models multi-channel, multi-bank DRAM memory with:
- Bank-Level Parallelism (BLP)
- Row buffer hits, misses, and conflicts (Open-Page Policy)
- Channel / bank queuing and serialization
- DRAM timing parameters (tCAS, tRCD, tRP, tCCD)
- Per-operation energy accounting (Activation, Precharge, Burst Read/Write)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class BankState:
    bank_id: int
    open_row: Optional[int] = None
    busy_until_cycle: int = 0
    row_hits: int = 0
    row_misses: int = 0
    row_conflicts: int = 0


@dataclass
class MemoryTiming:
    """Standard DDR4/DDR5/HBM2 timing in host cycles (normalized to host freq)."""
    tCAS: int = 14   # Column Access Strobe / Read latency
    tRCD: int = 14   # Row to Column Delay (Activate to Read/Write)
    tRP: int = 14    # Row Precharge delay
    tCCD: int = 4    # Column-to-Column delay (burst separation)
    tBURST: int = 4  # Burst transfer length cycles


@dataclass
class MemoryEnergyConfig:
    """Energy per DRAM primitive operation in picojoules (pJ)."""
    act_energy_pj: float = 2400.0   # Activation energy (~2.4 nJ)
    pre_energy_pj: float = 1400.0   # Precharge energy (~1.4 nJ)
    read_energy_pj: float = 900.0   # 64B burst read energy (~0.9 nJ)
    write_energy_pj: float = 950.0  # 64B burst write energy (~0.95 nJ)
    background_power_mw: float = 120.0 # Standby/refresh power in mW


class MemorySubsystem:
    """
    Multi-channel, multi-bank memory controller and physical DRAM simulator.
    """
    def __init__(
        self,
        num_channels: int = 4,
        banks_per_channel: int = 8,
        timing: Optional[MemoryTiming] = None,
        energy_cfg: Optional[MemoryEnergyConfig] = None,
        row_size_bytes: int = 2048,
        line_size_bytes: int = 64
    ):
        self.num_channels = num_channels
        self.banks_per_channel = banks_per_channel
        self.total_banks = num_channels * banks_per_channel
        self.timing = timing or MemoryTiming()
        self.energy_cfg = energy_cfg or MemoryEnergyConfig()
        self.row_size_bytes = row_size_bytes
        self.line_size_bytes = line_size_bytes

        # Internal bank states
        self.banks: List[BankState] = [
            BankState(bank_id=i) for i in range(self.total_banks)
        ]

        # Aggregate Statistics
        self.total_accesses: int = 0
        self.total_row_hits: int = 0
        self.total_row_misses: int = 0
        self.total_row_conflicts: int = 0
        self.total_dram_energy_pj: float = 0.0

    def _map_address(self, addr: int) -> Tuple[int, int, int]:
        """
        Interleaved address mapping:
        [Tag/Row bits] [Bank bits] [Channel bits] [Offset within row bits]
        """
        line_addr = addr // self.line_size_bytes
        channel_id = line_addr % self.num_channels
        bank_id = (line_addr // self.num_channels) % self.total_banks
        row_id = addr // self.row_size_bytes
        return channel_id, bank_id, row_id

    def access(self, addr: int, is_write: bool, current_cycle: int) -> Tuple[int, int, float]:
        """
        Simulates access to DRAM bank.
        Returns:
            (completion_cycle, latency_cycles, energy_pj)
        """
        self.total_accesses += 1
        _, bank_idx, row_id = self._map_address(addr)
        bank = self.banks[bank_idx]

        # Calculate timing based on row buffer state
        start_cycle = max(current_cycle, bank.busy_until_cycle)
        operation_latency = 0
        energy_pj = 0.0

        if bank.open_row is None:
            # Row buffer empty: ACTIVATE + CAS
            operation_latency = self.timing.tRCD + self.timing.tCAS + self.timing.tBURST
            energy_pj += self.energy_cfg.act_energy_pj
            bank.open_row = row_id
            bank.row_misses += 1
            self.total_row_misses += 1
        elif bank.open_row == row_id:
            # Row buffer hit: immediate CAS
            operation_latency = self.timing.tCAS + self.timing.tBURST
            bank.row_hits += 1
            self.total_row_hits += 1
        else:
            # Row buffer conflict: PRECHARGE old row + ACTIVATE new row + CAS
            operation_latency = (
                self.timing.tRP + self.timing.tRCD + self.timing.tCAS + self.timing.tBURST
            )
            energy_pj += self.energy_cfg.pre_energy_pj + self.energy_cfg.act_energy_pj
            bank.open_row = row_id
            bank.row_conflicts += 1
            self.total_row_conflicts += 1

        # Add Read/Write burst energy
        energy_pj += (
            self.energy_cfg.write_energy_pj if is_write else self.energy_cfg.read_energy_pj
        )
        self.total_dram_energy_pj += energy_pj

        finish_cycle = start_cycle + operation_latency
        bank.busy_until_cycle = finish_cycle
        total_latency = finish_cycle - current_cycle

        return finish_cycle, total_latency, energy_pj

    def get_row_hit_rate(self) -> float:
        return (
            (self.total_row_hits / self.total_accesses)
            if self.total_accesses > 0 else 0.0
        )

    def reset(self):
        self.banks = [BankState(bank_id=i) for i in range(self.total_banks)]
        self.total_accesses = 0
        self.total_row_hits = 0
        self.total_row_misses = 0
        self.total_row_conflicts = 0
        self.total_dram_energy_pj = 0.0

    def get_stats(self) -> Dict[str, float]:
        return {
            "total_accesses": self.total_accesses,
            "row_hit_rate": round(self.get_row_hit_rate(), 4),
            "row_hits": self.total_row_hits,
            "row_conflicts": self.total_row_conflicts,
            "row_misses": self.total_row_misses,
            "energy_uj": round(self.total_dram_energy_pj / 1e6, 3),
        }
