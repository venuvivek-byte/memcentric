"""
MemCentric-Sim: Comprehensive Energy & EDP Accounting System
============================================================
Tracks Joules, Watts, and Energy-Delay Product (EDP) across all subsystems:
- Host CPU Active / Static Energy
- Multi-Level Cache Read/Write Energy
- CXL/PCIe Link Serialization & SerDes PHY Energy
- DRAM Activation, Precharge, Burst Energy
- PIM PE Compute Energy
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class EnergyReport:
    total_cycles: int
    simulation_time_sec: float
    host_compute_energy_uj: float
    cache_energy_uj: float
    interconnect_energy_uj: float
    dram_energy_uj: float
    pim_energy_uj: float
    total_energy_uj: float
    edp_joule_seconds: float
    data_moved_mb: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cycles": self.total_cycles,
            "simulation_time_sec": self.simulation_time_sec,
            "host_compute_energy_uj": round(self.host_compute_energy_uj, 4),
            "cache_energy_uj": round(self.cache_energy_uj, 4),
            "interconnect_energy_uj": round(self.interconnect_energy_uj, 4),
            "dram_energy_uj": round(self.dram_energy_uj, 4),
            "pim_energy_uj": round(self.pim_energy_uj, 4),
            "total_energy_uj": round(self.total_energy_uj, 4),
            "total_energy_mj": round(self.total_energy_uj / 1000.0, 4),
            "edp_joule_seconds": round(self.edp_joule_seconds, 8),
            "data_moved_mb": round(self.data_moved_mb, 4),
        }


class EnergyModel:
    @staticmethod
    def calculate_edp(total_energy_joules: float, total_time_seconds: float) -> float:
        """EDP = Joules * Seconds (lower is better)."""
        return total_energy_joules * total_time_seconds

    @staticmethod
    def generate_report(
        cycles: int,
        freq_ghz: float,
        host_energy_pj: float,
        interconnect_energy_pj: float,
        dram_energy_pj: float,
        pim_energy_pj: float,
        bytes_transferred: int
    ) -> EnergyReport:
        time_sec = cycles / (freq_ghz * 1e9)
        host_uj = host_energy_pj / 1e6
        interconnect_uj = interconnect_energy_pj / 1e6
        dram_uj = dram_energy_pj / 1e6
        pim_uj = pim_energy_pj / 1e6
        total_uj = host_uj + interconnect_uj + dram_uj + pim_uj
        total_joules = total_uj / 1e6
        edp = total_joules * time_sec
        data_mb = bytes_transferred / (1024.0 * 1024.0)

        return EnergyReport(
            total_cycles=cycles,
            simulation_time_sec=time_sec,
            host_compute_energy_uj=host_uj,
            cache_energy_uj=0.0, # Included in host_energy_pj
            interconnect_energy_uj=interconnect_uj,
            dram_energy_uj=dram_uj,
            pim_energy_uj=pim_uj,
            total_energy_uj=total_uj,
            edp_joule_seconds=edp,
            data_moved_mb=data_mb
        )
