"""
MemCentric-Sim: Complete Memory-Centric Computing System Simulator
==================================================================
Ties together:
- Discrete-Event Simulation Engine
- Host Processor with 3-Level Cache Hierarchy
- CXL 3.0 / PCIe Interconnect
- Multi-Channel Bank-Interleaved DRAM Subsystem
- Near-Memory Computing (PIM) Cluster
- Intelligent Runtime Placement Engine
- Energy & Telemetry Accounting
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from memcentric.core.engine import SimulationEngine
from memcentric.hardware.cache import CacheHierarchy
from memcentric.hardware.interconnect import Interconnect
from memcentric.hardware.memory import MemorySubsystem, MemoryTiming, MemoryEnergyConfig
from memcentric.hardware.host import HostProcessor, HostConfig
from memcentric.hardware.pim import PIMSubsystem, PIMConfig
from memcentric.hardware.energy_model import EnergyModel, EnergyReport
from memcentric.runtime.profiler import MicroProfiler
from memcentric.runtime.cost_model import AnalyticalCostModel
from memcentric.runtime.policy import PlacementEngine, PlacementPolicy, PlacementDecision
from memcentric.workloads.base import KernelTask


@dataclass
class TaskExecutionRecord:
    task_name: str
    category: str
    target: str
    policy: str
    start_cycle: int
    end_cycle: int
    elapsed_cycles: int
    bytes_transferred: int
    decision_overhead_cycles: int
    decision_reason: str
    energy_report: EnergyReport


@dataclass
class SimulationResult:
    total_cycles: int
    simulation_time_sec: float
    total_energy_uj: float
    total_energy_mj: float
    total_bytes_transferred: int
    edp_joule_sec: float
    bus_utilization: float
    decisions_summary: Dict[str, Any]
    task_records: List[TaskExecutionRecord]
    energy_breakdown: Dict[str, float]
    hardware_stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cycles": self.total_cycles,
            "simulation_time_sec": self.simulation_time_sec,
            "total_energy_uj": round(self.total_energy_uj, 4),
            "total_energy_mj": round(self.total_energy_mj, 4),
            "total_data_mb": round(self.total_bytes_transferred / (1024 * 1024), 4),
            "edp_joule_sec": round(self.edp_joule_sec, 8),
            "bus_utilization_pct": round(self.bus_utilization * 100.0, 2),
            "decisions_summary": self.decisions_summary,
            "energy_breakdown": self.energy_breakdown,
            "tasks": [
                {
                    "task_name": t.task_name,
                    "category": t.category,
                    "target": t.target,
                    "policy": t.policy,
                    "cycles": t.elapsed_cycles,
                    "bytes_moved_kb": round(t.bytes_transferred / 1024.0, 2),
                    "overhead_cycles": t.decision_overhead_cycles,
                    "reason": t.decision_reason
                }
                for t in self.task_records
            ]
        }


class MemoryCentricSystem:
    """
    Top-level simulated architecture.
    """
    def __init__(
        self,
        host_config: Optional[HostConfig] = None,
        pim_config: Optional[PIMConfig] = None,
        bus_bw_gbps: float = 64.0,
        bus_latency_cycles: int = 40,
        dram_channels: int = 4,
        banks_per_channel: int = 8,
        policy: PlacementPolicy = PlacementPolicy.DYNAMIC_ANALYTICAL
    ):
        self.engine = SimulationEngine(frequency_ghz=3.2)
        
        # Interconnect
        self.interconnect = Interconnect(
            bandwidth_gbps=bus_bw_gbps,
            base_latency_cycles=bus_latency_cycles,
            host_freq_ghz=3.2
        )

        # DRAM
        self.memory = MemorySubsystem(
            num_channels=dram_channels,
            banks_per_channel=banks_per_channel
        )

        # Cache & Host
        self.cache = CacheHierarchy()
        self.host = HostProcessor(
            config=host_config,
            cache_hierarchy=self.cache,
            interconnect=self.interconnect,
            memory=self.memory
        )

        # PIM Cluster
        self.pim = PIMSubsystem(
            config=pim_config,
            memory=self.memory,
            interconnect=self.interconnect
        )

        # Placement Engine
        self.placement_engine = PlacementEngine(
            policy=policy,
            cost_model=AnalyticalCostModel(
                host_freq_ghz=3.2,
                bus_bw_gbps=bus_bw_gbps,
                bus_latency_cycles=bus_latency_cycles,
                pim_freq_ghz=pim_config.frequency_ghz if pim_config else 1.0,
                pim_pes=pim_config.num_pes if pim_config else 16
            )
        )

        self.task_records: List[TaskExecutionRecord] = []

    def set_policy(self, policy: PlacementPolicy):
        self.placement_engine.policy = policy

    def run_task(self, task: KernelTask) -> TaskExecutionRecord:
        """
        Executes a single task through the architecture.
        """
        task_start_cycle = self.engine.current_cycle
        bytes_before = self.interconnect.total_bytes_transferred

        # 1. Runtime placement decision with explicit overhead
        current_congestion = max(0, self.interconnect.busy_until_cycle - self.engine.current_cycle)
        decision: PlacementDecision = self.placement_engine.decide(
            kernel_name=task.name,
            num_ops=task.num_ops,
            memory_accesses=task.memory_accesses,
            output_bytes=task.output_bytes,
            current_bus_congestion_cycles=current_congestion
        )

        # Advance engine clock by decision overhead
        self.engine.current_cycle += decision.decision_overhead_cycles

        # 2. Execute on chosen target
        target = decision.target
        task_execution_finish = self.engine.current_cycle

        if target == "PIM":
            # Execute on Near-Memory Computing cluster
            task_execution_finish, _ = self.pim.execute_kernel_in_memory(
                num_ops=task.num_ops,
                memory_accesses=task.memory_accesses,
                result_size_bytes=task.output_bytes,
                current_cycle=self.engine.current_cycle
            )
        else: # "HOST"
            # Execute on Host Processor
            # Host executes compute ops
            compute_lat = self.host.execute_compute(
                num_ops=task.num_ops,
                is_vector=(task.category in ["HIGH_REUSE", "STREAMING"])
            )
            self.engine.current_cycle += compute_lat

            # Host streams memory accesses
            max_cycle = self.engine.current_cycle
            for addr, is_write in task.memory_accesses:
                finish_c, _ = self.host.access_memory(
                    addr=addr,
                    is_write=is_write,
                    current_cycle=self.engine.current_cycle
                )
                if finish_c > max_cycle:
                    max_cycle = finish_c
            
            # Host writes back final output if needed
            if task.output_bytes > 0:
                out_lines = max(1, task.output_bytes // 64)
                for l in range(out_lines):
                    finish_c, _ = self.host.access_memory(
                        addr=0xF00000 + (l * 64),
                        is_write=True,
                        current_cycle=max_cycle
                    )
                    if finish_c > max_cycle:
                        max_cycle = finish_c

            task_execution_finish = max_cycle

        self.engine.current_cycle = task_execution_finish
        elapsed = task_execution_finish - task_start_cycle
        bytes_transferred = self.interconnect.total_bytes_transferred - bytes_before

        # Energy calculation for this task
        energy_report = EnergyModel.generate_report(
            cycles=elapsed,
            freq_ghz=self.engine.frequency_ghz,
            host_energy_pj=self.host.total_energy_pj,
            interconnect_energy_pj=self.interconnect.total_transfer_energy_pj,
            dram_energy_pj=self.memory.total_dram_energy_pj,
            pim_energy_pj=self.pim.total_energy_pj + decision.decision_overhead_energy_pj,
            bytes_transferred=bytes_transferred
        )

        record = TaskExecutionRecord(
            task_name=task.name,
            category=task.category,
            target=target,
            policy=decision.policy_used.value,
            start_cycle=task_start_cycle,
            end_cycle=task_execution_finish,
            elapsed_cycles=elapsed,
            bytes_transferred=bytes_transferred,
            decision_overhead_cycles=decision.decision_overhead_cycles,
            decision_reason=decision.reason,
            energy_report=energy_report
        )
        self.task_records.append(record)
        return record

    def run_workload(self, tasks: List[KernelTask]) -> SimulationResult:
        """
        Runs an entire workload sequence and produces aggregated results.
        """
        for task in tasks:
            self.run_task(task)

        total_cycles = self.engine.current_cycle
        total_time_sec = total_cycles / (self.engine.frequency_ghz * 1e9)

        # Aggregate Energy
        host_uj = self.host.total_energy_pj / 1e6
        bus_uj = self.interconnect.total_transfer_energy_pj / 1e6
        dram_uj = self.memory.total_dram_energy_pj / 1e6
        pim_uj = (self.pim.total_energy_pj + self.placement_engine.total_overhead_energy_pj) / 1e6
        total_energy_uj = host_uj + bus_uj + dram_uj + pim_uj
        total_joules = total_energy_uj / 1e6

        edp = total_joules * total_time_sec
        bus_util = self.interconnect.get_utilization(total_cycles)

        result = SimulationResult(
            total_cycles=total_cycles,
            simulation_time_sec=total_time_sec,
            total_energy_uj=total_energy_uj,
            total_energy_mj=total_energy_uj / 1000.0,
            total_bytes_transferred=self.interconnect.total_bytes_transferred,
            edp_joule_sec=edp,
            bus_utilization=bus_util,
            decisions_summary=self.placement_engine.get_summary(),
            task_records=self.task_records,
            energy_breakdown={
                "host_compute_uj": round(host_uj, 3),
                "interconnect_bus_uj": round(bus_uj, 3),
                "dram_uj": round(dram_uj, 3),
                "pim_uj": round(pim_uj, 3)
            },
            hardware_stats={
                "host": self.host.get_stats(),
                "pim": self.pim.get_stats(),
                "memory": self.memory.get_stats(),
                "interconnect": self.interconnect.get_stats()
            }
        )
        return result

    def reset(self):
        """Resets all subsystems to initial state."""
        self.engine.reset()
        self.interconnect.reset()
        self.memory.reset()
        self.host.reset()
        self.pim.reset()
        self.placement_engine.reset()
        self.task_records.clear()
