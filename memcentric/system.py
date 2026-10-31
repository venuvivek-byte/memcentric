"""
MemCentric-Sim: Complete Memory-Centric Computing System Simulator
==================================================================
Ties together:
- Discrete-Event Simulation Engine
- Host Processor with 3-Level Cache Hierarchy
- CXL 3.0 / PCIe Interconnect (shared with a background DMA agent)
- Multi-Channel Bank-Interleaved DRAM Subsystem
- Near-Memory Computing (PIM) Cluster
- Intelligent Runtime Placement Engine
- Energy & Telemetry Accounting

Every task moves through the event queue:
  TaskArrival   -> runtime profiles the task and observes bus/cache state
  TaskDispatch  -> fires after the decision overhead; task starts on its target
  TaskComplete  -> fires when the target finishes; the next task arrives
DMA bursts from other devices are events too, and they contend for the
same interconnect the host uses to reach memory.
"""

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional
from memcentric.core.engine import SimulationEngine
from memcentric.hardware.cache import CacheHierarchy
from memcentric.hardware.interconnect import Interconnect
from memcentric.hardware.memory import MemorySubsystem, MemoryTiming, MemoryEnergyConfig
from memcentric.hardware.host import HostProcessor, HostConfig
from memcentric.hardware.pim import PIMSubsystem, PIMConfig
from memcentric.hardware.dma import BackgroundDMA
from memcentric.hardware.energy_model import EnergyModel, EnergyReport
from memcentric.runtime.profiler import MicroProfiler
from memcentric.runtime.cost_model import AnalyticalCostModel
from memcentric.runtime.policy import PlacementEngine, PlacementPolicy, PlacementDecision
from memcentric.workloads.base import KernelTask

# Host-side buffer where results computed on the host are written
HOST_RESULT_BASE = 0xF00000


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
    bus_backlog_at_arrival: int = 0
    coherence_writebacks: int = 0
    predicted_host_cycles: Optional[int] = None
    predicted_pim_cycles: Optional[int] = None


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
    events_processed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_cycles": self.total_cycles,
            "simulation_time_sec": self.simulation_time_sec,
            "total_energy_uj": round(self.total_energy_uj, 4),
            "total_energy_mj": round(self.total_energy_mj, 4),
            "total_data_mb": round(self.total_bytes_transferred / (1024 * 1024), 4),
            "edp_joule_sec": round(self.edp_joule_sec, 8),
            "bus_utilization_pct": round(self.bus_utilization * 100.0, 2),
            "events_processed": self.events_processed,
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
        policy: PlacementPolicy = PlacementPolicy.DYNAMIC_ANALYTICAL,
        profiler: Optional[MicroProfiler] = None,
        decision_cycles: int = 35
    ):
        self.engine = SimulationEngine(frequency_ghz=3.2)

        # Interconnect
        self.interconnect = Interconnect(
            bandwidth_gbps=bus_bw_gbps,
            base_latency_cycles=bus_latency_cycles,
            host_freq_ghz=3.2
        )
        self.dma = BackgroundDMA(self.interconnect)

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

        # Placement Engine (cost model parameterised from the components above)
        self.placement_engine = PlacementEngine(
            policy=policy,
            profiler=profiler,
            decision_cycles=decision_cycles,
            cost_model=AnalyticalCostModel.from_hardware(
                self.host.config, self.pim.config, self.interconnect, self.memory, self.cache
            )
        )

        self.task_records: List[TaskExecutionRecord] = []
        self._task_queue: Deque[KernelTask] = deque()
        self._last_completion_cycle: int = 0

    def set_policy(self, policy: PlacementPolicy):
        self.placement_engine.policy = policy

    # ------------------------------------------------------------------
    # Execution on each target (pure timing + state update, no events)
    # ------------------------------------------------------------------
    def _execute(self, task: KernelTask, target: str, start_cycle: int) -> Dict[str, int]:
        if target == "PIM":
            return self._execute_pim(task, start_cycle)
        return self._execute_host(task, start_cycle)

    def _execute_host(self, task: KernelTask, start_cycle: int) -> Dict[str, int]:
        compute_lat = self.host.execute_compute(num_ops=task.num_ops, is_vector=task.vectorizable)
        issue_cycle = start_cycle + compute_lat

        # Host streams memory accesses (cache misses traverse the interconnect)
        max_cycle = issue_cycle
        for addr, is_write in task.memory_accesses:
            finish_c, _ = self.host.access_memory(addr=addr, is_write=is_write, current_cycle=issue_cycle)
            if finish_c > max_cycle:
                max_cycle = finish_c

        # Host writes back final output if needed
        if task.output_bytes > 0:
            out_lines = max(1, task.output_bytes // 64)
            for l in range(out_lines):
                finish_c, _ = self.host.access_memory(
                    addr=HOST_RESULT_BASE + (l * 64), is_write=True, current_cycle=max_cycle
                )
                if finish_c > max_cycle:
                    max_cycle = finish_c

        return {"finish": max_cycle, "writebacks": 0}

    def _execute_pim(self, task: KernelTask, start_cycle: int) -> Dict[str, int]:
        # Coherence: near-memory PEs read DRAM directly, so any copy of the
        # task's data held in the host caches is invalidated first, and dirty
        # copies are written back over the interconnect.
        ready_cycle = start_cycle
        writebacks = 0
        seen = set()
        for addr, _ in task.memory_accesses:
            line_addr = (addr // 64) * 64
            if line_addr in seen:
                continue
            seen.add(line_addr)
            if self.cache.invalidate_line(line_addr):
                writebacks += 1
                tx_done, _ = self.interconnect.transmit(64, start_cycle)
                wr_done, _, _ = self.memory.access(line_addr, True, tx_done)
                ready_cycle = max(ready_cycle, wr_done)

        finish, _ = self.pim.execute_kernel_in_memory(
            num_ops=task.num_ops,
            memory_accesses=task.memory_accesses,
            result_size_bytes=task.output_bytes,
            current_cycle=ready_cycle
        )
        return {"finish": finish, "writebacks": writebacks}

    def _oracle_target(self, task: KernelTask, now: int) -> str:
        """Tries both placements on copies of the current machine state."""
        finishes = {}
        for target in ("HOST", "PIM"):
            trial = copy.deepcopy(self)
            finishes[target] = trial._execute(task, target, now)["finish"]
        return "PIM" if finishes["PIM"] < finishes["HOST"] else "HOST"

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _schedule_arrival(self, task: KernelTask, delay: int = 0):
        if task.interfering_dma_bytes > 0:
            self.dma.schedule_burst(self.engine, task.interfering_dma_bytes, delay)
        self.engine.schedule(delay, self._on_task_arrival, priority=5,
                             payload={"task": task}, name=f"TaskArrival:{task.name}")

    def _on_task_arrival(self, engine: SimulationEngine, payload: Dict[str, Any]):
        task: KernelTask = payload["task"]
        now = engine.current_cycle
        backlog = self.interconnect.queue_backlog(now)

        if self.placement_engine.policy == PlacementPolicy.ORACLE:
            target = self._oracle_target(task, now)
            decision = self.placement_engine.record_external(
                task.name, target, f"Oracle: trial execution on both targets, {target} finished first"
            )
        else:
            out_lines = max(1, task.output_bytes // 64) if task.output_bytes > 0 else 0
            decision = self.placement_engine.decide(
                kernel_name=task.name,
                num_ops=task.num_ops,
                memory_accesses=task.memory_accesses,
                output_bytes=task.output_bytes,
                current_bus_congestion_cycles=backlog,
                vectorizable=task.vectorizable,
                residency_probe=self.cache.probe_line,
                bank_of=self.memory.bank_of,
                num_banks=self.memory.total_banks,
                output_probe_addrs=[HOST_RESULT_BASE + l * 64 for l in range(out_lines)]
            )

        engine.schedule(
            decision.decision_overhead_cycles, self._on_task_dispatch, priority=5,
            payload={
                "task": task,
                "decision": decision,
                "arrival": now,
                "backlog": backlog,
                "bytes_before": self.interconnect.total_bytes_transferred,
            },
            name=f"TaskDispatch:{task.name}"
        )

    def _on_task_dispatch(self, engine: SimulationEngine, payload: Dict[str, Any]):
        decision: PlacementDecision = payload["decision"]
        outcome = self._execute(payload["task"], decision.target, engine.current_cycle)
        payload["writebacks"] = outcome["writebacks"]
        engine.schedule(outcome["finish"] - engine.current_cycle, self._on_task_complete,
                        priority=5, payload=payload, name=f"TaskComplete:{payload['task'].name}")

    def _on_task_complete(self, engine: SimulationEngine, payload: Dict[str, Any]):
        task: KernelTask = payload["task"]
        decision: PlacementDecision = payload["decision"]
        finish = engine.current_cycle
        elapsed = finish - payload["arrival"]
        bytes_transferred = self.interconnect.total_bytes_transferred - payload["bytes_before"]

        energy_report = EnergyModel.generate_report(
            cycles=elapsed,
            freq_ghz=self.engine.frequency_ghz,
            host_energy_pj=self.host.total_energy_pj,
            interconnect_energy_pj=self.interconnect.total_transfer_energy_pj,
            dram_energy_pj=self.memory.total_dram_energy_pj,
            pim_energy_pj=self.pim.total_energy_pj + decision.decision_overhead_energy_pj,
            bytes_transferred=bytes_transferred
        )

        self.task_records.append(TaskExecutionRecord(
            task_name=task.name,
            category=task.category,
            target=decision.target,
            policy=decision.policy_used.value,
            start_cycle=payload["arrival"],
            end_cycle=finish,
            elapsed_cycles=elapsed,
            bytes_transferred=bytes_transferred,
            decision_overhead_cycles=decision.decision_overhead_cycles,
            decision_reason=decision.reason,
            energy_report=energy_report,
            bus_backlog_at_arrival=payload["backlog"],
            coherence_writebacks=payload["writebacks"],
            predicted_host_cycles=decision.estimate.host_cycles if decision.estimate else None,
            predicted_pim_cycles=decision.estimate.pim_cycles if decision.estimate else None
        ))
        self._last_completion_cycle = finish

        if self._task_queue:
            self._schedule_arrival(self._task_queue.popleft())
        else:
            engine.stop()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _drive(self, tasks: List[KernelTask]):
        if not tasks:
            return
        self._task_queue = deque(tasks)
        self._schedule_arrival(self._task_queue.popleft())
        self.engine.run()
        # Background traffic still queued after the last task is not part of this run
        self.engine.clear_pending()
        self.engine.current_cycle = self._last_completion_cycle

    def run_task(self, task: KernelTask) -> TaskExecutionRecord:
        """
        Executes a single task through the architecture.
        """
        self._drive([task])
        return self.task_records[-1]

    def run_workload(self, tasks: List[KernelTask]) -> SimulationResult:
        """
        Runs an entire workload sequence and produces aggregated results.
        """
        self._drive(tasks)

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
            },
            events_processed=self.engine.stats["total_events_processed"]
        )
        return result

    def reset(self):
        """Resets all subsystems to initial state."""
        self.engine.reset()
        self.interconnect.reset()
        self.memory.reset()
        self.host.reset()
        self.pim.reset()
        self.dma.reset()
        self.placement_engine.reset()
        self.task_records.clear()
        self._task_queue.clear()
        self._last_completion_cycle = 0
