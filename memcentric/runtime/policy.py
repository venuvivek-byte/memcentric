"""
MemCentric-Sim: Intelligent Computation Placement Engine
=========================================================
Dynamic runtime engine that arbitrates computation placement
(Host vs Near-Memory / PIM) using micro-profiling and cost models.
Explicitly accounts for decision overhead and bus congestion.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from memcentric.runtime.profiler import MicroProfiler, KernelProfile
from memcentric.runtime.cost_model import AnalyticalCostModel, CostEstimate


class PlacementPolicy(str, Enum):
    STATIC_HOST = "STATIC_HOST"
    STATIC_PIM = "STATIC_PIM"
    ROOFLINE_OI = "ROOFLINE_OI"
    DYNAMIC_ANALYTICAL = "DYNAMIC_ANALYTICAL"
    ADAPTIVE_CONGESTION = "ADAPTIVE_CONGESTION"


@dataclass
class PlacementDecision:
    kernel_name: str
    target: str                  # "HOST" or "PIM"
    policy_used: PlacementPolicy
    profile: KernelProfile
    estimate: Optional[CostEstimate]
    decision_overhead_cycles: int
    decision_overhead_energy_pj: float
    reason: str


class PlacementEngine:
    """
    Intelligent Runtime Scheduler that dynamically places tasks.
    """
    def __init__(
        self,
        policy: PlacementPolicy = PlacementPolicy.DYNAMIC_ANALYTICAL,
        profiler: Optional[MicroProfiler] = None,
        cost_model: Optional[AnalyticalCostModel] = None,
        oi_threshold: float = 2.0,       # FLOPs/Byte threshold for Roofline
        decision_cycles: int = 35,       # Latency to compute scheduling heuristic
        decision_energy_pj: float = 80.0 # Energy consumed by decision hardware
    ):
        self.policy = policy
        self.profiler = profiler or MicroProfiler()
        self.cost_model = cost_model or AnalyticalCostModel()
        self.oi_threshold = oi_threshold
        self.decision_cycles = decision_cycles
        self.decision_energy_pj = decision_energy_pj

        # Telemetry & History
        self.decisions_log: List[PlacementDecision] = []
        self.total_overhead_cycles: int = 0
        self.total_overhead_energy_pj: float = 0.0

    def decide(
        self,
        kernel_name: str,
        num_ops: int,
        memory_accesses: List[Tuple[int, bool]],
        output_bytes: int,
        current_bus_congestion_cycles: int = 0
    ) -> PlacementDecision:
        """
        Dynamically analyzes kernel characteristics and chooses placement.
        Incurs decision overhead cycles and energy.
        """
        # 1. Profile kernel
        profile = self.profiler.profile(
            name=kernel_name,
            num_ops=num_ops,
            memory_accesses=memory_accesses,
            output_bytes=output_bytes
        )

        estimate: Optional[CostEstimate] = None
        target = "HOST"
        reason = ""

        # 2. Apply Policy
        if self.policy == PlacementPolicy.STATIC_HOST:
            target = "HOST"
            reason = "Static policy: Baseline Host Execution"

        elif self.policy == PlacementPolicy.STATIC_PIM:
            target = "PIM"
            reason = "Static policy: Always Offload to PIM"

        elif self.policy == PlacementPolicy.ROOFLINE_OI:
            # Roofline model: If low operational intensity (memory bound), place in PIM
            if profile.operational_intensity < self.oi_threshold:
                target = "PIM"
                reason = f"Operational Intensity ({profile.operational_intensity:.2f} FLOP/B) < Threshold ({self.oi_threshold})"
            else:
                target = "HOST"
                reason = f"Operational Intensity ({profile.operational_intensity:.2f} FLOP/B) >= Threshold ({self.oi_threshold})"

        elif self.policy == PlacementPolicy.DYNAMIC_ANALYTICAL:
            estimate = self.cost_model.estimate(profile, current_bus_queue_cycles=0)
            target = estimate.recommended_placement
            reason = (
                f"Analytical model chose {target} (Host cycles={estimate.host_cycles}, "
                f"PIM cycles={estimate.pim_cycles}, margin={estimate.confidence_margin:.1%})"
            )

        elif self.policy == PlacementPolicy.ADAPTIVE_CONGESTION:
            estimate = self.cost_model.estimate(
                profile, current_bus_queue_cycles=current_bus_congestion_cycles
            )
            # If interconnect bus is heavily congested, bias towards PIM offload
            if current_bus_congestion_cycles > 50 and profile.reduction_ratio < 0.2:
                target = "PIM"
                reason = f"Adaptive: Interconnect saturated (queue={current_bus_congestion_cycles}c), offloading to relieve bus"
            else:
                target = estimate.recommended_placement
                reason = f"Adaptive Cost Model selected {target}"

        decision = PlacementDecision(
            kernel_name=kernel_name,
            target=target,
            policy_used=self.policy,
            profile=profile,
            estimate=estimate,
            decision_overhead_cycles=self.decision_cycles,
            decision_overhead_energy_pj=self.decision_energy_pj,
            reason=reason
        )

        self.decisions_log.append(decision)
        self.total_overhead_cycles += self.decision_cycles
        self.total_overhead_energy_pj += self.decision_energy_pj

        return decision

    def reset(self):
        self.decisions_log.clear()
        self.total_overhead_cycles = 0
        self.total_overhead_energy_pj = 0.0

    def get_summary(self) -> Dict[str, Any]:
        pim_count = sum(1 for d in self.decisions_log if d.target == "PIM")
        host_count = sum(1 for d in self.decisions_log if d.target == "HOST")
        total = len(self.decisions_log)
        return {
            "total_decisions": total,
            "placed_on_pim": pim_count,
            "placed_on_host": host_count,
            "pim_offload_ratio": round(pim_count / total, 3) if total > 0 else 0.0,
            "total_overhead_cycles": self.total_overhead_cycles,
            "total_overhead_energy_uj": round(self.total_overhead_energy_pj / 1e6, 4)
        }
