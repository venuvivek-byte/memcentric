"""
MemCentric-Sim: Intelligent Computation Placement Engine
=========================================================
Dynamic runtime engine that arbitrates computation placement
(Host vs Near-Memory / PIM) using micro-profiling and cost models.
Explicitly accounts for decision overhead and bus congestion.

Decision overhead per policy:
  STATIC_HOST / STATIC_PIM  no profiling, no decision hardware: 0 cycles
  ROOFLINE_OI               profiling pass + threshold compare
  DYNAMIC / ADAPTIVE        profiling pass + cost-model evaluation
  ORACLE                    evaluation-only upper bound (decided by the
                            system by trial execution), charged 0 cycles
The profiling pass cost scales with the number of sampled accesses and
cache-tag probes (see MicroProfiler).
"""

from enum import Enum
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any
from memcentric.runtime.profiler import MicroProfiler, KernelProfile
from memcentric.runtime.cost_model import AnalyticalCostModel, CostEstimate


class PlacementPolicy(str, Enum):
    STATIC_HOST = "STATIC_HOST"
    STATIC_PIM = "STATIC_PIM"
    ROOFLINE_OI = "ROOFLINE_OI"
    DYNAMIC_ANALYTICAL = "DYNAMIC_ANALYTICAL"
    ADAPTIVE_CONGESTION = "ADAPTIVE_CONGESTION"
    ORACLE = "ORACLE"


@dataclass
class PlacementDecision:
    kernel_name: str
    target: str                  # "HOST" or "PIM"
    policy_used: PlacementPolicy
    profile: Optional[KernelProfile]
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
        decision_cycles: int = 35,       # Latency to evaluate the cost model
        decision_energy_pj: float = 80.0,# Energy consumed by decision hardware
        compare_cycles: int = 5,         # Latency of the roofline threshold compare
        compare_energy_pj: float = 10.0
    ):
        self.policy = policy
        self.profiler = profiler or MicroProfiler()
        self.cost_model = cost_model or AnalyticalCostModel()
        self.oi_threshold = oi_threshold
        self.decision_cycles = decision_cycles
        self.decision_energy_pj = decision_energy_pj
        self.compare_cycles = compare_cycles
        self.compare_energy_pj = compare_energy_pj

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
        current_bus_congestion_cycles: int = 0,
        vectorizable: bool = True,
        residency_probe: Optional[Callable[[int], Tuple[bool, bool]]] = None,
        bank_of: Optional[Callable[[int], int]] = None,
        num_banks: int = 32,
        output_probe_addrs: Optional[List[int]] = None
    ) -> PlacementDecision:
        """
        Dynamically analyzes kernel characteristics and chooses placement.
        Incurs decision overhead cycles and energy.
        """
        profile: Optional[KernelProfile] = None
        estimate: Optional[CostEstimate] = None
        target = "HOST"
        reason = ""
        overhead_cycles = 0
        overhead_energy = 0.0

        if self.policy in (PlacementPolicy.STATIC_HOST, PlacementPolicy.STATIC_PIM):
            target = "HOST" if self.policy == PlacementPolicy.STATIC_HOST else "PIM"
            reason = (
                "Static policy: Baseline Host Execution" if target == "HOST"
                else "Static policy: Always Offload to PIM"
            )
        elif self.policy == PlacementPolicy.ORACLE:
            raise ValueError("ORACLE placement is decided by the system via record_external()")
        else:
            # 1. Profile kernel (sampled address stream + data-location probes)
            profile = self.profiler.profile(
                name=kernel_name,
                num_ops=num_ops,
                memory_accesses=memory_accesses,
                output_bytes=output_bytes,
                vectorizable=vectorizable,
                residency_probe=residency_probe,
                bank_of=bank_of,
                num_banks=num_banks,
                output_probe_addrs=output_probe_addrs
            )
            overhead_cycles = profile.profiling_cycles
            overhead_energy = profile.profiling_energy_pj

            # 2. Apply Policy
            if self.policy == PlacementPolicy.ROOFLINE_OI:
                overhead_cycles += self.compare_cycles
                overhead_energy += self.compare_energy_pj
                # Roofline model: If low operational intensity (memory bound), place in PIM
                if profile.operational_intensity < self.oi_threshold:
                    target = "PIM"
                    reason = f"Operational Intensity ({profile.operational_intensity:.2f} FLOP/B) < Threshold ({self.oi_threshold})"
                else:
                    target = "HOST"
                    reason = f"Operational Intensity ({profile.operational_intensity:.2f} FLOP/B) >= Threshold ({self.oi_threshold})"

            elif self.policy == PlacementPolicy.DYNAMIC_ANALYTICAL:
                overhead_cycles += self.decision_cycles
                overhead_energy += self.decision_energy_pj
                estimate = self.cost_model.estimate(
                    profile, current_bus_queue_cycles=current_bus_congestion_cycles
                )
                target = estimate.recommended_placement
                reason = (
                    f"Cost model chose {target} (Host~{estimate.host_cycles:,}c, PIM~{estimate.pim_cycles:,}c, "
                    f"margin {estimate.confidence_margin:.0%}; cached {profile.resident_fraction:.0%}, "
                    f"dirty {profile.dirty_fraction:.0%}, bus queue {current_bus_congestion_cycles:,}c)"
                )

            elif self.policy == PlacementPolicy.ADAPTIVE_CONGESTION:
                overhead_cycles += self.decision_cycles
                overhead_energy += self.decision_energy_pj
                estimate = self.cost_model.estimate(
                    profile, current_bus_queue_cycles=current_bus_congestion_cycles
                )
                # If interconnect bus is heavily congested, bias towards PIM offload
                if (current_bus_congestion_cycles > 50 and profile.reduction_ratio < 0.2
                        and profile.dirty_fraction == 0.0):
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
            decision_overhead_cycles=overhead_cycles,
            decision_overhead_energy_pj=overhead_energy,
            reason=reason
        )
        self._log(decision)
        return decision

    def record_external(self, kernel_name: str, target: str, reason: str) -> PlacementDecision:
        """Logs a placement chosen outside the engine (the evaluation oracle)."""
        decision = PlacementDecision(
            kernel_name=kernel_name,
            target=target,
            policy_used=self.policy,
            profile=None,
            estimate=None,
            decision_overhead_cycles=0,
            decision_overhead_energy_pj=0.0,
            reason=reason
        )
        self._log(decision)
        return decision

    def _log(self, decision: PlacementDecision):
        self.decisions_log.append(decision)
        self.total_overhead_cycles += decision.decision_overhead_cycles
        self.total_overhead_energy_pj += decision.decision_overhead_energy_pj

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
