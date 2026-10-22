"""
MemCentric-Sim: A Cycle-Accurate Simulation Framework for Near-Memory Computing
and Dynamic Computation Placement.
"""

from memcentric.system import MemoryCentricSystem, SimulationResult
from memcentric.runtime.policy import PlacementPolicy

__version__ = "1.0.0"
__all__ = ["MemoryCentricSystem", "SimulationResult", "PlacementPolicy"]
