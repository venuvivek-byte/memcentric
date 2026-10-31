"""
MemCentric-Sim: Background DMA Agent
=====================================
Models another device (storage controller, NIC, accelerator) that shares the
Host-Memory interconnect. Its bursts are scheduled as discrete events and
occupy the link, creating the bus congestion that the runtime placement
engine has to observe and react to.
"""

from typing import Any, Dict
from memcentric.core.engine import SimulationEngine
from memcentric.hardware.interconnect import Interconnect


class BackgroundDMA:
    def __init__(self, interconnect: Interconnect, packet_bytes: int = 4096):
        self.interconnect = interconnect
        self.packet_bytes = packet_bytes
        self.bursts_issued: int = 0
        self.bytes_issued: int = 0

    def schedule_burst(self, engine: SimulationEngine, total_bytes: int, delay_cycles: int = 0):
        """Schedules a DMA burst of total_bytes, split into link-sized packets."""
        engine.schedule(
            delay_cycles,
            self._on_burst,
            priority=0,  # lands on the link before work scheduled for the same cycle
            payload={"bytes": total_bytes},
            name="DMABurst"
        )

    def schedule_periodic(self, engine: SimulationEngine, burst_bytes: int,
                          period_cycles: int, count: int, start_delay: int = 0):
        """Schedules `count` bursts, one every `period_cycles`."""
        for i in range(count):
            self.schedule_burst(engine, burst_bytes, start_delay + i * period_cycles)

    def _on_burst(self, engine: SimulationEngine, payload: Dict[str, Any]):
        remaining = payload["bytes"]
        while remaining > 0:
            chunk = min(self.packet_bytes, remaining)
            self.interconnect.transmit(chunk, engine.current_cycle, account=False)
            remaining -= chunk
        self.bursts_issued += 1
        self.bytes_issued += payload["bytes"]

    def reset(self):
        self.bursts_issued = 0
        self.bytes_issued = 0
