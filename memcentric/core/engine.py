"""
MemCentric-Sim: Core Simulation Engine and Event Management
=============================================================
A high-precision, cycle-accurate discrete-event simulation engine
for modeling memory-centric and near-memory computing architectures.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import heapq


@dataclass(order=True)
class Event:
    """Represents a scheduled event in the simulation timeline."""
    cycle: int
    priority: int = field(default=10)
    event_id: int = field(default=0)
    callback: Callable[..., Any] = field(compare=False, default=None)
    payload: Dict[str, Any] = field(compare=False, default_factory=dict)
    name: str = field(compare=False, default="Event")


class SimulationEngine:
    """
    Cycle-accurate discrete-event engine.
    Maintains a priority queue of events sorted by timestamp (cycle) and priority.
    """
    def __init__(self, frequency_ghz: float = 2.0):
        self.frequency_ghz = frequency_ghz
        self.current_cycle: int = 0
        self._event_queue: List[Event] = []
        self._event_counter: int = 0
        self.is_running: bool = False
        self.stats: Dict[str, Any] = {
            "total_events_processed": 0,
            "max_queue_depth": 0,
        }

    @property
    def current_time_ns(self) -> float:
        """Returns the current simulation time in nanoseconds."""
        return (self.current_cycle / self.frequency_ghz)

    def schedule(self, delay_cycles: int, callback: Callable[..., Any], 
                 priority: int = 10, payload: Optional[Dict[str, Any]] = None, 
                 name: str = "Event") -> Event:
        """Schedules an event to occur `delay_cycles` in the future."""
        assert delay_cycles >= 0, f"Cannot schedule event in the past (delay={delay_cycles})"
        event_cycle = self.current_cycle + delay_cycles
        self._event_counter += 1
        
        event = Event(
            cycle=event_cycle,
            priority=priority,
            event_id=self._event_counter,
            callback=callback,
            payload=payload or {},
            name=name
        )
        heapq.heappush(self._event_queue, event)
        
        if len(self._event_queue) > self.stats["max_queue_depth"]:
            self.stats["max_queue_depth"] = len(self._event_queue)
            
        return event

    def run(self, max_cycles: Optional[int] = None) -> int:
        """
        Executes events in chronological order until the queue is empty
        or `max_cycles` is reached. Returns the total elapsed cycles.
        """
        self.is_running = True
        
        while self._event_queue and self.is_running:
            # Peek at next event
            next_event = self._event_queue[0]
            if max_cycles is not None and next_event.cycle > max_cycles:
                self.current_cycle = max_cycles
                break
                
            event = heapq.heappop(self._event_queue)
            self.current_cycle = event.cycle
            self.stats["total_events_processed"] += 1
            
            if event.callback:
                event.callback(self, event.payload)
                
        self.is_running = False
        return self.current_cycle

    def step(self) -> bool:
        """Executes a single event from the queue. Returns False if queue is empty."""
        if not self._event_queue:
            return False
        event = heapq.heappop(self._event_queue)
        self.current_cycle = event.cycle
        self.stats["total_events_processed"] += 1
        if event.callback:
            event.callback(self, event.payload)
        return True

    def reset(self):
        """Resets engine to initial cycle."""
        self.current_cycle = 0
        self._event_queue.clear()
        self._event_counter = 0
        self.is_running = False
        self.stats["total_events_processed"] = 0
        self.stats["max_queue_depth"] = 0
