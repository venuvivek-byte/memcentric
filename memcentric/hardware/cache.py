"""
MemCentric-Sim: Cache Hierarchy Model
======================================
Detailed set-associative cache simulation with LRU replacement,
dirty writebacks, tag lookup, and hit/miss latency modeling.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import copy
import math


@dataclass
class CacheLine:
    tag: int = -1
    valid: bool = False
    dirty: bool = False
    last_access_cycle: int = 0


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    writebacks: int = 0
    reads: int = 0
    writes: int = 0

    @property
    def total_accesses(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return (self.hits / self.total_accesses) if self.total_accesses > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        return 1.0 - self.hit_rate


class CacheLevel:
    """
    Models an N-way set-associative cache with LRU replacement.
    """
    def __init__(
        self,
        name: str,
        size_bytes: int,
        associativity: int,
        line_size_bytes: int = 64,
        hit_latency_cycles: int = 4,
        energy_per_hit_pj: float = 5.0,
        energy_per_miss_pj: float = 12.0
    ):
        self.name = name
        self.size_bytes = size_bytes
        self.associativity = associativity
        self.line_size_bytes = line_size_bytes
        self.hit_latency_cycles = hit_latency_cycles
        self.energy_per_hit_pj = energy_per_hit_pj
        self.energy_per_miss_pj = energy_per_miss_pj

        assert size_bytes > 0 and (size_bytes & (size_bytes - 1)) == 0, "Cache size must be power of 2"
        assert line_size_bytes > 0 and (line_size_bytes & (line_size_bytes - 1)) == 0, "Line size must be power of 2"

        self.num_lines = size_bytes // line_size_bytes
        self.num_sets = self.num_lines // associativity
        assert self.num_sets > 0, "Cache size too small for specified associativity"

        self.offset_bits = int(math.log2(line_size_bytes))
        self.index_bits = int(math.log2(self.num_sets))
        self.index_mask = self.num_sets - 1

        # Sets of cache lines
        self.sets: List[List[CacheLine]] = [
            [CacheLine() for _ in range(associativity)] for _ in range(self.num_sets)
        ]
        self.stats = CacheStats()

    def _parse_address(self, addr: int) -> Tuple[int, int, int]:
        """Splits an address into (tag, index, offset)."""
        offset = addr & ((1 << self.offset_bits) - 1)
        index = (addr >> self.offset_bits) & self.index_mask
        tag = addr >> (self.offset_bits + self.index_bits)
        return tag, index, offset

    def access(self, addr: int, is_write: bool, cycle: int) -> Tuple[bool, bool, int]:
        """
        Simulate cache access.
        Returns: (hit: bool, evicted_dirty: bool, evicted_addr: int)
        """
        tag, index, _ = self._parse_address(addr)
        cache_set = self.sets[index]

        if is_write:
            self.stats.writes += 1
        else:
            self.stats.reads += 1

        # Check for hit
        for line in cache_set:
            if line.valid and line.tag == tag:
                self.stats.hits += 1
                line.last_access_cycle = cycle
                if is_write:
                    line.dirty = True
                return True, False, 0

        # Cache Miss
        self.stats.misses += 1
        self.stats.evictions += 1

        # Find victim (invalid line first, else LRU)
        victim_idx = -1
        lru_cycle = float("inf")

        for i, line in enumerate(cache_set):
            if not line.valid:
                victim_idx = i
                break
            if line.last_access_cycle < lru_cycle:
                lru_cycle = line.last_access_cycle
                victim_idx = i

        victim = cache_set[victim_idx]
        evicted_dirty = victim.valid and victim.dirty
        evicted_addr = (
            ((victim.tag << self.index_bits) | index) << self.offset_bits
            if victim.valid else 0
        )

        if evicted_dirty:
            self.stats.writebacks += 1

        # Install new line
        victim.valid = True
        victim.tag = tag
        victim.dirty = is_write
        victim.last_access_cycle = cycle

        return False, evicted_dirty, evicted_addr

    def __deepcopy__(self, memo):
        # Fast path for trial executions: rebuild lines directly instead of
        # letting copy.deepcopy walk ~270k dataclass instances.
        clone = copy.copy(self)
        clone.sets = [
            [CacheLine(l.tag, l.valid, l.dirty, l.last_access_cycle) for l in s]
            for s in self.sets
        ]
        clone.stats = copy.copy(self.stats)
        memo[id(self)] = clone
        return clone

    def _find_line(self, addr: int) -> Optional[CacheLine]:
        tag, index, _ = self._parse_address(addr)
        for line in self.sets[index]:
            if line.valid and line.tag == tag:
                return line
        return None

    def probe(self, addr: int) -> Tuple[bool, bool]:
        """Side-effect-free lookup. Returns (resident, dirty); no LRU or stats update."""
        line = self._find_line(addr)
        if line is None:
            return False, False
        return True, line.dirty

    def invalidate(self, addr: int) -> bool:
        """Invalidates the line holding addr. Returns True if it was dirty."""
        line = self._find_line(addr)
        if line is None:
            return False
        was_dirty = line.dirty
        line.valid = False
        line.dirty = False
        line.tag = -1
        return was_dirty

    def flush(self) -> int:
        """Flushes all lines and returns count of dirty lines written back."""
        dirty_count = 0
        for cache_set in self.sets:
            for line in cache_set:
                if line.valid and line.dirty:
                    dirty_count += 1
                line.valid = False
                line.dirty = False
                line.tag = -1
        return dirty_count

    def reset_stats(self):
        self.stats = CacheStats()


class CacheHierarchy:
    """
    Multi-level cache hierarchy (L1 Data, L2, L3 Unified).
    Handles lookups sequentially or inclusively.
    """
    def __init__(
        self,
        l1_size_kb: int = 32,
        l1_assoc: int = 8,
        l1_latency: int = 4,
        l2_size_kb: int = 512,
        l2_assoc: int = 8,
        l2_latency: int = 14,
        l3_size_mb: int = 16,
        l3_assoc: int = 16,
        l3_latency: int = 40
    ):
        self.l1d = CacheLevel(
            name="L1D",
            size_bytes=l1_size_kb * 1024,
            associativity=l1_assoc,
            hit_latency_cycles=l1_latency,
            energy_per_hit_pj=4.0,
            energy_per_miss_pj=8.0
        )
        self.l2 = CacheLevel(
            name="L2",
            size_bytes=l2_size_kb * 1024,
            associativity=l2_assoc,
            hit_latency_cycles=l2_latency,
            energy_per_hit_pj=15.0,
            energy_per_miss_pj=30.0
        )
        self.l3 = CacheLevel(
            name="L3",
            size_bytes=l3_size_mb * 1024 * 1024,
            associativity=l3_assoc,
            hit_latency_cycles=l3_latency,
            energy_per_hit_pj=60.0,
            energy_per_miss_pj=120.0
        )
        self.levels = [self.l1d, self.l2, self.l3]

    def access(self, addr: int, is_write: bool, cycle: int) -> Tuple[int, int, bool]:
        """
        Traverses hierarchy.
        Returns: (latency_cycles, energy_pj, is_full_miss_to_memory)
        """
        cumulative_latency = 0
        cumulative_energy = 0.0

        # L1 Access
        cumulative_latency += self.l1d.hit_latency_cycles
        l1_hit, l1_evicted_dirty, _ = self.l1d.access(addr, is_write, cycle)
        if l1_hit:
            cumulative_energy += self.l1d.energy_per_hit_pj
            return cumulative_latency, int(cumulative_energy), False

        cumulative_energy += self.l1d.energy_per_miss_pj

        # L2 Access
        cumulative_latency += self.l2.hit_latency_cycles
        l2_hit, _, _ = self.l2.access(addr, is_write, cycle)
        if l2_hit:
            cumulative_energy += self.l2.energy_per_hit_pj
            return cumulative_latency, int(cumulative_energy), False

        cumulative_energy += self.l2.energy_per_miss_pj

        # L3 Access
        cumulative_latency += self.l3.hit_latency_cycles
        l3_hit, _, _ = self.l3.access(addr, is_write, cycle)
        if l3_hit:
            cumulative_energy += self.l3.energy_per_hit_pj
            return cumulative_latency, int(cumulative_energy), False

        cumulative_energy += self.l3.energy_per_miss_pj

        # Full Miss - must fetch from main memory over bus
        return cumulative_latency, int(cumulative_energy), True

    def probe_line(self, addr: int) -> Tuple[bool, bool]:
        """
        Side-effect-free residency check across all levels.
        Returns (resident_anywhere, dirty_anywhere).
        """
        resident = False
        dirty = False
        for lvl in self.levels:
            r, d = lvl.probe(addr)
            resident = resident or r
            dirty = dirty or d
        return resident, dirty

    def invalidate_line(self, addr: int) -> bool:
        """
        Removes a line from every level (coherence invalidation before a
        near-memory kernel touches it). Returns True if any copy was dirty
        and must be written back to DRAM.
        """
        dirty = False
        for lvl in self.levels:
            if lvl.invalidate(addr):
                dirty = True
        return dirty

    @property
    def hit_path_latency(self) -> int:
        """Lookup latency through every level before a miss reaches memory."""
        return sum(lvl.hit_latency_cycles for lvl in self.levels)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        return {
            lvl.name: {
                "hits": lvl.stats.hits,
                "misses": lvl.stats.misses,
                "hit_rate": round(lvl.stats.hit_rate, 4),
                "writebacks": lvl.stats.writebacks,
            }
            for lvl in self.levels
        }
