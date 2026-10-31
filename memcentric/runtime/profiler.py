"""
MemCentric-Sim: Micro-Profiler for Runtime Task Analysis
=========================================================
Extracts key data-movement and computation metrics from a kernel's address
stream before launch, using a sampled subset of it:
- Working Set Size (unique lines, estimated from the sample)
- Operational Intensity (FLOPs / Byte)
- Memory Access Stride & Spatial Locality
- Temporal Reuse Factor
- Reduction Ratio (Output Bytes / Input Bytes)
- Data location: fraction of the working set already resident (and dirty)
  in the host caches, and the load on the busiest DRAM bank

Profiling is not free: every sampled access and every cache-tag probe costs
cycles and energy, which the placement engine charges to the task.
"""

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class KernelProfile:
    name: str
    num_ops: int
    input_bytes: int
    output_bytes: int
    unique_lines: int
    operational_intensity: float  # FLOPs per Byte
    temporal_reuse_factor: float  # Average accesses per unique line
    reduction_ratio: float        # output_bytes / input_bytes
    is_sequential_stride: bool
    total_accesses: int = 0
    sampled_accesses: int = 0
    vectorizable: bool = True
    resident_fraction: float = 0.0          # working set already in host caches
    dirty_fraction: float = 0.0             # working set dirty in host caches
    output_resident_fraction: float = 0.0   # host result buffer already cached
    max_bank_accesses: int = 0              # accesses landing on the busiest DRAM bank
    pim_row_hit_fraction: float = 0.0       # open-row hits for in-memory execution
    host_row_hit_fraction: float = 0.0      # open-row hits for host cache misses
    profiling_cycles: int = 0
    profiling_energy_pj: float = 0.0


class MicroProfiler:
    """
    Lightweight hardware/runtime profiler modeling metadata sampling overhead.

    sample_rate        fraction of the address stream inspected (1.0 = all)
    profiling_latency_cycles  fixed start-up cost of a profiling pass
    cycles_per_sample  cost of processing one sampled access or tag probe
    """
    STRIDE_WINDOW = 32
    ROW_WINDOW = 128

    def __init__(
        self,
        sample_rate: float = 0.1,
        profiling_latency_cycles: int = 25,
        cycles_per_sample: float = 0.5,
        energy_per_sample_pj: float = 2.0,
        seed: int = 0
    ):
        assert 0.0 < sample_rate <= 1.0
        self.sample_rate = sample_rate
        self.profiling_latency_cycles = profiling_latency_cycles
        self.cycles_per_sample = cycles_per_sample
        self.energy_per_sample_pj = energy_per_sample_pj
        self.seed = seed

    def _sample(self, n: int) -> List[int]:
        if self.sample_rate >= 1.0:
            return list(range(n))
        m = max(1, math.ceil(n * self.sample_rate))
        rng = random.Random(self.seed + n)
        return sorted(rng.sample(range(n), m))

    def profile(
        self,
        name: str,
        num_ops: int,
        memory_accesses: List[Tuple[int, bool]],
        output_bytes: int,
        line_size_bytes: int = 64,
        vectorizable: bool = True,
        residency_probe: Optional[Callable[[int], Tuple[bool, bool]]] = None,
        bank_of: Optional[Callable[[int], int]] = None,
        num_banks: int = 32,
        output_probe_addrs: Optional[List[int]] = None,
        row_size_bytes: int = 2048
    ) -> KernelProfile:
        """
        Profiles a sample of the memory access trace to extract data-local characteristics.
        """
        total_accesses = len(memory_accesses)
        if total_accesses == 0:
            return KernelProfile(
                name=name,
                num_ops=num_ops,
                input_bytes=0,
                output_bytes=output_bytes,
                unique_lines=0,
                operational_intensity=float("inf"),
                temporal_reuse_factor=0.0,
                reduction_ratio=0.0,
                is_sequential_stride=True,
                vectorizable=vectorizable,
                profiling_cycles=self.profiling_latency_cycles
            )

        indices = self._sample(total_accesses)
        q = len(indices) / total_accesses

        line_counts: Dict[int, int] = {}
        bank_counts: Dict[int, int] = {}
        for i in indices:
            addr = memory_accesses[i][0]
            line = addr // line_size_bytes
            line_counts[line] = line_counts.get(line, 0) + 1
            if bank_of is not None:
                b = bank_of(addr)
                bank_counts[b] = bank_counts.get(b, 0) + 1

        sampled_unique = len(line_counts)
        if q >= 1.0:
            unique_est = sampled_unique
        else:
            # Lines seen 3+ times in the sample are hot: assume all of them
            # were caught and remove their share of the accesses. For the
            # rest, if each line is touched c times, the sample holds
            # f1 ~ U*c*q*(1-q)^(c-1) singletons and f2 ~ U*C(c,2)*q^2*(1-q)^(c-2)
            # doubletons, so c ~ 1 + 2*(1-q)*f2 / (q*f1) and U ~ accesses / c.
            hot = [c for c in line_counts.values() if c >= 3]
            f1 = sum(1 for c in line_counts.values() if c == 1)
            f2 = sum(1 for c in line_counts.values() if c == 2)
            rest_accesses = max(f1 + 2 * f2, int(total_accesses - sum(hot) / q))
            if f1 + f2 == 0:
                rest = 0
            elif f1 == 0:
                rest = f2  # every cold line seen twice: reuse is high, sample caught them
            else:
                reuse = 1.0 + 2.0 * (1.0 - q) * f2 / (q * f1)
                rest = max(f1 + f2, int(round(rest_accesses / reuse)))
            unique_est = max(1, min(total_accesses, len(hot) + rest))

        # Where the data currently lives
        probes = 0
        resident_fraction = dirty_fraction = 0.0
        if residency_probe is not None:
            resident = dirty = 0
            for line in line_counts:
                r, d = residency_probe(line * line_size_bytes)
                resident += r
                dirty += d
            probes += sampled_unique
            resident_fraction = resident / sampled_unique
            dirty_fraction = dirty / sampled_unique

        output_resident_fraction = 0.0
        if residency_probe is not None and output_probe_addrs:
            sampled_out = [output_probe_addrs[i] for i in self._sample(len(output_probe_addrs))]
            hits = sum(1 for a in sampled_out if residency_probe(a)[0])
            probes += len(sampled_out)
            output_resident_fraction = hits / len(sampled_out)

        if bank_counts:
            max_bank_accesses = int(math.ceil(max(bank_counts.values()) / q))
        else:
            max_bank_accesses = int(math.ceil(total_accesses / num_banks))

        # Stride analysis on a short contiguous window of the stream
        window = memory_accesses[:self.STRIDE_WINDOW]
        strides = [abs(window[i][0] - window[i - 1][0]) for i in range(1, len(window))]
        seq_strides = sum(1 for s in strides if 0 <= s <= line_size_bytes)
        is_sequential = (seq_strides / len(strides)) > 0.7 if strides else True

        # Row-buffer locality on a contiguous window, replaying the bank
        # mapping: every access for PIM, first touches of a line for the host
        # (later touches hit in the host caches).
        row_window = memory_accesses[:self.ROW_WINDOW]
        pim_hit = pim_n = host_hit = host_n = 0
        if bank_of is not None:
            pim_open: Dict[int, int] = {}
            host_open: Dict[int, int] = {}
            touched = set()
            for addr, _ in row_window:
                b, r = bank_of(addr), addr // row_size_bytes
                if b in pim_open:
                    pim_n += 1
                    pim_hit += pim_open[b] == r
                pim_open[b] = r
                line = addr // line_size_bytes
                if line not in touched:
                    touched.add(line)
                    if b in host_open:
                        host_n += 1
                        host_hit += host_open[b] == r
                    host_open[b] = r
        pim_row_hit = pim_hit / pim_n if pim_n else 0.0
        host_row_hit = host_hit / host_n if host_n else 0.0

        input_bytes = unique_est * line_size_bytes
        oi = num_ops / max(1, input_bytes)
        reuse = total_accesses / unique_est
        red_ratio = output_bytes / max(1, input_bytes)

        work_items = len(indices) + probes + len(window) + (len(row_window) if bank_of else 0)
        profiling_cycles = self.profiling_latency_cycles + int(math.ceil(work_items * self.cycles_per_sample))
        profiling_energy_pj = work_items * self.energy_per_sample_pj

        return KernelProfile(
            name=name,
            num_ops=num_ops,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            unique_lines=unique_est,
            operational_intensity=oi,
            temporal_reuse_factor=reuse,
            reduction_ratio=red_ratio,
            is_sequential_stride=is_sequential,
            total_accesses=total_accesses,
            sampled_accesses=len(indices),
            vectorizable=vectorizable,
            resident_fraction=resident_fraction,
            dirty_fraction=dirty_fraction,
            output_resident_fraction=output_resident_fraction,
            max_bank_accesses=max_bank_accesses,
            pim_row_hit_fraction=pim_row_hit,
            host_row_hit_fraction=host_row_hit,
            profiling_cycles=profiling_cycles,
            profiling_energy_pj=profiling_energy_pj
        )
