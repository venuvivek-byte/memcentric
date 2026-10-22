"""
MemCentric-Sim Workload: Random Access & Graph Traversal
========================================================
Characteristics:
- Random Access / Irregular Pointer-Chasing: Poor spatial & temporal locality.
- Memory Latency Bound: High cache miss rates thrash Host L1/L2/L3 caches.
- Bank-Level Parallelism: Benefits substantially from PIM PEs co-located with memory banks,
  avoiding interconnect roundtrip serialization.
"""

import random
from typing import List, Tuple
from memcentric.workloads.base import KernelTask


def generate_graph_bfs_task(
    num_nodes: int = 10000,
    edges_per_node: int = 8,
    frontier_size: int = 500,
    base_addr: int = 0xC00000,
    seed: int = 42
) -> KernelTask:
    """
    Simulates Graph BFS frontier expansion (Compressed Sparse Row / Pointer Chasing).
    """
    rng = random.Random(seed)
    accesses: List[Tuple[int, bool]] = []
    node_size = 32 # Node struct size

    # Random frontier node accesses (pointer chasing through adjacency list)
    for _ in range(frontier_size):
        curr_node = rng.randint(0, num_nodes - 1)
        node_addr = base_addr + (curr_node * node_size)
        accesses.append((node_addr, False))

        # Access random neighbors
        for _ in range(edges_per_node):
            neighbor = rng.randint(0, num_nodes - 1)
            neighbor_addr = base_addr + (neighbor * node_size)
            accesses.append((neighbor_addr, False))
            # Mark visited
            accesses.append((neighbor_addr + 16, True))

    num_ops = frontier_size * edges_per_node * 4
    output_bytes = frontier_size * 4 # New frontier IDs

    return KernelTask(
        name=f"Graph_BFS_{num_nodes}_Nodes",
        category="RANDOM_ACCESS",
        num_ops=num_ops,
        memory_accesses=accesses,
        output_bytes=output_bytes,
        description=f"Graph BFS Traversal ({num_nodes} nodes, irregular random pointer chasing)"
    )


def generate_sparse_embedding_lookup_task(
    table_entries: int = 20000,
    lookup_count: int = 1000,
    embedding_dim_bytes: int = 64, # 1 cache line per vector
    base_addr: int = 0xE00000,
    seed: int = 123
) -> KernelTask:
    """
    Simulates DLRM / Recommender System sparse embedding bag gather:
    Fetches random embedding vectors from high-capacity memory and reduces them.
    """
    rng = random.Random(seed)
    accesses: List[Tuple[int, bool]] = []

    for _ in range(lookup_count):
        entry_idx = rng.randint(0, table_entries - 1)
        addr = base_addr + (entry_idx * embedding_dim_bytes)
        accesses.append((addr, False))

    num_ops = lookup_count * (embedding_dim_bytes // 4) # Vector accumulate ops
    output_bytes = embedding_dim_bytes # Single pooled embedding vector

    return KernelTask(
        name=f"Sparse_Embedding_Lookup_{lookup_count}",
        category="RANDOM_ACCESS",
        num_ops=num_ops,
        memory_accesses=accesses,
        output_bytes=output_bytes,
        description=f"Sparse Embedding Table Gathering & Pooling ({lookup_count} random vector lookups)"
    )
