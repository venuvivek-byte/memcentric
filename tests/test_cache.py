import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memcentric.hardware.cache import CacheLevel, CacheHierarchy


def test_cache_hit_and_miss():
    cache = CacheLevel(name="TestL1", size_bytes=1024, associativity=2, line_size_bytes=64)
    # First access - miss
    hit, evicted_dirty, _ = cache.access(addr=0x1000, is_write=False, cycle=10)
    assert not hit
    assert not evicted_dirty
    assert cache.stats.misses == 1

    # Second access to same line - hit
    hit, _, _ = cache.access(addr=0x1000, is_write=False, cycle=20)
    assert hit
    assert cache.stats.hits == 1

    # Offset access within same line - hit
    hit, _, _ = cache.access(addr=0x1020, is_write=False, cycle=30)
    assert hit
    assert cache.stats.hits == 2


def test_cache_hierarchy_traversal():
    hierarchy = CacheHierarchy()
    # Initial access should miss L1, L2, L3 to memory
    lat, energy, full_miss = hierarchy.access(addr=0x200000, is_write=False, cycle=1)
    assert full_miss is True
    assert lat > 0

    # Repeat access should now hit in L1
    lat2, energy2, full_miss2 = hierarchy.access(addr=0x200000, is_write=False, cycle=50)
    assert full_miss2 is False
    assert lat2 == hierarchy.l1d.hit_latency_cycles


if __name__ == "__main__":
    test_cache_hit_and_miss()
    test_cache_hierarchy_traversal()
    print("Cache tests passed successfully!")
