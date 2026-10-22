import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memcentric.hardware.memory import MemorySubsystem, MemoryTiming


def test_dram_row_buffer_hit_and_miss():
    mem = MemorySubsystem(num_channels=2, banks_per_channel=4)

    # First access to row 0 in bank 0 -> Row buffer empty (tRCD + tCAS + tBURST)
    addr1 = 0x1000
    finish1, lat1, energy1 = mem.access(addr=addr1, is_write=False, current_cycle=0)
    assert lat1 == (mem.timing.tRCD + mem.timing.tCAS + mem.timing.tBURST)
    assert mem.total_row_misses == 1

    # Second access to same row in same bank -> Row buffer hit (tCAS + tBURST)
    addr2 = 0x1040 # same row
    finish2, lat2, energy2 = mem.access(addr=addr2, is_write=False, current_cycle=finish1)
    assert lat2 == (mem.timing.tCAS + mem.timing.tBURST)
    assert mem.total_row_hits == 1

    # Access to different row in same bank -> Conflict (tRP + tRCD + tCAS + tBURST)
    # Row size is 2048 bytes, offset by row size * num_channels * num_banks
    row_stride = mem.row_size_bytes * mem.total_banks
    addr3 = addr1 + row_stride
    finish3, lat3, energy3 = mem.access(addr=addr3, is_write=False, current_cycle=finish2)
    expected_conflict_lat = (mem.timing.tRP + mem.timing.tRCD + mem.timing.tCAS + mem.timing.tBURST)
    assert lat3 == expected_conflict_lat
    assert mem.total_row_conflicts == 1


if __name__ == "__main__":
    test_dram_row_buffer_hit_and_miss()
    print("Memory tests passed successfully!")
