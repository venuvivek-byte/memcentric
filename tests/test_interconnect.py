import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memcentric.hardware.interconnect import Interconnect


def test_interconnect_transmission_and_queuing():
    # 64 GB/s at 2.0 GHz = 32 Bytes per cycle
    bus = Interconnect(bandwidth_gbps=64.0, base_latency_cycles=40, host_freq_ghz=2.0)
    assert bus.bytes_per_cycle == 32.0

    # 64 bytes takes 2 cycles of transmission + 40 cycles base latency = 42 cycles
    finish1, lat1 = bus.transmit(size_bytes=64, current_cycle=100)
    assert finish1 == 142
    assert lat1 == 42
    assert bus.total_bytes_transferred == 64

    # Second transmission scheduled before finish1 arrives -> must queue
    # Scheduled at cycle 110, but bus busy until 142 -> queuing delay = 32 cycles
    finish2, lat2 = bus.transmit(size_bytes=64, current_cycle=110)
    assert finish2 == 142 + 40 + 2
    assert bus.total_queuing_delay_cycles == 32


if __name__ == "__main__":
    test_interconnect_transmission_and_queuing()
    print("Interconnect tests passed successfully!")
