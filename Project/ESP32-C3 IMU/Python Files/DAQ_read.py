import time
from daq_reader import DAQReader

def print_MPU_sample(ts, q, aa, gg):
    hz = daq.compute_hz()
    print(
        f"HZ: {hz:6.1f} "
        f"T: {ts:>12d} us "
        f"Q: ({q[0]:10.6f}, {q[1]:10.6f}, {q[2]:10.6f}, {q[3]:10.6f}) "
        f"A: ({aa[0]:6d}, {aa[1]:6d}, {aa[2]:6d}) "
        f"G: ({gg[0]:6d}, {gg[1]:6d}, {gg[2]:6d})"
    )

daq = DAQReader()
daq.start(callback=print_MPU_sample)

# main thread can do other things or sleep
while True:
    time.sleep(1)