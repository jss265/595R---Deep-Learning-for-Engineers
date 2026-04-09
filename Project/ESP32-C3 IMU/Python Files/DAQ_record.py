import time
from daq_reader import DAQReader

def record_MPU_sample(ts, q, aa, gg):
    hz = daq.compute_hz()
    # placeholder: save to file or process for ML later
    print(f"[RECORD] Packet T: {ts}, HZ: {hz:6.1f}")

daq = DAQReader()
daq.start(callback=record_MPU_sample)

while True:
    time.sleep(1)