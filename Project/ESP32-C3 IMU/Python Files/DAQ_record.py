import msvcrt
import winsound
import time
from daq_reader import DAQReader

def record_MPU_sample(ts, q, aa, gg):
    hz = daq.compute_hz()
    # placeholder: save to file or process for ML later
    print(f"[RECORD] Packet T: {ts}, HZ: {hz:6.1f}")

daq = DAQReader()
daq.start(callback=record_MPU_sample)

try:
    while daq._running:
        if msvcrt.kbhit() and msvcrt.getwch().lower() == 'q':
            break
        time.sleep(0.01)
except KeyboardInterrupt:
    pass
finally:
    daq.stop()
    winsound.Beep(750, 2000)
    print('DAQ halted')