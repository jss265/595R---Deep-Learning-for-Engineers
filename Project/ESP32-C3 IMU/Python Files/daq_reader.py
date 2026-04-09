import threading
import serial
import struct
import time
import winsound
from collections import deque

H1 = 0xAA
H2 = 0x55

# packet sizes
timestamp_size = 8  # uint64_t
quat_size = 16      # 4 floats
accel_size = 6      # 3 int16
gyro_size = 6       # 3 int16
packet_size = timestamp_size + quat_size + accel_size + gyro_size


class DAQReader:
    """Threaded DAQ reader with automatic HZ calculation and header alignment."""
    def __init__(self, port='COM24', baud=115200, maxlen=100):
        self.ser = serial.Serial(port, baud)
        self.sample_times = deque(maxlen=maxlen)
        self._data_callback = None
        self._running = False

    def start(self, callback=None):
        """Start the reader thread. Optional callback receives (timestamp, q, aa, gg)."""
        self._data_callback = callback
        self._running = True
        t = threading.Thread(target=self._reader_thread, daemon=True)
        t.start()

    def stop(self):
        self._running = False

    def _reader_thread(self):
        while self._running:
            try:
                ts, q, aa, gg = self.read_packet()
            except (serial.SerialException, OSError, IndexError, struct.error):
                self._running = False
                winsound.Beep(1000, 2000)
                print("Serial disconnected")
                break
            self.sample_times.append(time.perf_counter())
            if self._data_callback:
                self._data_callback(ts, q, aa, gg)

    def read_packet(self):
        """Aligns on header bytes and reads a single packet."""
        while True:
            if self.ser.read(1)[0] == H1:
                if self.ser.read(1)[0] == H2:
                    break
        data = self.ser.read(packet_size)
        timestamp_us = struct.unpack('<Q', data[0:8])[0]
        q = struct.unpack('<ffff', data[8:24])
        aa = struct.unpack('<hhh', data[24:30])
        gg = struct.unpack('<hhh', data[30:36])
        return timestamp_us, q, aa, gg

    def compute_hz(self):
        if len(self.sample_times) > 1:
            elapsed = self.sample_times[-1] - self.sample_times[0]
            if elapsed > 0:
                return (len(self.sample_times) - 1) / elapsed
        return 0.0