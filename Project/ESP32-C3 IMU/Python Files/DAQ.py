import serial
import struct
import time
from collections import deque

ser = serial.Serial('COM24', 115200)

H1 = 0xAA
H2 = 0x55

# sizes (bytes)
timestamp_size = 8  # uint64_t microseconds
quat_size = 16   # 4 floats
accel_size = 6   # 3 int16
gyro_size = 6    # 3 int16
packet_size = timestamp_size + quat_size + accel_size + gyro_size

sample_times = deque(maxlen=100)

def read_packet():
    # find header
    while True:
        if ser.read(1)[0] == H1:
            if ser.read(1)[0] == H2:
                break

    data = ser.read(packet_size)

    timestamp_us = struct.unpack('<Q', data[0:8])[0]
    q = struct.unpack('<ffff', data[8:24])
    aa = struct.unpack('<hhh', data[24:30])
    gg = struct.unpack('<hhh', data[30:36])

    return timestamp_us, q, aa, gg

while True:
    sample_times.append(time.perf_counter())

    hz_text = "HZ:   --.-"
    if len(sample_times) > 1:
        elapsed = sample_times[-1] - sample_times[0]
        if elapsed > 0:
            hz = (len(sample_times) - 1) / elapsed
            hz_text = f"HZ: {hz:6.1f}"

    timestamp_us, q, aa, gg = read_packet()
    print(
        f"{hz_text} "
        f"T: {timestamp_us:>12d} us "
        f"Q: ({q[0]:10.6f}, {q[1]:10.6f}, {q[2]:10.6f}, {q[3]:10.6f}) "
        f"A: ({aa[0]:6d}, {aa[1]:6d}, {aa[2]:6d}) "
        f"G: ({gg[0]:6d}, {gg[1]:6d}, {gg[2]:6d})"
    )