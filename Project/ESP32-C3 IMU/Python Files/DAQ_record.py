import csv
import queue
import msvcrt
import sys
import threading
import time
import winsound
from pathlib import Path

from daq_reader import DAQReader

FILE_PATH = 'recordings/training1'

def begin_test():

    subjects = [
        'Joshua Spiesman',
        'Ian Lake',
        'Tanner Gale',
        'Dean Black',
        'Brennan Johnson',
        'Reese Hammons'
        ]

    arms = [
        'left',
        'right',
        'neither'
        ]

    string = '\nPick a subject:\n' + ', '.join(subjects) + '\nChoice: '
    subject = input(string)

    string = '\nPick an arm:\n' + ', '.join(arms) + '\nChoice: '
    arm = input(string)

    return subject, arm

def label_num_reps():

    reps = ['0', '1', '2', '3']

    string = '\nHow many reps fully completed in this window?\n' + ', '.join(reps) + ', r (reject)\nPress that key to label this window.'
    print(string)
    while(True):
        if msvcrt.kbhit():
            key = msvcrt.getwch()
            if key in reps:
                return int(key)
            elif key == 'r':
                return False
        time.sleep(0.1)

def safe_name(text):
    return ''.join(char if char.isalnum() else '_' for char in text).strip('_')

def save_window_csv(subject, arm, label, samples):
    out_dir = Path(__file__).resolve().parent / FILE_PATH
    out_dir.mkdir(exist_ok=True)

    stamp = time.strftime('%Y%m%d-%H%M%S')
    file_name = (
        f"{safe_name(subject)}_{safe_name(arm)}_"
        f"r{label}_{stamp}.csv"
    )

    file_path = out_dir / file_name
    with file_path.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['subject', subject])
        writer.writerow(['arm', arm])
        writer.writerow(['label', label])
        writer.writerow(['hz', compute_window_hz(samples)])
        writer.writerow([])
        writer.writerow(['timestamp_us', 'q0', 'q1', 'q2', 'q3', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'])
        writer.writerows(samples)

def compute_window_hz(samples):
    if len(samples) < 2:
        return 0.0

    elapsed_us = samples[-1][0] - samples[0][0]
    if elapsed_us <= 0:
        return 0.0

    elapsed_s = elapsed_us / 1_000_000.0
    return (len(samples) - 1) / elapsed_s

def main():
    subject, arm = begin_test()

    daq = DAQReader()
    recording = threading.Event()
    sample_queue = queue.Queue()
    recording_start_time = None
    last_status_length = 0

    def drain_samples():
        samples = []
        while True:
            try:
                samples.append(sample_queue.get_nowait())
            except queue.Empty:
                break
        return samples

    def record_MPU_sample(ts, q, aa, gg):
        if recording.is_set():
            sample_queue.put([
                ts,
                q[0], q[1], q[2], q[3],
                aa[0], aa[1], aa[2],
                gg[0], gg[1], gg[2],
            ])

    daq.start(callback=record_MPU_sample)
    print('\nSpace starts/stops recording. q quits.')

    try:
        while daq._running:
            window_samples = None

            if msvcrt.kbhit():
                key = msvcrt.getwch().lower()

                if key == 'q':
                    break

                if key == ' ':
                    if not recording.is_set():
                        drain_samples()
                        recording.set()
                        recording_start_time = time.perf_counter()
                        last_status_length = 0
                        print('Recording...')
                    else:
                        recording.clear()
                        sys.stdout.write('\r' + (' ' * last_status_length) + '\r')
                        sys.stdout.flush()
                        seconds_elapsed = 0.0 if recording_start_time is None else time.perf_counter() - recording_start_time
                        window_samples = drain_samples()

                    if window_samples is not None:
                        hz = compute_window_hz(window_samples)
                        print(f'Recording stopped. HZ: {hz:6.1f}, Seconds: {seconds_elapsed:6.1f}.')
                        label = label_num_reps()

                        if label is False:
                            print('Window rejected.')
                        else:
                            save_window_csv(subject, arm, label, window_samples)
                            print(f'Saved window with label {label}.')

                        print('Press space to record the next window or q to quit.')

            if recording.is_set() and recording_start_time is not None:
                elapsed_s = time.perf_counter() - recording_start_time
                status_line = f'Recording elapsed: {elapsed_s:6.1f} s'
                last_status_length = max(last_status_length, len(status_line))
                sys.stdout.write('\r' + status_line.ljust(last_status_length))
                sys.stdout.flush()

            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:

        daq.stop()
        winsound.Beep(750, 2000)
        print('DAQ halted')


if __name__ == '__main__':
    main()