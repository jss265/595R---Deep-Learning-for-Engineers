import csv
import queue
import msvcrt
import threading
import time
import winsound
from pathlib import Path

from daq_reader import DAQReader


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
    out_dir = Path(__file__).resolve().parent / 'recordings'
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
                        print('Recording...')
                    else:
                        recording.clear()
                        window_samples = drain_samples()

                    if window_samples is not None:
                        hz = compute_window_hz(window_samples)
                        print(f'Recording stopped. HZ: {hz:6.1f}')
                        label = label_num_reps()

                        if label is False:
                            print('Window rejected.')
                        else:
                            save_window_csv(subject, arm, label, window_samples)
                            print(f'Saved window with label {label}.')

                        print('Press space to record the next window or q to quit.')

            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        daq.stop()
        winsound.Beep(750, 2000)
        print('DAQ halted')


if __name__ == '__main__':
    main()