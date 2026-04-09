from collections import Counter
import csv
from pathlib import Path

FOLDER = 'recordings/training1'
RECORDINGS_DIR = Path(__file__).resolve().parent / FOLDER
LOW_HZ_THRESHOLD = 99.0


def _parse_meta_data(file_path):
	with file_path.open(newline='') as file:
		reader = csv.reader(file)

		meta = {}
		for _ in range(4):
			row = next(reader, None)
			if not row or len(row) < 2:
				raise ValueError(f'{file_path.name}: missing metadata rows')
			meta[row[0].strip()] = row[1].strip()

	subject = meta['subject']
	arm = meta['arm']
	label = int(meta['label'])
	hz = float(meta['hz'])
	return subject, arm, label, hz


def _scan_recordings():
	records = []
	low_hz_files = []

	if not RECORDINGS_DIR.exists():
		print(f'Recordings folder not found: {RECORDINGS_DIR}')
		return records, low_hz_files

	for file_path in sorted(RECORDINGS_DIR.glob('*.csv')):
		try:
			subject, arm, label, hz = _parse_meta_data(file_path)
		except Exception as error:
			print(f'Skipping {file_path.name}: {error}')
			continue

		records.append({
			'file': file_path,
			'subject': subject,
			'arm': arm,
			'label': label,
			'hz': hz,
		})

		if hz < LOW_HZ_THRESHOLD:
			low_hz_files.append(file_path)

	return records, low_hz_files


def _delete_low_hz_files(low_hz_files):
	if not low_hz_files:
		return

	print('\nFiles below the HZ threshold:')
	for file_path in low_hz_files:
		print(f'  {file_path.name}')

	answer = input('\nDelete these files? [y/N]: ').strip().lower()
	if answer != 'y':
		print('No files deleted.')
		return

	for file_path in low_hz_files:
		file_path.unlink(missing_ok=True)
		print(f'Deleted {file_path.name}')


def _report_duplicate_counts(records):
	counts = Counter(
		(record['subject'], record['arm'], record['label'])
		for record in records
	)
	subject_width = max((len(record['subject']) for record in records), default=len('subject'))
	arm_width = max((len(record['arm']) for record in records), default=len('arm'))
	label_width = max((len(f"reps={label}") for _, _, label in counts), default=len('label'))

	print('\nFile counts by subject, arm, and label:')
	print('subject, arm, label, count')
	for subject, arm, label in sorted(counts):
		label_text = f'reps={label}'
		print(f'{subject:<{subject_width}}, {arm:<{arm_width}}, {label_text:>{label_width}}: {counts[(subject, arm, label)]}')


def main():
	records, low_hz_files = _scan_recordings()

	print(f'Found {len(records)} recording file(s) in {FOLDER}.')
	print(f'{len(low_hz_files)} file(s) are below {LOW_HZ_THRESHOLD:g} Hz.')

	_delete_low_hz_files(low_hz_files)

	if low_hz_files:
		remaining_records, _ = _scan_recordings()
	else:
		remaining_records = records

	_report_duplicate_counts(remaining_records)


if __name__ == '__main__':
	main()
