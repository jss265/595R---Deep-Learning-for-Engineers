import msvcrt
import time

def begin_test():

    subjects = ['Joshua Spiesman',
            'Ian Lake',
            'Tanner Gale',
            'Dean Black',
            'Brennan Johnson',
            'Reese Hammons'
            ]

    arms = ['left',
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

    string = '\nHow many reps fully completed in this window?\n' + ', '.join(reps) + '\nChoice: '
    print(string)
    while(True):
        if msvcrt.kbhit():
            key = msvcrt.getwch()
            if key in reps:
                return int(key)
            elif key == 'r':
                return False
        time.sleep(0.1)