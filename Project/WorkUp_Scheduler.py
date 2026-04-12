import WorkUp_Train

def scheduler():
    AA = 'all'
    A = [['Joshua Spiesman', 
            'Ian Lake',
            'Tanner Gale',
            'Dean Black',
            'Brennan Johnson',
            'Reese Hammons',
            'Peter Cope',
            'Still with noise',
            'Two-hands and jerked',],['right', 'left', 'neither'],[0,1,2,3]]
    BB = 'all but still/jerked cases'
    B = [['Joshua Spiesman',
            'Ian Lake',
            'Tanner Gale',
            'Dean Black',
            'Brennan Johnson',
            'Reese Hammons',
            'Peter Cope'],['right', 'left'],[0,1,2,3]]
    CC = 'all 0,1 reps'
    C = [['Joshua Spiesman',
            'Ian Lake',
            'Tanner Gale',
            'Dean Black',
            'Brennan Johnson',
            'Reese Hammons',
            'Peter Cope',
            'Still with noise',
            'Two-hands and jerked',],['right', 'left', 'neither'],[0,1]]
    DD = 'all 0,1 reps except still/jerked cases'
    D = [['Joshua Spiesman',
            'Ian Lake',
            'Tanner Gale',
            'Dean Black',
            'Brennan Johnson',
            'Reese Hammons',
            'Peter Cope'],['right', 'left'],[0,1]]
    EE = 'only mine'
    E = [['Joshua Spiesman',
            'Still with noise',
            'Two-hands and jerked',],['right', 'left', 'neither'],[0,1,2,3]]
    FF = 'only mine except still/jerked cases'
    F = [['Joshua Spiesman'],['right', 'left'],[0,1,2,3]]

    GG = 'only my 0,1 reps'
    G = [['Joshua Spiesman'],['right', 'left'],[0,1]]

    lists = [[A,AA],
             [B,BB],
             [C,CC],
             [D,DD],
             [E,EE],
             [F,FF],
             [G,GG]]
    
    WorkUp_Train.train_and_eval(A,AA)
    return 
    for list, string in lists:
        string = 'TRAINING: ' + string
        print("\n" + "="*50 + "\n" + string + "\n" + "="*50)
        WorkUp_Train.train_and_eval(list)

scheduler()