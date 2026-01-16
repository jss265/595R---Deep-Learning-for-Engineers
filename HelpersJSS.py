###
# This file is a list of helpfule function that will be used in multiple projects.
# Haven't figured out yet how to import it into python files in other folders.
# For now just copy and paste.
### 

import os
import matplotlib.pyplot as plt
import numpy as np

def savePicInSequence(folder_path):
    ###
    # Saves the current figure as the highest numbered photo in the folder passed in. 
    # That folder will be created if it doesn't already exist.
    # NOTE!: This folder should have no other files inside
    ###

    os.makedirs(folder_path, exist_ok=True)
    existing_files = os.listdir(folder_path)
    if not existing_files:
        next_num = 1
    else:
        numbers = [int(file_name[:-4]) for file_name in existing_files]
        next_num = np.max(numbers) + 1

    plt.savefig(f'{folder_path}\{next_num}', dpi=300, bbox_inches="tight")
    print(f'Saved figure as {next_num}.png')