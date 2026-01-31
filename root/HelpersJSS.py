###
# This file is a list of helpfule function that will be used in multiple projects.
### 

import os
import numpy as np

def savePicInSequence(figure, folder_path):
    '''
    Saves the current figure as the highest numbered photo in the folder passed in. 
    That folder will be created if it doesn't already exist.
    NOTE!: This folder should have no other files inside
    '''

    # check directory exists
    os.makedirs(folder_path, exist_ok=True)
    existing_files = os.listdir(folder_path)

    # get next number
    if not existing_files:
        next_num = 1
    else:
        numbers = [int(file_name[:-4]) for file_name in existing_files]
        next_num = np.max(numbers) + 1

    # save the figure
    filename = f"{next_num:03d}.png"  # zero-padded
    save_path = os.path.join(folder_path, filename)
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure as {filename}")