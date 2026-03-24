###
# This file is a list of helpfule function that will be used in multiple projects.
### 

import os
import numpy as np
import matplotlib.pyplot as plt

def savePicInSequence(figure, folder_path):
    '''
    Saves the figure as the highest numbered photo in the folder passed in. 
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

def text_box_to_fig(fig, hyper_params):
    """
    Adds a hyperparameter text box to a matplotlib figure.
    
    Parameters:
        fig : matplotlib figure
        hyper_params : dict
    """
    ax = fig.gca()

    text = "\n".join(
        f"{key}: {value}" for key, value in hyper_params.items()
    )

    # Calculate height for text box
    num_params = len(hyper_params)
    text_height = 0.03 * num_params  # heuristic factor
    bottom_margin = text_height + 0.1  # base margin + text height

    # Adjust plot layout to make room for text at the bottom
    fig.subplots_adjust(bottom=bottom_margin)

    # Place text below the axes, centered horizontally, text aligned right
    fig.text(
        0.5, 0.01,   # x, y coordinates (0.5 center, 0.01 bottom)
        text,
        ha='center',
        va='bottom',
        multialignment='left',
        fontsize=2,  # fontsize = max(5, 10 - 0.3 * len(hyper_params)) something like this is another idea
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )