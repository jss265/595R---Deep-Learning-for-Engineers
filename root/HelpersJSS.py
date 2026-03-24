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
    Adds a hyperparameter text box to a matplotlib figure,
    reserving ~1/3 of the figure for the text at the bottom,
    and ensures it doesn't overlap the y-axis labels.

    Parameters:
        fig : matplotlib.figure.Figure
        hyper_params : dict
    """
    # Compute text strings
    text_lines = []
    for key, value in hyper_params.items():
        if isinstance(value, type):  # class
            text_lines.append(f"{key}: {value.__name__}")
        else:
            text_lines.append(f"{key}: {value}")
    text = "\n".join(text_lines)

    # Reserve bottom 1/3 for text
    bottom_reserved = 0.35  # ~1/3 for text box
    fig.subplots_adjust(bottom=bottom_reserved + 0.05)  # extra margin for y-axis

    # Center the text inside the reserved bottom space
    y_center = bottom_reserved / 2

    # Place the text
    fig.text(
        0.5, y_center,
        text,
        ha='center',
        va='center',
        multialignment='left',
        fontsize=max(5, 10 - 0.3 * len(hyper_params)),
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )