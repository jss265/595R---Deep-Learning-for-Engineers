import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1])) # add parent folder to Python path so sibling modules can be imported
from root import HelpersJSS as jss

# ----------------------------------------

import pandas as pd

df = pd.read_csv('HW 1 - MLP/auto+mpg/auto-mpg.data', sep=r'\s+', header=None, quotechar='"') # sep in place of delim_whitespace=True
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']
assert len(column_names) == df.shape[1], f"Expected {df.shape[1]} column names, but got {len(column_names)}"
df.columns = column_names

