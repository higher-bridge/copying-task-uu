"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

from helperfunctions import euclidean_distance

# TODO: description of EDFConverter settings etc.

def convert_fix(row: List[str]) -> pd.DataFrame:
    # Extract al necessary information and return a dataframe row with columns=columns
    row_dict = {'type' : 'fixation',
                'start': float(row[2]),
                'end': float(row[3]),
                'dur': float(row[4]),
                'gstx': np.nan,
                'gsty': np.nan,
                'genx': np.nan,
                'geny': np.nan,
                'gavx': float(row[5]),
                'gavy': float(row[6]),
                'avel': np.nan,
                'pvel': np.nan,
                'pups': float(row[7])}

    row_df = pd.DataFrame(row_dict, index=[0])

    return row_df


def convert_sacc(row: List[str]) -> pd.DataFrame:
    # Extract al necessary information and return a dataframe row with columns=columns

    # Avg velocity = distance (px) / time (ms)
    dur, gstx, gsty, genx, geny = float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8])
    dist = euclidean_distance((gstx, gsty), (genx, geny))
    avel = dist / dur

    row_dict = {'type' : 'saccade',
                'start': float(row[2]),
                'end':float(row[3]),
                'dur': dur,
                'gstx': gstx,
                'gsty': gsty,
                'genx': genx,
                'geny': geny,
                'gavx': np.nan,
                'gavy': np.nan,
                'avel': avel,  # Don't know
                'pvel': float(row[10]),
                'pups': np.nan}

    row_df = pd.DataFrame(row_dict, index=[0])

    return row_df


# These are the columns that originate in the .asc file, dependent on whether it's a SACC or FIX
# columns_sacc = ['event_type',
#                 'eye',
#                 'start_time',
#                 'end_time',
#                 'duration',
#                 'start_x',
#                 'start_y',
#                 'end_x',
#                 'end_y',
#                 'amplitude',
#                 'peak_vel'
#                 ]
# columns_fix = ['event_type',
#                'eye',
#                'start_time',
#                'end_time',
#                'duration',
#                'avg_x',
#                'avg_y',
#                'avg_pupilsize'
#                ]

# Set path to results folder
path = Path(__file__).parent.parent
path = path / 'results'

# Search all .asc files
files = list(path.glob('*.asc'))
print(f'Extracting data from {len(files)} files and converting from .asc to .csv')

for file in files:
    asc = open(file, 'r')
    f = asc.read()

    # f is now one long string, so split into list of separate rows
    rows = f.split('\n')

    # Find in which rows a measurement was started. Namely, we only need the last 'block' of measurement
    start_rows = []
    for i, row in enumerate(rows):
        if row.startswith('EVENTS\tGAZE'):
            start_rows.append(i)

    # Take the last index of starting rows
    starting_row = start_rows[-1] + 1

    # Remove everything before the new starting point
    rows = rows[starting_row:-1]

    # Filter for fixations and saccades
    new_rows = []
    for row in rows:
        if row.startswith('EFIX') or row.startswith('ESACC'):
            split_row = row.split('\t')
            new_rows.append(split_row)

    df = pd.DataFrame(columns=['type', 'start', 'end', 'dur', 'gstx', 'gsty', 'genx', 'geny', 'gavx', 'gavy', 'avel', 'pvel', 'pups'])
    for row in new_rows:
        if row[0] == 'EFIX':
            converted_row = convert_fix(row)
        elif row[0] == 'ESACC':
            converted_row = convert_sacc(row)
        else:
            raise ValueError("I'm just doing this for clarity")

        df = df.append(converted_row, ignore_index=True)

    # Write to csv
    new_path = str(file).replace('.asc', '.csv')
    df.to_csv(Path(new_path))

print(f'Converted {len(files)} files!')
