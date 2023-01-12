"""
visual_search
Copyright (C) 2022 Utrecht University, Alex Hoogerbrugge

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Use SR Research's EDFConverter or EDF2ASC, with the following settings (v = on, x = off):
v UTF Encoding
- Samples/Events -> Output samples only
- Binocular recording -> Output must be monocular (setting output as binocular for a monocular recording also works)

x Output resolution data
v Output velocity data
v Output float time
x Output input values

- Eye position type -> Gaze
x Load EB log messages

v Block start event output
v Block message event output
x Block eye event output
v Block flags output
x Block target data
v Block viewer commands

v Use tabs only as delimiters
"""

from pathlib import Path

import numpy as np
import pandas as pd
from constants_analysis import N_JOBS, RESULT_DIR


def dots_to_nan(x):
    if str(x) == '   .':
        return np.nan
    else:
        return x


def get_duration(ts):
    ts = np.array(ts)
    start, end = ts[0], ts[-1]
    print(f'Eyetracking data of {round((end - start) / 1000 / 60, 1)} minutes')


def asc_to_df(path: Path, save_to_csv=False) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', header=None)
    df.columns = ['timestamp', 'x', 'y', 'pupilsize', 'xvel', 'yvel']

    for col in list(df.columns):
        df[col] = df[col].apply(dots_to_nan)

    if save_to_csv:
        new_path = str(path).replace('.asc', '-samples.csv')
        df.to_csv(Path(new_path))

    return df


if __name__ == '__main__':
    files = sorted(list(RESULT_DIR.rglob('*.asc')))

    print(f'Converting {len(files)} files from .asc to .csv')

    if N_JOBS == 1:
        for i, f in enumerate(files):
            _ = asc_to_df(f, save_to_csv=True)
            print(f'Processed {i + 1} of {len(files)} files', end='\r')
    else:
        from joblib import Parallel, delayed
        save_to = [True] * len(files)
        _ = Parallel(n_jobs=N_JOBS, backend='loky', verbose=True)(delayed(asc_to_df)(f, st) for f, st in zip(files, save_to))

