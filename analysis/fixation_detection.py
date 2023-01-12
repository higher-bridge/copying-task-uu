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
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from constants_analysis import N_JOBS, RESULT_DIR

from analysis.hessels_classifier import classify_hessels2020


def detect_fixations(path: Path, ID, to_csv=True, skip_existing=False) -> pd.DataFrame:
    new_path = Path(str(path).replace('samples', 'events'))

    if skip_existing and new_path in list(RESULT_DIR.rglob('*-events.csv')):
        return pd.DataFrame()

    df = pd.read_csv(path)
    fixations = classify_hessels2020(df, ID)

    if to_csv:
        fixations.to_csv(new_path)

    return fixations


if __name__ == '__main__':
    paths = sorted(list(RESULT_DIR.rglob('*-samples.csv')))
    IDs = [str(path.name)[0:4] for path in paths]

    if N_JOBS == 1:
        for i, (p, ID) in enumerate(zip(paths, IDs)):
            _ = detect_fixations(p, ID)
            print(f'Processed {i + 1} of {len(paths)} files')

    else:
        from joblib import Parallel, delayed
        _ = Parallel(n_jobs=N_JOBS, backend='loky', verbose=True)(delayed(detect_fixations)(p, ID) for p, ID in zip(paths, IDs))

