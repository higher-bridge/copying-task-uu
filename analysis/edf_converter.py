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

import helperfunctions as hf
import pandas as pd
from joblib import Parallel, delayed
from pyedfread import edf
from constants_analysis import base_location
from pathlib import Path


def convert_edf(file, allfiles):
    try:
        new_filename = file.replace('.edf', '-samples.csv')
        
        if new_filename not in allfiles:
            samples, events, messages = edf.pread(file)
            samples.to_csv(file.replace('.edf', '-samples.csv'))
            events.to_csv(file.replace('.edf', '-events.csv'))
        
        return True
    
    except Exception as e:
        print(file, e)
        return False


if __name__ == '__main__':
    allfiles = [f for f in hf.getListOfFiles(base_location)]
    files = [f for f in allfiles if '.edf' in f]

    allfiles_repeated = [allfiles] * len(files)
    
    results = Parallel(n_jobs=10, backend='loky', verbose=True)(delayed(convert_edf)(f, af) for f, af in zip(files, allfiles_repeated))
