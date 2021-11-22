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

import time

import constants_analysis as constants
import numpy as np
import pandas as pd
from cimulate_batch import simulate_batch
from joblib import Parallel, delayed

if __name__ == '__main__':
    start = time.time()
    print(f'Simulating with {constants.NUM_SIMS} jobs')
    
    IDs = np.arange(1, constants.NUM_SIMS + 1)
    linear_saccade_model = pd.read_excel('../results/lm_results.xlsx', engine='openpyxl')
    linear_mouse_model = pd.read_excel('../results/lm_results_mouse.xlsx', engine='openpyxl')    

    lm_saccades = [linear_saccade_model] * len(IDs)
    lm_mouse = [linear_mouse_model] * len(IDs)

    results = Parallel(n_jobs=8, backend='loky', verbose=True)(delayed(simulate_batch)\
                                                            (ID, lm_s, lm_m) for \
                                                                ID, lm_s, lm_m in \
                                                                    zip(IDs, lm_saccades, lm_mouse))

    # results = []
    # for ID in IDs:
    #     results.append(simulate_batch(ID, linear_saccade_model, linear_mouse_model))
    
    print(f'{round((time.time() - start) / 60, 1)} minutes')
    

