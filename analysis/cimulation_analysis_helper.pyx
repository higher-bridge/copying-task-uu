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

import numpy as np
# cimport numpy as np
import pandas as pd
import time

from sklearn.preprocessing import MinMaxScaler

import constants_analysis as constants
from typing import List, Any


PARSE_RESULTS = False

cdef compute_se(x : np.array, y : np.array):
    squared_error = (np.mean(x) - np.mean(y)) ** 2
    
    mms = MinMaxScaler()
    x = mms.fit_transform(np.array(x).reshape(-1, 1))
    y = mms.transform(np.array(y).reshape(-1, 1))
    
    norm_squared_error = (np.mean(x) - np.mean(y)) ** 2
    
    return (squared_error, norm_squared_error)

def parse_results(sim_data_s : pd.DataFrame,
                  exp_data : pd.DataFrame,
                  scheme : List[Any],
                  features : List[str]):
    start = time.time()
    
    cdef dict results_dict, scaled_squared_errors

    cdef exp_datas =  [exp_data.loc[exp_data['Condition'] == c] for c in constants.CONDITIONS]
    
    results_dict = {key: [] for key in ['Encoding scheme', 'Repetitions', 'Parameters', 
                                        'Mean Scaled RMSE',
                                        'Crossings', 'Time', 'Fixations']}
    
    # Loop through params and repetition strategies
    cdef all_repetitions = np.array(sim_data_s['Repetitions'].unique())
    cdef all_params = np.array(sim_data_s['Parameters'].unique())

    for repetitions in all_repetitions:
        start1 = time.time()
        sim_data_r = sim_data_s.loc[sim_data_s['Repetitions'] == repetitions]
        
        for params in all_params:
            sim_data_p = sim_data_r.loc[sim_data_r['Parameters'] == params]

            scaled_squared_errors = {key: [] for key in features}
            
            for condition_number, condition in enumerate(constants.CONDITIONS):
                exp_data_c = exp_datas[condition_number]
                sim_data_c = sim_data_p.loc[sim_data_p['Condition'] == condition]    
                
                sim_grouped = sim_data_c.groupby(['ID', 'Condition']).agg({f: ['mean'] for f in features}).reset_index()
                sim_grouped.columns = sim_grouped.columns.get_level_values(0)
                
                # For every feature, calculate scaled squared error
                for feat in features:
                    x = np.array(exp_data_c[feat])
                    y = np.array(sim_grouped[feat])
                    
                    se, nse = compute_se(x, y)
                    
                    scaled_squared_errors[feat].append(nse)
                 
    
            # After calculating statistics for each condition, calculate the RMSE for each feature
            all_scaled_rmse = [np.sqrt(np.mean(scaled_squared_errors[feat])) for feat in features]
                        
            results_dict['Encoding scheme'].append(scheme)
            results_dict['Repetitions'].append(repetitions)
            results_dict['Parameters'].append(params)
            results_dict['Mean Scaled RMSE'].append(np.mean(all_scaled_rmse))
            results_dict['Crossings'].append(all_scaled_rmse[0])
            results_dict['Time'].append(all_scaled_rmse[1])
            results_dict['Fixations'].append(all_scaled_rmse[2])
            
        print(f'Inner loop took {round(time.time() - start1, 3)}s')
        
    print(f'Loop took {round(time.time() - start, 3)}s')
                
    return results_dict


