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

import pickle
import time

import constants_analysis as constants
import numpy as np
import pandas as pd
import simulation_helper as sh
from cimulate_trial import simulate_trial
from joblib import Parallel, delayed


cpdef simulate_batch(ID, linear_saccade_model, linear_mouse_model):    
    ID = str(ID).zfill(3)
    
    features = ['Number of crossings',
                'Completion time (s)',
                'Fixations per second']
    cols = ['ID', 'Encoding scheme', 'Repetitions', 'Parameters', 'Condition', 'Trial', 'Processing Time']
    [cols.append(i) for i in features]
    
    cdef tracking_dict = {key: [] for key in cols}
    
    # Load (linear) saccade models
    cdef saccade_models = [linear_saccade_model.loc[linear_saccade_model['Condition'] == condition] \
                           for condition in constants.CONDITIONS]
    cdef intercepts = [list(i['Intercept'])[0] for i in saccade_models]
    cdef coefs =   [list(i['Coefficient'])[0] for i in saccade_models]  
                            
    cdef mouse_models = [linear_mouse_model.loc[linear_saccade_model['Condition'] == condition] \
                         for condition in constants.CONDITIONS]
    cdef intercepts_m = [list(i['Intercept'])[0] for i in mouse_models]
    cdef coefs_m =   [list(i['Coefficient'])[0] for i in mouse_models]     

    # Load parameter set    
    cdef encoding_schemes = sh.create_all_encoding_schemes()
    cdef memory_repetitions = sh.create_all_encoding_schemes(max_k=constants.MAX_MEMORY_REPETITIONS)   
    cdef range_params = sh.get_param_combinations()
  
    # Declare loop vars
    cdef float f, decay, thresh, noise
    cdef float intercept, coefficient, intercept_mouse, coefficient_mouse
    cdef float error_probability
    cdef int visible_time, occlude_time, k_items, n_repetitions
    cdef int num_crossings, num_fixations
    cdef float cumul_time, cond_start
      
    for encoding_scheme in encoding_schemes:
        cond_start = time.time()
        for repetition_range in memory_repetitions:
            for params in range_params:
                f, decay, thresh, noise = params[0], params[1], params[2], params[3]
        
                
                # Start the simulation of a condition        
                for condition_number, condition in enumerate(constants.CONDITIONS):
                    
                    
                    # Get coefficients for saccade speed in this condition. Convert from ms to s by /1000
                    intercept = intercepts[condition_number]
                    coefficient = coefs[condition_number]
                    
                    intercept_mouse = intercepts_m[condition_number]
                    coefficient_mouse = coefs_m[condition_number]
                    
                    # Set probability of incorrect placement
                    error_probability = constants.ERROR_RATES[condition_number] / constants.STIMULI_PER_TRIAL
                    
                    # Retrieve condition timings
                    visible_time = constants.CONDITION_TIMES[condition_number][0]
                    occlude_time = constants.CONDITION_TIMES[condition_number][1]
                    
                    # Add variation to timings
                    # if occlude_time != 0:
                    #     visible_time = visible_time * random.gauss(1.0, .1)
                    #     occlude_time = constants.SUM_DURATION - visible_time
                                
                    # Retrieve the number of items to encode
                    k_items = encoding_scheme[condition_number] 
                    
                    # Retrieve the amount of repetitions to perform when encoding
                    n_repetitions = repetition_range[condition_number]
                    
                    for trial in range(1, constants.NUM_TRIALS + 1):
                        start = time.time()
                        num_crossings, cumul_time, num_fixations = simulate_trial(tracking_dict, k_items, 
                                                                                  visible_time, occlude_time, 
                                                                                  intercept, coefficient, 
                                                                                  intercept_mouse, coefficient_mouse, 
                                                                                  n_repetitions, 
                                                                                  f, thresh, noise, decay, 
                                                                                  error_probability)
                        
                        tracking_dict['ID'].append(int(ID))
                        tracking_dict['Encoding scheme'].append(str(encoding_scheme))
                        tracking_dict['Repetitions'].append(str(repetition_range))
                        tracking_dict['Parameters'].append((str(params)))
                        tracking_dict['Condition'].append(int(condition))
                        tracking_dict['Trial'].append(int(trial))
                        tracking_dict['Number of crossings'].append(int(num_crossings))
                        tracking_dict['Completion time (s)'].append(cumul_time / 1000)
                        tracking_dict['Fixations per second'].append(num_fixations / (cumul_time / 1000))
                        tracking_dict['Processing Time'].append(float(time.time() - start))
                                                
        # print(f'Scheme loop took {round((time.time() - cond_start) * 1000, 1)}ms')

    try:    
        pickle.dump(tracking_dict, open(f'../results/simulations/simulation_{ID}_results.p', 'wb'))    
    except Exception as e:
        print(f'Could not save pickle {ID}:', e)     

    try:    
        df = pd.DataFrame(tracking_dict)
        df.to_csv(f'../results/simulations/simulation_{ID}_results.csv')
    except Exception as e:
        print(f'Could not save dataframe {ID}:', e)          
    
    
    return True