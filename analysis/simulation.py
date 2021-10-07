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
import random
import time

import constants_analysis as constants
import numpy as np
import pandas as pd
import simulation_helper as sh
from joblib import Parallel, delayed


def simulate_participant(ID, linear_saccade_model, linear_mouse_model):    
    ID = str(ID).zfill(3)
    
    
    features = ['Number of crossings',
                'Completion time (s)',
                'Fixations per second']
    cols = ['ID', 'Encoding scheme', 'Repetitions', 'Parameters', 'Condition', 'Trial', 'Processing Time'] #, 'Waited']
    [cols.append(f) for f in features]
    tracking_dict = {key: [] for key in cols}

    encoding_schemes = sh.create_all_encoding_schemes()
    memory_repetitions = sh.create_all_encoding_schemes(max_k=constants.MAX_MEMORY_REPETITIONS)   
    range_params = sh.get_param_combinations()
    
    for encoding_scheme in encoding_schemes:
        for repetition_range in memory_repetitions:
            for params in range_params:
                f, decay, thresh, noise = params[0], params[1], params[2], params[3]
        
                # Start the simulation of a condition        
                for condition_number, condition in enumerate(constants.CONDITIONS):
                    # Get coefficients for saccade speed in this condition. Convert from ms to s by /1000
                    saccade_model = linear_saccade_model.loc[linear_saccade_model['Condition'] == condition]
                    intercept = list(saccade_model['Intercept'])[0]
                    coefficient = list(saccade_model['Coefficient'])[0]
                    
                    mouse_model = linear_mouse_model.loc[linear_saccade_model['Condition'] == condition]
                    intercept_mouse = list(mouse_model['Intercept'])[0]
                    coefficient_mouse = list(mouse_model['Coefficient'])[0]
                    
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
                        # Tracking variables
                        start = time.time()
                        cumul_time = 0
                        remaining_items = constants.STIMULI_PER_TRIAL 
                        num_crossings = 0
                        num_fixations = 0
                        
                        # Generate lists of stimulus locations
                        example_locs, workspace_locs, resource_locs = sh.generate_locations(n_items=remaining_items)
                        placed_locs = []
                        
                        activated_at = {i: [] for i in range(len(example_locs))}
                        
                        # Init eye location in center of screen
                        eye_location = (1280, 720)
                        mouse_location = (1280, 720)
            
            
                        while cumul_time < constants.TIMEOUT and len(placed_locs) < constants.STIMULI_PER_TRIAL:
                            
                            # If there are fewer items than in the encoding scheme
                            if k_items > remaining_items:
                                k_items = remaining_items
                            
                            
                            if sh.example_grid_visible(cumul_time, visible_time, occlude_time):
                                # Shift eyes to center of example grid
                                new_location = (640, 720)
                                cumul_time += sh.estim_saccade_time(eye_location, new_location, a=intercept, b=coefficient)
                                eye_location = new_location
             
                                num_fixations += 1
                                num_crossings += 1
                                
                                locations_memorized = []
                                for ki in range(k_items):
                                    
                                    if sh.example_grid_visible(cumul_time, visible_time, occlude_time):
                                    
                                        # Pick a new item to memorize
                                        new_item = random.choice([l for l in range(len(example_locs)) \
                                                                  if l not in placed_locs and l not in locations_memorized])
                                            
                                        new_location = example_locs[new_item]
                                        
                                        # Move eyes to new item
                                        cumul_time += sh.estim_saccade_time(eye_location, new_location, a=intercept, b=coefficient)                            
                                        eye_location = new_location
                                        num_fixations += 1
                                        
                                        if sh.example_grid_visible(cumul_time, visible_time, occlude_time):
                                            # Store item in memory
                                            activated_at[new_item].append(cumul_time / 1000)
                                            cumul_time += 50 * random.gauss(1, .01)
                                            
                                            for rep in range(n_repetitions):
                                                retrieval, succesful = sh.compute_dme_retrieval(cumul_time / 1000, activated_at[new_item],
                                                                                                f=f, thresh=thresh, noise=noise, decay=decay)
                                                
                                                cumul_time += retrieval * 1000                                             
                                            
                                            if succesful:
                                                locations_memorized.append(new_item)
                            
                            
                            if len(locations_memorized) > 0:
                                # Shift eyes to resource grid
                                new_location = (1920, 1160)
                                cumul_time += sh.estim_saccade_time(eye_location, new_location, a=intercept, b=coefficient)                       
                                eye_location = new_location
                                num_fixations += 1
                                
                                # Run through each location in memory
                                for k in locations_memorized:
                                    found_match = False
                                    
                                    # Run through each item in the resource grid and try
                                    # to match it to memory
                                    for l, r_loc in enumerate(resource_locs): 
                                        # Move eyes to new item in resource grid
                                        new_location = r_loc
                                        cumul_time += sh.estim_saccade_time(eye_location, new_location, a=intercept, b=coefficient)                                
                                        eye_location = new_location
                                        num_fixations += 1
                                    
                                        # Try to match item to memory
                                        retrieval, succesful = sh.compute_dme_retrieval(cumul_time / 1000, activated_at[k],
                                                                                          f=f, thresh=thresh, noise=noise, decay=decay)
                                        activated_at[k].append(cumul_time / 1000)
                                        cumul_time += retrieval * 1000
                                        
                                        # If memory matches viewed item
                                        if l == k:
                                            found_match = True
                                    
                                    if succesful and found_match:
                                        # Move mouse to resource grid
                                        new_mouse_location = resource_locs[k]
                                        cumul_time += sh.estim_mouse_time(mouse_location, new_mouse_location,
                                                                          a=intercept_mouse, b=coefficient_mouse)
                                        mouse_location = new_mouse_location
                                        
                                        # Click on item
                                        cumul_time += 150 * random.gauss(1, .1) # Gray & Boehm-Davis, 2000
                                        
                                        # Move eyes to workspace
                                        new_location = workspace_locs[k]
                                        cumul_time += sh.estim_saccade_time(eye_location, new_location, a=intercept, b=coefficient)                               
                                        eye_location = new_location
                                        num_fixations += 1
                                        
                                        # Drag item to workspace
                                        new_mouse_location = workspace_locs[k]
                                        cumul_time += sh.estim_mouse_time(mouse_location, new_mouse_location, 
                                                                          a=intercept_mouse, b=coefficient_mouse)
                                        mouse_location = new_mouse_location
                                        
                                        # Release click
                                        cumul_time += 150 * random.gauss(1, .1)
                                        
                                        
                                        # DETERMINE WHETHER MISTAKE WAS MADE
                                        if random.uniform(0, 1) > error_probability:
                                            # Succesfully placed
                                            remaining_items -= 1
                                            placed_locs.append(k)
                                            locations_memorized.remove(k)
                                        else:
                                            # =====================================================
                                            # FIX MISTAKE                                            
                                            # =====================================================
                                            # Item placed incorrectly, right-click item to remove
                                            cumul_time += 150 
                                            
                                            # Shift eyes to the item on the example grid
                                            new_location = example_locs[k]
                                            cumul_time += sh.estim_saccade_time(eye_location, new_location, a=intercept, b=coefficient)
                                            eye_location = new_location
                                            num_fixations += 1
                                            num_crossings +=1
                                            
                                            # Sample item again
                                            cumul_time += 50 * random.gauss(1., .1)
                                            activated_at[k].append(cumul_time / 1000)

                                            # Move eyes to item in resource grid
                                            new_location = r_loc
                                            cumul_time += sh.estim_saccade_time(eye_location, new_location, a=intercept, b=coefficient)                                
                                            eye_location = new_location
                                            num_fixations += 1
                                            
                                            # Move mouse to resource grid
                                            new_mouse_location = resource_locs[k]
                                            cumul_time += sh.estim_mouse_time(mouse_location, new_mouse_location,
                                                                              a=intercept_mouse, b=coefficient_mouse)
                                            mouse_location = new_mouse_location
                                            
                                            # Click on item
                                            cumul_time += 150 * random.gauss(1, .1) # Gray & Boehm-Davis, 2000
                                            
                                            # Move eyes to workspace
                                            new_location = workspace_locs[k]
                                            cumul_time += sh.estim_saccade_time(eye_location, new_location, a=intercept, b=coefficient)                               
                                            eye_location = new_location
                                            num_fixations += 1
                                            
                                            # Drag item to workspace
                                            new_mouse_location = workspace_locs[k]
                                            cumul_time += sh.estim_mouse_time(mouse_location, new_mouse_location, 
                                                                              a=intercept_mouse, b=coefficient_mouse)
                                            mouse_location = new_mouse_location
                                            
                                            # Release click
                                            cumul_time += 150 * random.gauss(1, .1)                                            

                                            # Succesfully placed
                                            remaining_items -= 1
                                            placed_locs.append(k)
                                            locations_memorized.remove(k)
            
                                        
                            else:
                                # If no items in memory and example grid not visible, just wait.
                                # In fact participants will likely be checking their work, but
                                # not necessary to model here
                                cumul_time += 10   
            
                        # Code would check for finished trial every 500ms to conserve time, 
                        # so we add a random value between 1-500ms
                        cumul_time += int(round(random.uniform(1, 500)))
                        
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

    # try:    
    #     pickle.dump(tracking_dict, open(f'../results/simulations/simulation_{ID}_results.p', 'wb'))    
    # except Exception as e:
    #     print(f'Could not save pickle {ID}:', e)     

    # try:    
    #     df = pd.DataFrame(tracking_dict)
    #     df.to_csv(f'../results/simulations/simulation_{ID}_results.csv')
    # except Exception as e:
    #     print(f'Could not save dataframe {ID}:', e)          
    
    
    return True


if __name__ == '__main__':
    start = time.time()
    
    IDs = np.arange(1, constants.NUM_JOBS_SIM + 1)
    linear_saccade_model = pd.read_excel('../results/lm_results.xlsx', engine='openpyxl')
    linear_mouse_model = pd.read_excel('../results/lm_results_mouse.xlsx', engine='openpyxl')    

    lm_saccades = [linear_saccade_model] * len(IDs)
    lm_mouse = [linear_mouse_model] * len(IDs)

    results = Parallel(n_jobs=8, backend='loky', verbose=True)(delayed(simulate_participant)\
                                                            (ID, lm_s, lm_m) for \
                                                                ID, lm_s, lm_m in \
                                                                    zip(IDs, lm_saccades, lm_mouse))

    # results = []
    # for ID in IDs:
    #     results.append(simulate_participant(ID, linear_saccade_model, linear_mouse_model))
    
    print(f'{round((time.time() - start) / 60, 1)} minutes')
    

