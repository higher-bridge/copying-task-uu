#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:55:34 2020
Author: Alex Hoogerbrugge (@higher-bridge)
"""

import numpy as np
import pandas as pd
import random
import time

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from joblib import Parallel, delayed

import simulation_helper as sh
import constants

# for encoding_scheme in encoding_schemes:

#     for condition in conditions:
#         k_items = encoding_scheme[condition]
#         while not finished:

#             if k_items > remaining_items:
#                 k_items = remaining items

#             Unit task: Encode items
#                 if example grid is visible:
#                     Shift eyes to example grid               (Fitts' law)
#                     Sub-unit task: Encode items
#                         for k in k_items:
#                             Move eyes to new item            (Fitts' law)
#                             Store item in memory             (Time: ACT-R)
#                             ? Rehearse item in memory        (Time: ACT-R)
#                 else wait until grid is visible
            
#             Unit task: Select and place encoded items
#                 Sub-unit task: select and place item
#                 for k in k_items:
#                     Shift eyes to resource grid              (Fitts' law)
#                     Retrieve item                            (Time: ACT-R. 
#                                                              (Probability of succesful retrieval)
#                     if succesful:
#                         Move mouse to resource grid          (Fitts' law)
#                         Pick up item                         (Time: taken from data or research)
#                         Shift eyes to workspace grid         (Fitts' law)
#                         Drag item to workspace grid          (Fitts' law)
#                         Drop item                            (Time: taken from data or research)

#         Update whether finished (if all items placed)



def simulate_participant(ID, encoding_schemes, linear_saccade_model, linear_mouse_model, features):    
    ID = str(ID).zfill(3)
    
    cols = ['ID', 'Encoding scheme', 'Repetitions', 'Condition', 'Trial']
    [cols.append(f) for f in features]
    # tracking_df = pd.DataFrame(columns=cols)
    tracking_dict = {key: [] for key in cols}

    # encoding_schemes = sh.create_all_encoding_schemes()
    memory_repetitions = sh.create_all_encoding_schemes(max_k=constants.MAX_MEMORY_REPETITIONS)    
    
    for encoding_scheme in encoding_schemes:
        for repetition_range in memory_repetitions:
        
            for i, condition in enumerate(constants.CONDITIONS):
                # Get coefficients for saccade speed in this condition. Convert from ms to s by /1000
                saccade_model = linear_saccade_model.loc[linear_saccade_model['Condition'] == condition]
                intercept = list(saccade_model['Intercept'])[0]
                coefficient = list(saccade_model['Coefficient'])[0]
                
                mouse_model = linear_mouse_model.loc[linear_saccade_model['Condition'] == condition]
                intercept_mouse = list(mouse_model['Intercept'])[0]
                coefficient_mouse = list(mouse_model['Coefficient'])[0]
                
                # Retrieve condition timings
                visible_time = constants.CONDITION_TIMES[i][0]
                occlude_time = constants.CONDITION_TIMES[i][1]
                            
                # Retrieve the number of items to encode
                k_items = encoding_scheme[condition] 
                
                # Retrieve the amount of repetitions to perform when encoding
                n_repetitions = repetition_range[condition]
                
                for trial in range(1, constants.NUM_TRIALS + 1):            
                    # Tracking variables
                    cumul_time = 0
                    remaining_items = 4 
                    num_crossings = 0
                    num_fixations = 0
                    
                    # Generate lists of stimulus locations
                    example_locs, workspace_locs, resource_locs = sh.generate_locations(n_items=remaining_items)
                    placed_locs = []
                    
                    # Init eye location in center of screen
                    eye_location = (1280, 720)
                    mouse_location = (1280, 720)
        
        
                    while cumul_time < constants.TIMEOUT and remaining_items > 0:
                        
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
                            for k in range(k_items):
                                
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
                                        cumul_time += 50 * random.gauss(1, .01)
                                        
                                        for rep in range(n_repetitions):
                                            cumul_time += 50 * random.gauss(1, .01)

                                        succesful = True if random.uniform(0, 1) > .1 else False                                                
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
                                
                                # Run through each item in the resource grid and try
                                # to match it to memory
                                for l, r_loc in enumerate(resource_locs): 
                                    # Move eyes to new item in resource grid
                                    new_location = r_loc
                                    cumul_time += sh.estim_saccade_time(eye_location, new_location, a=intercept, b=coefficient)                                
                                    eye_location = new_location
                                    num_fixations += 1
                                
                                    # Try to match item to memory
                                    cumul_time += 150 * random.gauss(1, .1)
                                    
                                    # If the two items match, there is still a certain chance 
                                    # of successful retrieval which needs to be overcome
                                    if l == k:
                                        succesful = True if random.uniform(0, 1) > .2 else False
                                
                                if succesful:
                                    # Move mouse to resource grid
                                    new_mouse_location = resource_locs[k]
                                    cumul_time += sh.estim_mouse_time(mouse_location, new_mouse_location,
                                                                      a=intercept_mouse, b=coefficient_mouse)
                                    mouse_location = new_mouse_location
                                    
                                    # Click on item
                                    cumul_time += 150 * random.gauss(1, .1)
                                    
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
                                    
                                    
                        else:
                            # If no items in memory and example grid not visible, just wait.
                            # In fact participants will likely be checking their work, but
                            # not necessary to model here
                            cumul_time += 1   
        
                    # Code would check for finished trial every 500ms to conserve time, 
                    # so we add a random value between 1-500ms
                    cumul_time += random.uniform(1, 500)
        
                    tracking_dict['ID'].append(ID)
                    tracking_dict['Encoding scheme'].append(str(encoding_scheme))
                    tracking_dict['Repetitions'].append(str(repetition_range))
                    tracking_dict['Condition'].append(int(condition))
                    tracking_dict['Trial'].append(int(trial))
                    tracking_dict['Number of crossings'].append(float(num_crossings))
                    tracking_dict['Completion time (s)'].append(float(cumul_time / 1000))
                    tracking_dict['Fixations per second'].append(float(num_fixations / (cumul_time / 1000)))
                    
                
    tracking_df = pd.DataFrame(tracking_dict)
                
    return tracking_df


if __name__ == '__main__':
    start = time.time()
    
    features = ['Number of crossings',
                'Completion time (s)',
                # 'Timeout',
                # 'Fixations',
                'Fixations per second']
    
    IDs = np.arange(1, constants.NUM_PPS_SIM + 1)
    linear_saccade_model = pd.read_excel('../results/lm_results.xlsx')
    linear_mouse_model = pd.read_excel('../results/lm_results_mouse.xlsx')    
    encoding_schemes = sh.create_all_encoding_schemes()

    schemes = [encoding_schemes] * len(IDs)
    lm_saccades = [linear_saccade_model] * len(IDs)
    lm_mouse = [linear_mouse_model] * len(IDs)
    feature_list = [features] * len(IDs)

    dfs = Parallel(n_jobs=-3, backend='loky', verbose=True)(delayed(simulate_participant)\
                                                            (ID, scheme, lm_s, lm_m, f) for \
                                                                ID, scheme, lm_s, lm_m, f in \
                                                                    zip(IDs, schemes, lm_saccades, lm_mouse, feature_list))

    # dfs = []
    # for ID in IDs:
    #     dfs.append(simulate_participant(ID, encoding_schemes, linear_saccade_model, linear_mouse_model, features))

    results = pd.concat(dfs, ignore_index=True)
    results.to_csv('../results/simulation_results.csv')
    
    print(f'{round(time.time() - start, 1)} seconds')
    

