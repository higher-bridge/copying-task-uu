#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:55:34 2020
Author: Alex Hoogerbrugge (@higher-bridge)
"""

import numpy as np
import pandas as pd
import random

import simulation_helper as sh
import constants

# for encoding_scheme in encoding_schemes:

#     for condition in conditions:
#         while not finished:
#             k_items = encoding_scheme[condition]

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


encoding_schemes = sh.create_all_encoding_schemes()

features = ['ID',
            'Encoding scheme',
            'Condition',
            'Trial',
            'Crossings',
            'Completion time',
            'Timeout',
            'Fixations',
            'Fixations per second']

tracking_df = pd.DataFrame(columns=features)

for encoding_scheme in encoding_schemes:
    
    for i, condition in enumerate(constants.CONDITIONS):
        # Retrieve condition timings
        visible_time = constants.CONDITION_TIMES[i][0]
        occlude_time = constants.CONDITION_TIMES[i][1]
        
        k_items = encoding_scheme[condition] # Retrieve the number of items to encode
        
        for trial in range(constants.NUM_TRIALS):
            # print(f'Trial {trial}, scheme {encoding_scheme}')
            
            # Tracking variables
            time = 0
            remaining_items = 4 
            num_crossings = 0
            num_fixations = 0
            
            # Generate lists of stimulus locations
            example_locs, workspace_locs, resource_locs = sh.generate_locations(n_items=remaining_items)
            placed_locs = []
            
            # Init eye location in center of screen
            eye_location = (1280, 720)

            
            while time < constants.TIMEOUT and remaining_items > 0:
                
                # If there are fewer items than in the encoding scheme
                if k_items > remaining_items:
                    k_items = remaining_items
                
                
                if sh.example_grid_visible(time, visible_time, occlude_time):
                    # Shift eyes to center of example grid
                    new_location = (640, 720)
                    time += sh.fitts_time(eye_location, new_location)
                    eye_location = new_location
 
                    num_fixations += 1
                    num_crossings += 1
                    
                    locations_memorized = []
                    for k in range(k_items):
                        
                        # Pick a new item to memorize
                        new_item = random.choice([l for l in range(len(example_locs)) \
                                                  if l not in placed_locs and l not in locations_memorized])
                            
                        new_location = example_locs[new_item]
                        
                        # Move eyes to new item
                        time += sh.fitts_time(eye_location, new_location)
                        eye_location = new_location
                        num_fixations += 1
                        
                        # Store item in memory
                        time += 15
                        locations_memorized.append(new_item)
                
                
                if len(locations_memorized) > 0:
                    # Shift eyes to resource grid
                    new_location = (1920, 720)
                    time += sh.fitts_time(eye_location, new_location)
                    eye_location = new_location
                    num_fixations += 1
                    
                    # try_locations = locations_memorized
                    for l in range(len(locations_memorized)):
                        k = locations_memorized[l]
                        
                        # Move eyes to new item
                        new_location = resource_locs[k]
                        time += sh.fitts_time(eye_location, new_location)
                        eye_location = new_location
                        num_fixations += 1
                        
                        # Match item to memory
                        time += 20
                        succesful = True if random.random() > .2 else False
                        
                        if succesful:
                            # Move mouse to resource grid
                            time += 50
                            
                            # Click
                            time += 10
                            
                            # Move eyes to workspace
                            new_location = workspace_locs[k]
                            time += sh.fitts_time(eye_location, new_location)
                            eye_location = new_location
                            num_fixations += 1
                            
                            # Drag item to workspace
                            time += 300
                            
                            # Release click
                            time += 10
                            
                            # Succesfully placed
                            remaining_items -= 1
                            placed_locs.append(k)
                            
                            
                else:
                    # If no items in memory and example grid not visible, just wait.
                    # In fact participants will likely be checking their work, but
                    # not necessary to model here
                    time += 1   

                
            # print(f'Trial time {time}ms')
            d = pd.DataFrame({'ID': '001',
                              'Encoding scheme': str(encoding_scheme),
                              'Condition': condition,
                              'Trial': trial,
                              'Crossings': num_crossings,
                              'Completion time': time,
                              'Timeout': time > constants.TIMEOUT,
                              'Fixations': num_fixations,
                              'Fixations per second': num_fixations / (time / 1000)},
                             index=[0])
            tracking_df = tracking_df.append(d, ignore_index=True)
        
        


