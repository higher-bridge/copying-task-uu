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


for encoding_scheme in encoding_schemes:
    
    for i, condition in enumerate(constants.CONDITIONS):
        # Retrieve condition timings
        visible_time = constants.CONDITION_TIMES[i][0]
        occlude_time = constants.CONDITION_TIMES[i][1]
        
        k_items = encoding_scheme[condition] # Retrieve the number of items to encode
        remaining_items = 4                  # Init the number of items remaining
        
        # Generate lists of stimulus locations
        example_locs, workspace_locs, resource_locs = sh.generate_locations(n_items=remaining_items)
        
        
        for trial in range(constants.NUM_TRIALS):
            # Tracking variables
            time = 0
            items_in_memory = 0
            num_crossings = 0
            
            
            while time < constants.TIMEOUT and remaining_items > 0:
                
                # If there are fewer items than in the encoding scheme
                if k_items > remaining_items:
                    k_items = remaining_items
                
                
                if sh.example_grid_visible(time, visible_time, occlude_time):
                    # Shift eyes to example grid
                    time += 100
                    num_crossings += 1
                    
                    for k in range(k_items):
                        # Move eyes to new item
                        time += 20
                        
                        # Store item in memory
                        time += 15
                        items_in_memory += 1
                
                
                if items_in_memory > 0:
                    # Shift eyes to resource grid
                    time += 100
                    
                    for k in range(k_items):
                        # Move eyes to new item
                        time += 20
                        
                        # Match item to memory
                        time += 20
                        succesful = True if random.random() > .2 else False
                        
                        if succesful:
                            # Move mouse to resource grid
                            time += 50
                            
                            # Click
                            time += 10
                            
                            # Move eyes to workspace
                            time += 100
                            
                            # Drag item to workspace
                            time += 300
                            
                            # Release click
                            time += 10
                            
                            # Succesfully placed
                            remaining_items -= 1
                            items_in_memory -= 1
                            
                else:
                    # If no items in memory and example grid not visible, just wait.
                    # In fact participants will likely be checking their work, but
                    # not necessary to model here
                    time += 1   
                
                # if remaining_items == 0:
                #     print(f'Trial time {time}ms')
                # elif time >= constants.TIMEOUT:
                #     print('Timeout')
                
        
        


