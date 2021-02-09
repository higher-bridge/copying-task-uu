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
import random

import constants
import cimulation_helper as sh

def simulate_trial(dict tracking_dict, int k_items, 
                   int visible_time, int occlude_time,
                   float intercept, float coefficient,
                   float intercept_mouse, float coefficient_mouse,
                   int n_repetitions, 
                   float f, float thresh, float noise, float decay,
                   float error_probability):

    # Tracking variables
    cdef float cumul_time
    cdef int num_crossings, num_fixations
    
    timeout = constants.TIMEOUT
    stimuli_per_trial = constants.STIMULI_PER_TRIAL
    
    cumul_time = 0
    remaining_items = constants.STIMULI_PER_TRIAL 
    num_crossings = 0
    num_fixations = 0
    
    # Generate lists of stimulus locations
    example_locs, workspace_locs, resource_locs = sh.generate_locations(n_items=remaining_items)
    placed_locs = []
    
    cdef activated_at = {i: [] for i in range(len(example_locs))}
    
    # Init eye location in center of screen
    eye_location = (1280, 720)
    mouse_location = (1280, 720)


    while cumul_time < timeout and len(placed_locs) < stimuli_per_trial:
        
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
    

    return num_crossings, cumul_time, num_fixations

