#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:55:34 2020
Author: Alex Hoogerbrugge (@higher-bridge)
"""

import numpy as np
import math
import random

import constants

def euclidean_distance(loc1:tuple, loc2:tuple):
    dist = [(a - b) ** 2 for a, b in zip(loc1, loc2)]
    dist = math.sqrt(sum(dist))
    
    return dist

def fitts_id(loc1:tuple, loc2:tuple):
    # Fitts' law: ID = log2(2D / W). W = target_size[0] for now
    
    distance = euclidean_distance(loc1, loc2)
    dw = (2 * distance) / constants.TARGET_SIZE[0]
    fitts = np.log2(dw)

    return fitts

def fitts_time(loc1:tuple, loc2:tuple, a:int=.05, b:int=.1, sigma:int=.1):
    # Fitts' time = a + b * ID
    # a & b can be empirically calculated. Maybe separately for eyes and mouse.
    duration = a + b * fitts_id(loc1, loc2)
    noise = random.gauss(mu=1.0, sigma=sigma)
    
    return duration * noise * 1000 # milliseconds

def generate_locations(n_items:int=4):
    all_example_locations = [(515, 570), (650, 570), (785, 570),
                             (515, 730), (650, 730), (785, 730),
                             (515, 870), (650, 870), (785, 870)]
    all_workspace_locations = [(1775, 570), (1910, 570), (2045, 570),
                               (1775, 730), (1910, 730), (2045, 730),
                               (1775, 870), (1910, 870), (2045, 870)]
    all_resource_locations = [(1775, 1075), (1910, 1075), (2045, 1075),
                              (1775, 1300), (1910, 1300), (2045, 1300)]
    
    # Sample a set of locations so we can index matching one from example and workspace
    locations = random.sample(list(range(len(all_example_locations))), n_items)
    
    example_locations = [all_example_locations[i] for i in locations]
    workspace_locations = [all_workspace_locations[i] for i in locations]
    
    resource_locations = all_resource_locations[0:n_items]
    
    return example_locations, workspace_locations, resource_locations

def example_grid_visible(time:int, visible_time:int, occlude_time:int):
    while time > (visible_time + occlude_time):
        time -= (visible_time + occlude_time)
        
    if time <= visible_time:
        return True
    else:
        return False
    
def create_all_encoding_schemes(max_k:int=4):
    max_k += 1
    combinations = []
    
    for i in range(1, max_k):
        for j in range(1, max_k):
            for k in range(1, max_k):
                for l in range(1, max_k):
                    if i <= j and j <= k and k <= l:
                        # if abs(i - j) < 3 and abs(j - k) < 3 and abs(k - l) < 3:
                        option = [i, j, k, l]
                        combinations.append(option)
                    
    return combinations

# combinations = create_all_encoding_schemes()

# =============================================================================
# TEST
# =============================================================================

# def plot_combinations(combinations):
#     import matplotlib.pyplot as plt
    
#     plt.figure()
#     x = np.arange(.25, 1.25, .25)
    
#     for y in combinations:
#        plt.plot(x, y, alpha=.8)
       
#     plt.xticks(np.arange(.25, 1.25, .25), np.arange(.25, 1.25, .25))
#     plt.xlabel('Condition')
#     plt.ylabel('Encode k')
#     plt.show()

# conditions = [.25, .5, .75, 1]
# combinations = sorted(create_n_encoding_schemes(conditions), reverse=True)
# plot_combinations(combinations)








