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

def compute_dme_activation(current_time:int, activated_at:list,
                           decay:float, epsilon:float):
    
    summed = sum([(current_time - act_at) ** -decay for act_at in activated_at])
    activation = np.log(summed) + epsilon if summed != 0 else epsilon
    
    return activation

def compute_dme_retrieval(current_time:int, activated_at:list,
                          f:float=.9, e:float=.325,
                          noise:float=.28, decay:float=.5):
    
    ai = compute_dme_activation(current_time, activated_at, decay, noise)
    rt = (f * e ** -ai) * 1000
    
    if rt > e:
        return rt, False
    else:    
        return rt, True

def fitts_id(loc1:tuple, loc2:tuple):
    # Fitts' law: ID = log2(2D / W). W = target_size[0] for now
    
    distance = euclidean_distance(loc1, loc2)
    dw = (2 * distance) / constants.TARGET_SIZE[0] + 1
    fitts = np.log2(dw)   

    return fitts

# def fitts_time(loc1:tuple, loc2:tuple, a:int=.025, b:int=.04, sigma:int=.3):
#     # Fitts' time = a + b * ID
#     # a & b can be empirically calculated. Maybe separately for eyes and mouse.
#     duration = a + b * fitts_id(loc1, loc2)
#     noise = random.gauss(mu=1.0, sigma=sigma)
    
#     return duration * noise * 1000 # milliseconds

def estim_saccade_time(loc1:tuple, loc2:tuple, a:int=25, b:int=.04, sigma:int=.25):
    duration = a + (b * euclidean_distance(loc1, loc2))
    noise = random.gauss(mu=(1.0), sigma=sigma)
    result = duration * noise
    
    while result < 5:
        noise = random.gauss(mu=(1.0), sigma=sigma)
        result = duration * noise
    
    return result

def estim_mouse_time(loc1:tuple, loc2:tuple, a:int=15, b:int=105, sigma:int=.25):
    duration = a + (b * fitts_id(loc1, loc2))
    noise = random.gauss(mu=(1.0), sigma=sigma)
    result = duration * noise 
    
    while result < 5:
        noise = random.gauss(mu=(1.0), sigma=sigma)
        result = duration * noise 
        
    return result

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

def get_param_combinations():
    f_range = np.arange(constants.F_RANGE[0], constants.F_RANGE[1], constants.F_RANGE[2])
    e_range = np.arange(constants.E_RANGE[0], constants.E_RANGE[1], constants.E_RANGE[2])
    noise_range = np.arange(constants.NOISE_RANGE[0], constants.NOISE_RANGE[1], constants.NOISE_RANGE[2])
    
    result = []
    
    for l1 in f_range:
        for l2 in e_range:
            for l3 in noise_range:
                result.append([round(l1, 3), round(l2, 3), round(l3, 3)])
                
    return result






