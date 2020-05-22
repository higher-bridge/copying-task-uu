#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:55:34 2020
Author: Alex Hoogerbrugge (@higher-bridge)
"""

import numpy as np
import math
import random

def euclidean_distance(loc1:tuple, loc2:tuple):
    dist = [(a - b) ** 2 for a, b in zip(loc1, loc2)]
    dist = math.sqrt(sum(dist))
    
    return dist

def fitts_id(loc1:tuple, loc2:tuple, target_size:tuple):
    # Fitts' law: ID = log2(2D / W). W = target_size[0] for now
    distance = euclidean_distance(loc1, loc2)
    fitts = np.log2((2 * distance) / target_size[0])

    return fitts

def fitts_time(loc1:tuple, loc2:tuple, target_size:tuple, a:int=.05, b:int=.1, sigma:int=.1):
    # Fitts' time = a + b * ID
    # a & b can be empirically calculated. Maybe separately for eyes and mouse.
    duration = a + b * fitts_id(loc1, loc2, target_size)
    noise = random.gauss(mu=1.0, sigma=sigma)
    
    return duration * noise

# def create_n_encoding_schemes(conditions:list, n:int=100, min_k:int=1, max_k:int=6):
#     # Creates n unique schemes (lists) of length len(conditions), 
#     # with values in the range min_k:max_k. schemes can start at any value,
#     # but thereafter must be descending or equal, and the difference may be 2 at most. 
#     # There are many combinations in reality, so this is a sampling of the 
#     # full problem space.
#     combinations = []
    
#     i = 0
#     while i < n:
#         option = []
        
#         first = True
#         temp_max_k = max_k
        
#         while len(option) < len(conditions):
#             num = random.randint(min_k, temp_max_k)
            
#             if first:
#                 option.append(num)
#                 temp_max_k = num
#                 first = False
            
#             elif num <= temp_max_k:
#                 if abs(num - temp_max_k) <= 2:
#                     option.append(num)
#                     temp_max_k = num            
                
            
#         if option not in combinations:
#             combinations.append(option)
#             i += 1
    
#     return combinations

def create_all_encoding_schemes(max_k=6):
    max_k += 1
    combinations = []
    
    for i in range(1, max_k):
        for j in range(1, max_k):
            for k in range(1, max_k):
                for l in range(1, max_k):
                    if i >= j and j >= k and k >= l:
                        # if abs(i - j) < 3 and abs(j - k) < 3 and abs(k - l) < 3:
                        option = [i, j, k, l]
                        combinations.append(option)
                    
    return combinations

combinations = create_all_encoding_schemes()

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








