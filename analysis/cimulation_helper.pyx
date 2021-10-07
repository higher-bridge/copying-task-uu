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

from libc.math cimport exp, log, log2, sqrt

import random

import constants_analysis as constants


cdef float euclidean_distance(tuple loc1, tuple loc2):
    cdef list dist
    dist = [(a - b) ** 2 for a, b in zip(loc1, loc2)]
    return sqrt(sum(dist))

cdef float compute_dme_activation(int current_time, list activated_at,
                                  float decay, float noise):
    cdef float summed, act_at, log_noise, activation
    
    summed = sum([abs(current_time - act_at) ** (-decay) for act_at in activated_at])
    
    log_noise = random.gauss(0, noise) ** 2
    activation = log(summed) + log_noise  # if summed != 0 else log_noise
    
    return activation

cpdef tuple compute_dme_retrieval(int current_time, list activated_at,
                          float f=.9, float thresh=.325,
                          float noise=.28, float decay=.5):
    cdef float ai, rt
    cdef bint surpassed

    ai = compute_dme_activation(current_time, activated_at, decay, noise)
    rt = f * exp(-ai)
    surpassed = ai > thresh
    
    return (rt, surpassed)

cdef float fitts_id(tuple loc1, tuple loc2):
    # Fitts' law: ID = log2(2D / W). W = target_size[0] for now

    cdef float distance, dw, fitts

    distance = euclidean_distance(loc1, loc2)
    dw = (2 * distance) / constants.TARGET_SIZE[0] + 1
    fitts = log2(dw)   

    return fitts

cpdef int estim_saccade_time(tuple loc1, tuple loc2, int a=25, float b=.04, float sigma=.25):
    cdef float duration, noise
    cdef int result

    duration = a + (b * euclidean_distance(loc1, loc2))
    noise = random.gauss(mu=1.0, sigma=sigma)
    result = int(round(duration * noise))
        
    return result

cpdef int estim_mouse_time(tuple loc1, tuple loc2, int a=15, float b=105, float sigma=.25):
    cdef float duration, noise
    cdef int result

    duration = a + (b * fitts_id(loc1, loc2))
    noise = random.gauss(mu=1.0, sigma=sigma)
    result = int(round(duration * noise))
        
    return result

cpdef generate_locations(n_items:int=4):    
    all_example_locations = constants.all_example_locations
    all_workspace_locations = constants.all_workspace_locations
    all_resource_locations = constants.all_resource_locations
    
    # Sample a set of locations so we can index matching one from example and workspace
    cdef locations = random.sample(list(range(len(all_example_locations))), n_items)
    
    example_locations = [all_example_locations[i] for i in locations]
    workspace_locations = [all_workspace_locations[i] for i in locations]
    resource_locations = all_resource_locations[0:n_items]
    
    return example_locations, workspace_locations, resource_locations

cpdef example_grid_visible(int time, int visible_time, int occlude_time):
    cdef summed_time = visible_time + occlude_time
    
    while time > summed_time:
        time -= summed_time
        
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
                        option = [i, j, k, l]
                        combinations.append(option)
                    
    return combinations

def get_param_combinations():
    f_range = np.arange(constants.F_RANGE[0], constants.F_RANGE[1], constants.F_RANGE[2])
    decay_range = np.arange(constants.DECAY_RANGE[0], constants.DECAY_RANGE[1], constants.DECAY_RANGE[2])
    thresh_range = np.arange(constants.THRESH_RANGE[0], constants.THRESH_RANGE[1], constants.THRESH_RANGE[2])
    noise_range = np.arange(constants.NOISE_RANGE[0], constants.NOISE_RANGE[1], constants.NOISE_RANGE[2])

    # print(len(f_range), len(decay_range), len(thresh_range), len(noise_range))

    result = []
    
    for l1 in f_range:
        for l2 in decay_range:
            for l3 in thresh_range:
                for l4 in noise_range:
                    result.append([round(l1, 3), round(l2, 3), round(l3, 3), round(l4, 3)])
                
    return result






