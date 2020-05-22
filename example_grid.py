#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 19:00:15 2020
Author: Alex Hoogerbrugge (@higher-bridge)
"""

import numpy as np
from random import choices

class Examplegrid():
    # Represents the exaple grid from which participants copy
    def __init__(self, nrow:int, ncol:int):
        self.nrow = nrow
        self.ncol = ncol

def generate_grid(stimuli:list, nrow:int, ncol:int):
    """Returns an array with grid positions for stimuli. If there are fewer
    stimuli than locations, random positions are chosen. """
        
    if nrow * ncol == len(stimuli):
        grid = np.ones([nrow, ncol], dtype=bool)
        return grid
    
    else:
        grid = np.zeros([nrow, ncol], dtype=bool)
        
        random_x = choices(range(nrow), k=len(stimuli))
        random_y = choices(range(ncol), k=len(stimuli))
        
        for (x, y) in zip(random_x, random_y):
            grid[x, y] = True
        
        return grid

