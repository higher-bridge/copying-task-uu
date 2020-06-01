"""
Created on Wed Feb 26 19:00:15 2020
Author: Alex Hoogerbrugge (@higher-bridge)
"""

import numpy as np
from random import sample

class Examplegrid():
    # Represents the exaple grid from which participants copy
    def __init__(self, nrow:int, ncol:int):
        self.nrow = nrow
        self.ncol = ncol

def generate_grid(stimuli:list, nrow:int, ncol:int):
    """Returns an array with grid positions for stimuli. If there are fewer
    stimuli than locations, random positions are chosen.

    Arguments:
        stimuli {list} -- [description]
        nrow {int} -- [description]
        ncol {int} -- [description]

    Returns:
        np.array([nrow, ncol], dtype=bool) -- [description]
    """
        
    if nrow * ncol == len(stimuli):
        grid = np.ones([nrow, ncol], dtype=bool)
        return grid
    
    else:
        grid = np.zeros([nrow, ncol], dtype=bool)

        k = 0
        while k < len(stimuli):
            x = sample(range(nrow), k=1)
            y = sample(range(ncol), 1)

            if grid[x, y] == False:
                grid[x, y] = True
                k += 1
        
        return grid

