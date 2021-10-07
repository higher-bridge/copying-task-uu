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

from random import sample

import numpy as np


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

