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

base_location = '../results'

# EXPERIMENT PARAMETERS
EXCLUDE_TRIALS = [1, 2, 3, 999]
NUM_PPS_SIM = 8
NUM_TRIALS = 4

STIMULI_PER_TRIAL = 4

CONDITIONS = [0, 1, 2, 3]
CONDITION_TIMES = [(6000, 0), (4000, 2000), (3000, 3000), (2000, 4000)]
SUM_DURATION = 6000

TIMEOUT = 20000

# COMPUTER PARAMETERS
TARGET_SIZE = (50, 50) # I have divided by 2 because it is more likely to grab
                       # the stimuli around the center
MIDLINE = 1200

RESOLUTION = (2560, 1440)
DIMENSIONS = (598, 336) # mm size of screen
DISTANCE = 700 # mm distance from screen

# MODEL PARAMETERS
MAX_MEMORY_REPETITIONS = 3

# Consist of (start, stop, step)
F_RANGE = (.1, .5, .1) 
DECAY_RANGE = (.5, 1., .1)
THRESH_RANGE = (.175, .275, .025)
NOISE_RANGE = (.26, .32, .02)


ERROR_RATES = [.11, .2, .29, .38]

# NO NEED TO FILL IN
SCREEN_CENTER = (RESOLUTION[0] / 2, RESOLUTION[1] / 2)
PIXEL_WIDTH = DIMENSIONS[0] / RESOLUTION[0]
PIXEL_HEIGHT = DIMENSIONS[1] / RESOLUTION[1] 

