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
EXCLUDE_TRIALS = [999]
EXCLUDE_EXCEL_BASED = True

NUM_JOBS_SIM = 8
NUM_TRIALS = 4

STIMULI_PER_TRIAL = 6

CONDITIONS = [0, 1, 2, 3]
CONDITION_TIMES = [(6000, 0), (4000, 2000), (3000, 3000), (2000, 4000)]
SUM_DURATION = 6000

TIMEOUT = 42000

# COMPUTER PARAMETERS
TARGET_SIZE = (50, 50)  # I have divided by 2 because it is more likely to grab
# the stimuli around the center
MIDLINE = 1200

RESOLUTION = (2560, 1440)
DIMENSIONS = (598, 336)  # mm size of screen
DISTANCE = 675  # mm distance from screen

# MODEL PARAMETERS
MAX_MEMORY_REPETITIONS = 3

# Consist of (start, stop, step)
F_RANGE = (.1, .5, .1)
DECAY_RANGE = (.5, 1., .1)
THRESH_RANGE = (.175, .275, .025)
NOISE_RANGE = (.26, .32, .02)

ERROR_RATES = [.11, .2, .29, .38]

# Center locations of each stimulus
all_example_locations = [(515, 570), (650, 570), (785, 570),
                         (515, 720), (650, 720), (785, 720),
                         (515, 870), (650, 870), (785, 870)]
all_workspace_locations = [(1780, 570), (1910, 570), (2045, 570),
                           (1780, 720), (1910, 720), (2045, 720),
                           (1780, 870), (1910, 870), (2045, 870)]
all_resource_locations = [(1780, 1080), (1910, 1080), (2045, 1080),
                          (1780, 1300), (1910, 1300), (2045, 1300)]

example_min_x = min([x[0] - 100 for x in all_example_locations])
example_min_y = min(x[1] - 100 for x in all_example_locations)
example_max_x = max([x[0] + 100 for x in all_example_locations])
example_max_y = max(x[1] + 100 for x in all_example_locations)

example_boundaries = [(example_min_x, example_min_y),
                      (example_max_x, example_max_y)]

resource_grid_corner = (1715, 1010)  # , (2110, 1370)]
resource_grid_size = (395, 360)

# NO NEED TO FILL IN
SCREEN_CENTER = (RESOLUTION[0] / 2, RESOLUTION[1] / 2)
PIXEL_WIDTH = DIMENSIONS[0] / RESOLUTION[0]
PIXEL_HEIGHT = DIMENSIONS[1] / RESOLUTION[1]
