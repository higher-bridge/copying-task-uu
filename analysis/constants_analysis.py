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

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
RESULT_DIR = ROOT_DIR / 'results'
# base_location = '../results'

N_JOBS = 8

# EXPERIMENT PARAMETERS
EXCLUDE_TRIALS = [1, 2, 3, 4, 5, 999]
EXCLUDE_EXCEL_BASED = False

NUM_TRIALS = 1

STIMULI_PER_TRIAL = 6

CONDITIONS = [0, 1]
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

# FIXATION DETECTION
SAMPLING_RATE       = 250
STEP_SIZE           = 1000
HESSELS_SAVGOL_LEN  = 31          # Window length of Savitzky-Golay filter in pre-processing
HESSELS_THR         = 10e12       # Initial slow/fast phase threshold
HESSELS_LAMBDA      = 2.5         # Number of standard deviations (default 2.5)
HESSELS_MAX_ITER    = 100         # Max iterations for threshold adaptation (default 200)
HESSELS_WINDOW_SIZE = 8 * SAMPLING_RATE      # Threshold adaptation window (default 8 seconds) * sampling rate
HESSELS_MIN_AMP     = 1.0         # Minimal amplitude of fast candidates for merging slow candidates (default 1.0)
HESSELS_MIN_FIX     = 60         # Minimal fixation duration in ms (default .06)
PX2DEG              = 0.01982

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
