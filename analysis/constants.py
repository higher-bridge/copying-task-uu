base_location = '../results'

# EXPERIMENT PARAMETERS
EXCLUDE_TRIALS = [1, 2, 3, 999]
NUM_PPS_SIM = 8
NUM_TRIALS = 10

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
F_RANGE = (.5, 1., .1) 
DECAY_RANGE = (.4, .8, .1)
THRESH_RANGE = (.225, .325, .025)
NOISE_RANGE = (.24, .32, .02)



# NO NEED TO FILL IN
SCREEN_CENTER = (RESOLUTION[0] / 2, RESOLUTION[1] / 2)
PIXEL_WIDTH = DIMENSIONS[0] / RESOLUTION[0]
PIXEL_HEIGHT = DIMENSIONS[1] / RESOLUTION[1] 

