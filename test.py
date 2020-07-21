"""
Created on Mon Mar  2 20:03:30 2020

@author: mba
"""

import os
import sys

from PyQt5.QtWidgets import QApplication
from pathlib import Path

from canvas import Canvas
from stimulus import load_stimuli

NUM_STIMULI = 4
IMAGE_WIDTH = 100
ROW_COL_NUM = 3

NUM_TRIALS = 2
CONDITIONS = [(4000, 0),
              (3000, 1000),
              (2000, 2000),
              (1000, 3000)]
CONDITION_ORDER = [0, 1, 2, 3] # The lowest number should be 0


if __name__ == '__main__':
    path = Path(f'{os.getcwd()}/stimuli/')
    images = load_stimuli(path, IMAGE_WIDTH, extension='.png')
        
    app = QApplication(sys.argv)    

    ex = Canvas(images=images, nStimuli=NUM_STIMULI, imageWidth=IMAGE_WIDTH, 
                nrow=ROW_COL_NUM, ncol=ROW_COL_NUM,
                conditions=CONDITIONS, conditionOrder=CONDITION_ORDER, 
                nTrials=NUM_TRIALS, useCustomTimer=False)
    
    # ex.showFullScreen()
    
    sys.exit(app.exec_())
