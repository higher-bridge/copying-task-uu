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

import os
import sys
from random import shuffle

from PyQt5.QtWidgets import QApplication
from pathlib import Path

from canvas import Canvas
from stimulus import load_stimuli

NUM_STIMULI = 6
IMAGE_WIDTH = 100
ROW_COL_NUM = 3
TIME_OUT_MS = 30000

NUM_TRIALS = 5
CONDITIONS = [(4000, 0),
              (3000, 1000),
              (2000, 2000),
              (1000, 3000)]

CONDITION_ORDER = [0] # The lowest number should be 0
shuffle(CONDITION_ORDER)
print(CONDITION_ORDER)

if __name__ == '__main__':
    path = Path(f'{os.getcwd()}/stimuli/')
    images = load_stimuli(path, IMAGE_WIDTH, extension='.png')
        
    app = QApplication(sys.argv)    

    ex = Canvas(images=images, nStimuli=NUM_STIMULI, imageWidth=IMAGE_WIDTH, 
                nrow=ROW_COL_NUM, ncol=ROW_COL_NUM,
                conditions=CONDITIONS, conditionOrder=CONDITION_ORDER, 
                nTrials=NUM_TRIALS, useCustomTimer=False, trialTimeOut=TIME_OUT_MS)
    
    ex.showFullScreen()
    
    sys.exit(app.exec_())
