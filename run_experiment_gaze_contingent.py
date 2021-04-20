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

import sys
from random import shuffle

from PyQt5.QtWidgets import QApplication
from pathlib import Path

from canvas import Canvas
from stimulus import load_stimuli
from constants import (NUM_STIMULI,
                       IMAGE_WIDTH,
                       ROW_COL_NUM,
                       TIME_OUT_MS)

NUM_TRIALS = 15

# Define as (0, GAZE_CONTINGENT_DELAY_IN_MS)
CONDITIONS = [(0, 0),     # No occlusion delay
              (0, 500),   # 500ms occlusion delay
              (0, 1000),  # 100ms
              (0, 1500)]  # etc.

CONDITION_ORDER = [0, 1, 2, 3]  # The lowest number should be 0
shuffle(CONDITION_ORDER)
print(CONDITION_ORDER)

if __name__ == '__main__':
    project_folder = Path(__file__).parent
    path = project_folder / 'stimuli'
    images = load_stimuli(path, IMAGE_WIDTH, extension='.png')

    app = QApplication(sys.argv)

    ex = Canvas(images=images, nStimuli=NUM_STIMULI, imageWidth=IMAGE_WIDTH,
                nrow=ROW_COL_NUM, ncol=ROW_COL_NUM,
                conditions=CONDITIONS, conditionOrder=CONDITION_ORDER,
                nTrials=NUM_TRIALS, useCustomTimer=True, addNoise=False,
                customCalibration=True,
                trialTimeOut=TIME_OUT_MS)

    ex.showFullScreen()

    sys.exit(app.exec_())
