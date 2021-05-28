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

import time

import numpy as np
import pandas as pd
import math

from PyQt5.QtCore import QMimeData, Qt, QTimer
from PyQt5.QtGui import QDrag, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel

import constants
from constants import MIDLINE


def customTimer(parent, now):
    """Implement your own custom timer. Integrate with eyetracker if necessary.

    Args:
        parent: Receives all class variables from Canvas
        now (int): Receives the current time in milliseconds

    Returns:
        str or None: Return how to update. Returning None retains current visibility state.
                     Alternatives: 'hide', 'show', 'flip'.
    """

    if parent.occludedTime == 0:
        return 'show', 'hide'

    eyeLoc = parent.tracker.sample()[0]
    # eyeLoc = parent.mouse.pos().x()

    if eyeLoc == -1:
        return None, None

    if parent.crossingStart is None:  # No timer is running yet
        if eyeLoc < MIDLINE:  # Start the counter, hide the example grid if visible, show hourglass
            parent.crossingStart = now
            return 'hide', 'show'

        else:  # If timer isn't running and gaze > midline, hide example, do not show hourglass
            return 'hide', 'hide'

    else:  # Timer is running
        if not parent.backCrossing:  # Only check for completion if we're not backcrossing
            if (now - parent.crossingStart) > parent.occludedTime:  # Timer is completed
                if eyeLoc > MIDLINE:  # Gaze > midline, so reset timer and hide both
                    parent.crossingStart = None
                    return 'hide', 'hide'

                else:  # Timer is completed but gaze is still on example, so leave the example
                    return 'show', 'hide'

        if eyeLoc > MIDLINE:
            if not parent.backCrossing:  # We're now starting a backcrossing, so note the time for that.
                parent.backCrossing = True
                parent.backCrossStart = now
            else:  # If we were already backcrossing, do nothing
                pass
            return 'hide', 'hide'  # Do not show example nor hourglass (since timer isn't counting)

        else:  # If eyeLoc < midline
            if parent.backCrossing:  # We were backcrossing, but gaze is < midline now, so stop the timer
                parent.backCrossing = False
                timeSpentAway = now - parent.backCrossStart
                parent.crossingStart += timeSpentAway

                print(f'Time spent away: {round(timeSpentAway, 2)}')

            # Timer is not completed, gaze is < midline, so hide example and show hourglass.
            # This is actual waiting time.
            return 'hide', 'show'


def customTimerWithoutPause(parent, now):
    """Implement your own custom timer. Integrate with eyetracker if necessary.

    Args:
        parent: Receives all class variables from Canvas
        now (int): Receives the current time in milliseconds

    Returns:
        str or None: Return how to update. Returning None retains current visibility state.
                     Alternatives: 'hide', 'show', 'flip'.
    """

    if parent.occludedTime == 0:
        return 'show', 'hide'

    eyeLoc = parent.tracker.sample()[0]
    # eyeLoc = parent.mouse.pos().x()

    if eyeLoc == -1:
        return None, None

    if parent.crossingStart is None:
        # If midline is crossed to the left, start a counter (crossingStart)
        # and hide the example
        if eyeLoc < MIDLINE:
            parent.crossingStart = now
            return 'hide', 'show'

    # If gaze is on the right and no counter is running, hide example grid and show no hourglass
        elif eyeLoc > MIDLINE:
            return 'hide', 'hide'

    # If a counter is running, hide the grid until time is reached. This is
    # regardless of where gaze position is
    else:
        if (now - parent.crossingStart) < parent.occludedTime:
            return 'hide', 'show'

        elif (now - parent.crossingStart) > parent.occludedTime:
            if eyeLoc > MIDLINE:
                parent.crossingStart = None
                
            return 'show', 'hide'

    return None, None


def euclidean_distance(loc1: tuple, loc2: tuple):
    dist = [(a - b) ** 2 for a, b in zip(loc1, loc2)]
    dist = math.sqrt(sum(dist))
    return dist


def calculateMeanError(samples):
    center = constants.SCREEN_CENTER

    errors = [euclidean_distance(sample, center) for sample in samples]

    try:
        result = (round(np.mean(errors), 2), round(np.std(errors), 2))
    except:
        result = (np.nan, np.nan)

    return result


class DraggableLabel(QLabel):

    def __init__(self, parent, image):
        super(DraggableLabel, self).__init__(parent)

        self.dragStartP = None
        self.parent = parent

        self.setFixedSize(image.qimage.size())

        self.imageName = image.name
        self.setPixmap(QPixmap(image.qimage))
        self.show()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            # Local tracking var
            self.dragStartP = e.pos()

            # 'Global' tracking var
            self.parent.dragStartPosition = e.pos()
            self.parent.dragStartTime = round(time.time() * 1000)

    def mouseMoveEvent(self, e):
        if not (e.buttons() & Qt.LeftButton):
            return
        if (e.pos() - self.dragStartP).manhattanLength() < QApplication.startDragDistance():
            return

        mimedata = QMimeData()
        mimedata.setImageData(self.pixmap())
        mimedata.setText(self.imageName)

        drag = QDrag(self)
        drag.setMimeData(mimedata)

        pixmap = QPixmap(self.size())
        painter = QPainter(pixmap)
        painter.drawPixmap(self.rect(), self.grab())
        painter.end()

        drag.setPixmap(pixmap)
        drag.setHotSpot(e.pos())
        drag.exec_(Qt.CopyAction | Qt.MoveAction)

    def dragEnterEvent(self, e):
        e.ignore()

    def dropEvent(self, e):
        e.ignore()

    def mouseReleaseEvent(self, e):
        e.ignore()


class CustomLabel(QLabel):

    def __init__(self, title, parent, x, y, trial, condition, shouldBe=''):
        super().__init__(title, parent)

        self.x, self.y = x, y
        self.containedImage = None
        self.parent = parent
        self.trial = trial
        self.condition = condition

        self.shouldBe = shouldBe
        self.timer = QTimer(self)

        self.setAcceptDrops(True)

    def setTransparent(self):
        try:
            self.setStyleSheet("background-color:transparent")
        except Exception as e:
            print(e)

    def singleShotTransparent(self):
        try:
            self.timer.singleShot(700, self.setTransparent)
        except Exception as e:
            print(e)

    def mousePressEvent(self, e):
        e.ignore()

    def dragEnterEvent(self, e):
        if e.mimeData().hasImage():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        if self.containedImage is not None:
            e.ignore()

        # Check if item was dropped in the correct location
        elif e.mimeData().text() != self.shouldBe:
            e.ignore()
            self.setStyleSheet("background-color:rgba(255, 51, 0, 200)")
            self.singleShotTransparent()

        else:
            e.acceptProposedAction()
            self.setStyleSheet("background-color:rgba(51, 204, 51, 100)")
            self.singleShotTransparent()

            # Set new pixmap in this position if position is valid
            self.setPixmap(QPixmap.fromImage(QImage(e.mimeData().imageData())))

            # Retrieve the image name
            self.containedImage = e.mimeData().text()

            # Clear mimedata
            e.mimeData().clear()

            try:
                # Find in which row of the df x/y == self.x/y and trial == self.trial
                rowIndex = np.where((self.parent.correctPlacements['x'] == self.x) & \
                                    (self.parent.correctPlacements['y'] == self.y) & \
                                    (self.parent.correctPlacements['Trial'] == self.trial) & \
                                    (self.parent.correctPlacements['Condition'] == self.condition))
                rowIndex = rowIndex[0][-1]

                # Retrieve drag characteristics
                dragDuration = round(time.time() * 1000) - self.parent.dragStartTime
                dragDistance = (e.pos() - self.parent.dragStartPosition).manhattanLength()

                # Fill correctPlacements dataframe
                self.parent.correctPlacements['Name'][rowIndex] = self.containedImage
                self.parent.correctPlacements['Time'][rowIndex] = round(time.time() * 1000)
                self.parent.correctPlacements['dragDuration'][rowIndex] = dragDuration
                self.parent.correctPlacements['dragDistance'][rowIndex] = dragDistance
                self.parent.correctPlacements['Correct'][rowIndex] = True

            except:
                print(f'Item incorrectly placed in ({self.x}, {self.y})')
