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

import math
import time
from math import atan2

import numpy as np
import pandas as pd
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

                    # If the timer is completed, sometimes blinking would reset the timer. Give an extra 200ms
                    # to allow for the possibility of blinks (will make removing the grid slower).
                    if parent.possibleBlinkStart is None:  # Midline is crossed, start a timer if not yet running
                        parent.possibleBlinkStart = now
                        return 'show', 'hide'  # Keep the grid visible

                    else:
                        # If we've left the example for >200ms, this is probably not a blink, so reset
                        if (now - parent.possibleBlinkStart) > 200:
                            parent.possibleBlinkStart = None
                            parent.crossingStart = None
                            return 'hide', 'hide'
                        else:
                            return 'show', 'hide'

                else:  # Timer is completed but gaze is still on example, so leave the example
                    parent.possibleBlinkStart = None  # Rest the blink timer
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

                # print(f'Time spent away: {round(timeSpentAway, 2)}')

            # Timer is not completed, gaze is < midline, so hide example and show hourglass.
            # This is actual waiting time.
            return 'hide', 'show'


def euclidean_distance(loc1: tuple, loc2: tuple):
    dist = [(a - b) ** 2 for a, b in zip(loc1, loc2)]
    dist = math.sqrt(sum(dist))
    return dist


def get_angle(x):
    return np.rad2deg(atan2(x * constants.PIXEL_WIDTH, constants.SCREENDIST))


def compute_angle_from_center(sample, center):
    return get_angle(euclidean_distance(sample, center))


def calculateMeanError(samples):
    center = constants.SCREEN_CENTER
    try:
        # Discard the first half of samples and the last few
        half_length = int(len(samples) / 2)
        errors = [compute_angle_from_center(sample, center) for sample in samples[-half_length:-10]]

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
            self.parent.dragStartPosition = e.globalPos()
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
            self.parent.writeEvent(f'Incorrectly placed - {e.mimeData().text()} - {self.shouldBe}')
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
                self.parent.correctPlacements['cameFromX'][rowIndex] = self.parent.dragStartPosition.x()
                self.parent.correctPlacements['cameFromY'][rowIndex] = self.parent.dragStartPosition.y()

            except Exception as e:
                print(e)
                # print(f'Item incorrectly placed in ({self.x}, {self.y})')
