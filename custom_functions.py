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

from PyQt5.QtCore import QMimeData, Qt, QTimer
from PyQt5.QtGui import QDrag, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel

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

    eyeLoc = parent.tracker.sample()[0]
    # eyeLoc = parent.mouse.pos().x()
    # gridVisible = parent.exampleGridBox.isVisible()
    # hourglassVisible = parent.hourGlass.isVisible()

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

def customTimerWithReturn(parent, now):
    """Implement your own custom timer. Integrate with eyetracker if necessary.

    Args:
        parent: Receives all class variables from Canvas
        now (int): Receives the current time in milliseconds

    Returns:
        str or None: Return how to update. Returning None retains current visibility state.
                     Alternatives: 'hide', 'show', 'flip'.
    """

    eyeLoc = parent.tracker.sample()
    currentlyVisible = parent.exampleGridBox.isVisible()

    if eyeLoc[0] > MIDLINE:
        parent.crossingStart = None
        return None if currentlyVisible else 'show'

    elif eyeLoc[0] < MIDLINE:
        if parent.crossingStart is None:
            parent.crossingStart = now
            return 'hide' if currentlyVisible else None

        elif (now - parent.crossingStart) < parent.occludedTime:
            return 'hide' if currentlyVisible else None

        elif (now - parent.crossingStart) > parent.occludedTime:
            return 'show' if not currentlyVisible else None

    return None


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
            self.timer.singleShot(700, lambda: self.setStyleSheet("background-color:transparent"))

        else:
            e.acceptProposedAction()
            self.setStyleSheet("background-color:rgba(51, 204, 51, 100)")
            self.timer.singleShot(700, lambda: self.setStyleSheet("background-color:transparent"))

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
