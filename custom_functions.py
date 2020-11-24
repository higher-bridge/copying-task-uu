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

from PyQt5.QtCore import QMimeData, Qt
from PyQt5.QtGui import QDrag, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel


def customTimer(parent, now):
    """Implement your own custom timer. Integrate with eyetracker if necessary.

    Args:
        parent: Receives all class variables from Canvas
        now (int): Receives the current time in milliseconds

    Returns:
        bool: Return whether to update. Returning True inverts the current occlusion settings
    """
    
    shouldUpdate = False
    
    # If example grid is visible and visible time expires, update
    if parent.exampleGridBox.isVisible():
        if now - parent.start >= parent.visibleTime:
            shouldUpdate = True
            
    # If example grid is occluded and occlusion time expires, update
    else:
        if now - parent.start >= parent.occludedTime:
            shouldUpdate = True
    
    return shouldUpdate

class DraggableLabel(QLabel):
    
    def __init__(self, parent, image):
        super(QLabel,self).__init__(parent)
        
        self.parent = parent
        self.image = image
        self.setPixmap(QPixmap(self.image.qimage))   
        self.show()
    
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.parent.dragStartPos = e.pos()
            self.parent.dragStartTime = round(time.time() * 1000)

    def mouseMoveEvent(self, e):
        if not (e.buttons() & Qt.LeftButton):
            return
        if (e.pos() - self.parent.dragStartPos).manhattanLength() < QApplication.startDragDistance():
            return

        mimedata = QMimeData()
        mimedata.setImageData(self.pixmap().toImage())
        mimedata.setText(self.image.name)

        drag = QDrag(self)
        drag.setMimeData(mimedata)
        pixmap = QPixmap(self.size())
        painter = QPainter(pixmap)
        painter.drawPixmap(self.rect(), self.grab())
        painter.end()
        
        drag.setPixmap(pixmap)
        drag.setHotSpot(e.pos())
        drag.exec_(Qt.CopyAction | Qt.MoveAction)

class CustomLabel(QLabel):
    
    def __init__(self, title, parent, x, y, trial, condition):
        super().__init__(title, parent)

        self.x, self.y = x, y
        self.containedImage = None
        self.parent = parent
        self.trial = trial
        self.condition = condition

        self.setAcceptDrops(True)

    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            self.setPixmap(QPixmap())
            self.containedImage = None
    
    def dragEnterEvent(self, e):
        if e.mimeData().hasImage():
            e.accept()
        else:
            e.ignore()
    
    def dropEvent(self, e):
        # Set new pixmap in this position if position is valid
        self.setPixmap(QPixmap.fromImage(QImage(e.mimeData().imageData())))

        # Retrieve the image name
        self.containedImage = e.mimeData().text()

        try:
            # Find in which row of the df x/y == self.x/y and trial == self.trial
            rowIndex = np.where((self.parent.correctPlacements['x'] == self.x) &\
                        (self.parent.correctPlacements['y'] == self.y) &\
                        (self.parent.correctPlacements['Trial'] == self.trial) &\
                        (self.parent.correctPlacements['Condition'] == self.condition))
            rowIndex = rowIndex[0][-1]

            # Retrieve drag characteristics
            dragDuration = round(time.time() * 1000) - self.parent.dragStartTime
            dragDistance = (e.pos() - self.parent.dragStartPos).manhattanLength()
            # print(f'Moved image {self.containedImage} {dragDistance}px (to ({self.x}, {self.y})) in {dragDuration}s')

            # Fill correctPlacements dataframe
            self.parent.correctPlacements['Name'][rowIndex] = self.containedImage
            self.parent.correctPlacements['Time'][rowIndex] = round(time.time() * 1000)
            self.parent.correctPlacements['dragDuration'][rowIndex] = dragDuration
            self.parent.correctPlacements['dragDistance'][rowIndex] = dragDistance
            
            # If image matches 'shouldBe', set 'Correct' to True
            shouldBe = self.parent.correctPlacements['shouldBe'][rowIndex]
            if shouldBe == self.containedImage:
                self.parent.correctPlacements['Correct'][rowIndex] = True
        
        except:
            print(f'Item incorrectly placed in ({self.x}, {self.y})')

        
        # Now write regardless of correctness to a different df
        try:
            correct = shouldBe == self.containedImage
        except:
            shouldBe = 'Empty'
            correct = False

        allPlacementsDict = pd.DataFrame({'x': self.x,
                             'y': self.y,
                             'Name': self.containedImage,
                             'shouldBe': shouldBe,
                             'Correct': correct,
                             'Time': round(time.time() * 1000),
                             'Trial': self.trial,
                             'Condition': self.condition,
                             'visibleTime': self.parent.visibleTime
                             }, index=[0])
        self.parent.allPlacements = self.parent.allPlacements.append(allPlacementsDict, ignore_index=True)
