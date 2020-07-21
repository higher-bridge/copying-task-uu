"""
Created on Thu May 28 15:42:10 2020

@author: mba
"""
import time

import numpy as np
import pandas as pd
# from PyQt5 import QtCore
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
    
    ### EXAMPLE CODE ###
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
            self.parent.dragStartTime = time.time()
            # print(f'Start: {e.pos()}')

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

    def dragEnterEvent(self, e):
        if e.mimeData().hasImage():
            # print('accepted')
            e.accept()
        else:
            # print('rejected')
            e.ignore()
    
    def dropEvent(self, e):
        try:
            # Find in which row of the df x/y == self.x/y and trial == self.trial
            rowIndex = np.where((self.parent.copiedImages['x'] == self.x) &\
                        (self.parent.copiedImages['y'] == self.y) &\
                        (self.parent.copiedImages['Trial'] == self.trial) &\
                        (self.parent.copiedImages['Condition'] == self.condition))
            rowIndex = rowIndex[0][-1]

            # Set new pixmap in this position if position is valid
            self.setPixmap(QPixmap.fromImage(QImage(e.mimeData().imageData())))

            # Retrieve the image name
            self.containedImage = e.mimeData().text()

            # Retrieve drag characteristics
            dragDuration = time.time() - self.parent.dragStartTime
            dragDistance = (e.pos() - self.parent.dragStartPos).manhattanLength()
            # print(f'Moved image {self.containedImage} {dragDistance}px (to ({self.x}, {self.y})) in {dragDuration}s')

            # Fill copiedImages dataframe
            self.parent.copiedImages['Name'][rowIndex] = self.containedImage
            self.parent.copiedImages['Time'][rowIndex] = time.time()
            self.parent.copiedImages['dragDuration'][rowIndex] = dragDuration
            self.parent.copiedImages['dragDistance'][rowIndex] = dragDistance
            
            # If image matches 'shouldBe', set 'Correct' to True
            if self.parent.copiedImages['shouldBe'][rowIndex] == self.containedImage:
                self.parent.copiedImages['Correct'][rowIndex] = True
        
        except:
            print(f'Item could not be placed in ({self.x}, {self.y})')
