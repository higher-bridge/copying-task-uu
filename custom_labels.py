"""
Created on Thu May 28 15:42:10 2020

@author: mba
"""
import time

import numpy as np
import pandas as pd
from PyQt5 import QtCore
from PyQt5.QtCore import QMimeData, Qt
from PyQt5.QtGui import QDrag, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel


class DraggableLabel(QLabel):
    
    def __init__(self,parent,image):
        super(QLabel,self).__init__(parent)
        
        self.image = image
        self.setPixmap(QPixmap(self.image.qimage))    
        self.show()
    
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drag_start_position = e.pos()

    def mouseMoveEvent(self, e):
        if not (e.buttons() & Qt.LeftButton):
            return
        if (e.pos() - self.drag_start_position).manhattanLength() < QApplication.startDragDistance():
            return
       
        drag = QDrag(self)
        mimedata = QMimeData()
        # mimedata.setText(self.text())
        mimedata.setImageData(self.pixmap().toImage())
        mimedata.setText(self.image.name)

        drag.setMimeData(mimedata)
        pixmap = QPixmap(self.size())
        painter = QPainter(pixmap)
        painter.drawPixmap(self.rect(), self.grab())
        painter.end()
        
        drag.setPixmap(pixmap)
        drag.setHotSpot(e.pos())
        drag.exec_(Qt.CopyAction | Qt.MoveAction)
        # print(f'Moved image {self.image.name}')

class CustomLabel(QLabel):
    
    def __init__(self, title, parent, x, y, trial):
        super().__init__(title, parent)

        self.x, self.y = x, y
        self.containedImage = None
        self.parent = parent
        self.trial = trial
        # parent.copiedImages[x, y] = None

        self.setAcceptDrops(True)
        # self.setDragEnabled(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasImage():
            print('accepted')
            e.accept()
        else:
            print('rejected')
            e.ignore()
    
    def dropEvent(self, e):
        self.setPixmap(QPixmap.fromImage(QImage(e.mimeData().imageData())))
        self.containedImage = e.mimeData().text()
        # self.parent.copiedImages[self.x, self.y] = self.containedImage
        
        # rowIndexX = list(np.where(self.parent.copiedImages['x'] == self.x))
        # print(rowIndexX)
        # rowIndexY = list(np.where(self.parent.copiedImages['y'] == self.y))
        # print(rowIndexY)

        # rowLoc = [i for i in rowIndexY if i in rowIndexX][0]
        # rowIndex = np.argwhere(rowIndexX == rowIndexY)
        
        rowIndex = np.where((self.parent.copiedImages['x'] == self.x) &\
            (self.parent.copiedImages['y'] == self.y) &\
                (self.parent.copiedImages['Trial'] == self.trial))
        
        print(rowIndex)
        rowIndex = rowIndex[0][0]
        print(rowIndex)

        self.parent.copiedImages['Name'][rowIndex] = self.containedImage
        self.parent.copiedImages['Time'][rowIndex] = time.time()
        
        if self.parent.copiedImages['shouldBe'][rowIndex] == self.containedImage:
            self.parent.copiedImages['Correct'][rowIndex] = True
        
        # placeDict = pd.DataFrame({'x': self.x, 'y': self.y, 'Name': self.containedImage, 'Time': time.time()}, index=[0])
        # self.parent.copiedImages = self.parent.copiedImages.append(placeDict, ignore_index=True)
        print(f'Moved {self.containedImage} to ({self.x}, {self.y})')
