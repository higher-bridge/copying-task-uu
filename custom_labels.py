"""
Created on Thu May 28 15:42:10 2020

@author: mba
"""

from PyQt5 import QtCore
from PyQt5.QtWidgets import QLabel, QApplication
from PyQt5.QtGui import QPixmap, QImage, QDrag, QPainter
from PyQt5.QtCore import QMimeData, Qt

class DraggableLabel(QLabel):
    
    def __init__(self,parent,image):
        super(QLabel,self).__init__(parent)
        
        self.setPixmap(QPixmap(image))    
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

        drag.setMimeData(mimedata)
        pixmap = QPixmap(self.size())
        painter = QPainter(pixmap)
        painter.drawPixmap(self.rect(), self.grab())
        painter.end()
        
        drag.setPixmap(pixmap)
        drag.setHotSpot(e.pos())
        drag.exec_(Qt.CopyAction | Qt.MoveAction)

class CustomLabel(QLabel):
    
    def __init__(self, title, parent):
        super().__init__(title, parent)
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