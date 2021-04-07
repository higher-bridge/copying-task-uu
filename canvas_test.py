import sys
from stimulus import load_stimuli

from random import shuffle
import os
import time

from pathlib import Path
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QMimeData, QRect, QPoint
from PyQt5.QtGui import QPixmap, QFont, QDrag, QPainter, QImage
from PyQt5.QtWidgets import (QFrame, QGridLayout, QGroupBox, QLabel,
                             QSizePolicy, QVBoxLayout, QWidget, QApplication)

from stimulus import pick_stimuli

class DraggableLabel(QLabel):

    def __init__(self, parent, image):
        super(DraggableLabel, self).__init__(parent)

        self.dragStartP = None
        self.parent = parent

        self.setFixedSize(image.qimage.size())

        # self.image = image
        self.imageName = image.name
        self.setPixmap(QPixmap(image.qimage))
        self.show()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.dragStartP = e.pos()

    def mouseMoveEvent(self, e):
        if not (e.buttons() & Qt.LeftButton):
            e.mimeData().clear()
            return
        if (e.pos() - self.dragStartP).manhattanLength() < QApplication.startDragDistance():
            return

        mimedata = QMimeData()
        mimedata.setImageData(self.pixmap())
        mimedata.setText(self.imageName)

        drag = QDrag(self)
        drag.setMimeData(mimedata)

        print(self.rect(), self.grab().size())
        pixmap = QPixmap(self.size())
        painter = QPainter(pixmap)
        # painter.drawPixmap(self.rect(), self.grab())
        painter.drawPixmap(QRect(QPoint(0, 0), self.pixmap().size()), self.grab())
        painter.end()

        drag.setPixmap(pixmap)
        drag.setHotSpot(e.pos())
        drag.exec_(Qt.CopyAction | Qt.MoveAction)

    def dragEnterEvent(self, e):
        e.rejectProposedAction()

    def dropEvent(self, e):
        e.rejectProposedAction()
        e.mimeData().clear()

    def mouseReleaseEvent(self, e):
        self.dragStartP = None
        e.rejectProposedAction()
        e.mimeData().clear()


class CustomLabel(QLabel):

    def __init__(self, title, parent, x, y):
        super().__init__(title, parent)

        self.x, self.y = x, y
        self.containedImage = None
        self.parent = parent

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
        e.acceptProposedAction()

        # Set new pixmap in this position if position is valid
        self.setPixmap(QPixmap.fromImage(QImage(e.mimeData().imageData())))

        # Retrieve the image name
        self.containedImage = e.mimeData().text()

        # Clear mimedata
        e.mimeData().clear()

class Canvas(QWidget):
    def __init__(self, images, width: int = 2560, height: int = 1440):

        super().__init__()
        # Set window params
        self.title = 'TEST'
        self.left = 5
        self.top = 5
        self.width = width
        self.height = height

        self.sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.sizePolicy.setHeightForWidth(True)
        self.setSizePolicy(self.sizePolicy)
        self.styleStr = "background-color:transparent"

        # Set stimuli params
        self.nStimuli = 6
        self.allImages = images
        self.imageWidth = 100
        self.nrow = 3
        self.ncol = 3

        # Set experiment params
        self.nTrials = 1

        self.currentTrial = 1
        self.spacePushed = False

        # Set tracking vars
        self.dragStartTime = None
        self.dragStartPos = None

        self.disp = None

        self.inOpeningScreen = True
        self.initUI()

    def eventFilter(self, widget, e):
        if e.type() == QtCore.QEvent.KeyPress:
            key = e.key()
            if key == QtCore.Qt.Key_Space and not self.spacePushed:

                # Set spacePushed to true and remove all widgets
                self.spacePushed = True
                self.clearScreen()

                # Start the task
                self.initTask()
                return True

            # Trigger early exit
            elif key == QtCore.Qt.Key_Tab:
                self.close()
                print('\nEarly exit')
                raise SystemExit(0)

        return QWidget.eventFilter(self, widget, e)

    # =============================================================================
    #     INITIALIZATION OF SCREENS
    # =============================================================================

    def clearScreen(self):
        for i in reversed(range(self.layout.count())):
            self.layout.itemAt(i).widget().setParent(None)

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setStyleSheet("background-color:rgb(128, 128, 128)")
        self.layout = QVBoxLayout()

        self.initOpeningScreen()

    def initOpeningScreen(self):
        self.inOpeningScreen = True
        self.spacePushed = False

        self.label = QLabel('Press space to start')

        self.label.setFont(QFont("Times", 12))
        self.label.setAlignment(Qt.AlignCenter | Qt.AlignHCenter)
        self.installEventFilter(self)

        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

        self.show()

    def initTask(self):
        self.images = pick_stimuli(self.allImages, self.nStimuli)
        # self.grid = np.ones([self.nrow, self.ncol], dtype=bool)

        import example_grid
        self.grid = example_grid.generate_grid(self.images, self.nrow, self.ncol)

        # Create the example grid
        self.createMasterGrid()

        self.layout.addWidget(self.masterGrid)
        self.setLayout(self.layout)

        self.show()

        self.inOpeningScreen = False

    # =============================================================================
    #    GENERATE GRIDS
    # =============================================================================
    def createMasterGrid(self):
        print('master')
        self.masterGrid = QGroupBox("Grid", self)
        layout = QGridLayout()

        # Essentially create 3x6 empty grids, except for in gridLocs
        masterGridRows = 3
        masterGridCols = 6
        gridLocs = [(1, 1), (1, 4), (2, 4)]

        print('empty')
        self.emptyGridLayout()
        for row in range(masterGridRows):
            for col in range(masterGridCols):
                if (row, col) not in gridLocs:
                    layout.addWidget(self.emptyGridBox, row, col)

        print('example')
        self.exampleGridLayout()
        layout.addWidget(self.exampleGridBox, gridLocs[0][0], gridLocs[0][1])

        print('copy')
        self.copyGridLayout()
        layout.addWidget(self.copyGridBox, gridLocs[1][0], gridLocs[1][1])

        print('resource')
        self.resourceGridLayout()
        layout.addWidget(self.resourceGridBox, gridLocs[2][0], gridLocs[2][1])

        self.masterGrid.setLayout(layout)
        self.masterGrid.setTitle('')
        self.masterGrid.setStyleSheet(self.styleStr)

    def emptyGridLayout(self):
        self.emptyGridBox = QGroupBox("Grid", self)
        layout = QGridLayout()

        self.emptyGridBox.setLayout(layout)
        self.emptyGridBox.setTitle('')
        self.emptyGridBox.setStyleSheet(self.styleStr + "; border:0px")

    def exampleGridLayout(self):
        self.exampleGridBox = QGroupBox("Grid", self)
        layout = QGridLayout()

        i = 0
        for x in range(self.nrow):
            for y in range(self.ncol):
                label = QLabel(self)
                label.setFrameStyle(QFrame.Panel)

                if self.grid[x, y]:
                    image = self.images[i]
                    pixmap = QPixmap.fromImage(image.qimage)
                    label.setPixmap(pixmap)
                    label.setAlignment(QtCore.Qt.AlignCenter)
                    label.setSizePolicy(self.sizePolicy)

                    i += 1

                layout.addWidget(label, x, y)

        self.exampleGridBox.setLayout(layout)
        self.exampleGridBox.setTitle('')
        self.exampleGridBox.setSizePolicy(self.sizePolicy)
        self.exampleGridBox.setStyleSheet(self.styleStr)

    def copyGridLayout(self):
        self.copyGridBox = QGroupBox("Grid", self)
        layout = QGridLayout()

        for x in range(self.nrow):
            for y in range(self.ncol):
                label = CustomLabel('', self, x, y)
                label.setFrameStyle(QFrame.Panel)
                label.resize(self.imageWidth, self.imageWidth)
                label.setAlignment(QtCore.Qt.AlignCenter)
                label.setSizePolicy(self.sizePolicy)
                layout.addWidget(label, x, y)

        self.copyGridBox.setLayout(layout)
        self.copyGridBox.setTitle('')
        self.copyGridBox.setStyleSheet(self.styleStr)

    def resourceGridLayout(self):
        self.resourceGridBox = QGroupBox("Grid", self)
        layout = QGridLayout()

        shuffledImages = self.images
        shuffle(shuffledImages)

        i = 0
        row = 0
        col = 0
        for x in range(self.nrow):
            for y in range(self.ncol):
                if self.grid[x, y]:
                    image = shuffledImages[i]
                    label = DraggableLabel(self, image)
                    label.setFrameStyle(QFrame.Panel)  # temp
                    label.setAlignment(QtCore.Qt.AlignCenter)
                    label.setSizePolicy(self.sizePolicy)

                    if i % self.ncol == 0:
                        row += 1
                        col = 0

                    layout.addWidget(label, row, col)
                    i += 1
                    col += 1

        self.resourceGridBox.setLayout(layout)
        self.resourceGridBox.setTitle('')
        self.resourceGridBox.setSizePolicy(self.sizePolicy)
        self.resourceGridBox.setStyleSheet(self.styleStr)


if __name__ == '__main__':
    path = Path(f'{os.getcwd()}/stimuli/')
    images = load_stimuli(path, 100, extension='.png')

    app = QApplication(sys.argv)
    ex = Canvas(images=images)

    ex.showFullScreen()
    sys.exit(app.exec_())
