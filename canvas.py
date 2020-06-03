"""
Created on Wed Feb 26 19:10:04 2020
Author: Alex Hoogerbrugge (@higher-bridge)
"""

import time
from random import shuffle

import numpy as np
import pandas as pd
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QFrame, QGridLayout, QGroupBox, QLabel,
                             QSizePolicy, QVBoxLayout, QWidget)

import example_grid
from stimulus import pick_stimuli
from custom_labels import CustomLabel, DraggableLabel


class Canvas(QWidget):
    # Generates a canvas with a Copygrid and an Examplegrid
    def __init__(self, images:list, nStimuli:int, imageWidth:int, nrow:int, ncol:int, 
                 left:int=10, top:int=10, width:int=640, height:int=480,
                 visibleTime:int=2000, occludedTime:int=100):
        
        super().__init__()
        self.title = 'Copying task TEST'
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        
        self.nStimuli = nStimuli
        self.allImages = images        
        self.imageWidth = imageWidth
        self.nrow = nrow
        self.ncol = ncol

        self.sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.sizePolicy.setHeightForWidth(True)
        
        self.visibleTime = visibleTime
        self.occludedTime = occludedTime
        self.spacePushed = False

        self.dragStartTime = None
        self.dragStartPos = None

        # self.stylestr = "background-color:rgba(255, 255, 255, 0)" # doet niks
        # self.stylestr = "background-color:rgb(128, 128, 128)" # wordt steeds donkerder
        self.styleStr = "background-color:transparent"

        self.currentTrial = 0
        self.copiedImages = pd.DataFrame(columns=['x', 'y', 'Name', 'shouldBe', 'Correct', 'Time', 'dragDuration', 'dragDistance', 'Trial'])

        self.initUI()

    def updateTimer(self):
        shouldUpdate = False
        now = round(time.time() * 1000)
        
        if self.exampleGridBox.isVisible():
            if now - self.start >= self.visibleTime:
                shouldUpdate = True
        else:
            if now - self.start >= self.occludedTime:
                shouldUpdate = True
        
        if shouldUpdate:
            self.showHideExampleGrid()
            self.checkIfFinished()
            # print(f'Update took {round(time.time() * 1000) - self.start}ms')
            self.start = now
        
    def runTimer(self):
        timer = QTimer(self)
        timer.setInterval(1)
        timer.timeout.connect(self.updateTimer)
        
        self.start = round(time.time() * 1000)
        timer.start()
    
    def clearScreen(self):
        for i in reversed(range(self.layout.count())): 
            self.layout.itemAt(i).widget().setParent(None)

    def checkIfFinished(self):
        copiedTemp = self.copiedImages.loc[self.copiedImages['Trial'] == self.currentTrial]
        allCorrect = np.all(copiedTemp['Correct'].values)

        if len(copiedTemp) > 0 and allCorrect:
            print(f'All correct: {allCorrect}')
            print(copiedTemp)

            self.copiedImages.to_csv('results/placements.csv')

            self.clearScreen()
            self.initOpeningScreen()
    
    def showHideExampleGrid(self):
        if self.exampleGridBox.isVisible():
            self.exampleGridBox.setVisible(False)
        else:
            self.exampleGridBox.setVisible(True)
    
    def eventFilter(self, widget, e):
        # print('Event filtered')
        if e.type() == QtCore.QEvent.KeyPress:
            key = e.key()
            if key == QtCore.Qt.Key_Space and not self.spacePushed:
                print('Space')
                
                # Set spacePushed to true and remove all widgets
                self.spacePushed = True
                self.clearScreen()

                # Start the task
                self.initTask()
                return True

        return QWidget.eventFilter(self, widget, e)
        
    def initUI(self):
        print('Starting UI')
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setStyleSheet("background-color:rgb(128, 128, 128)")
        self.layout = QVBoxLayout()

        self.initOpeningScreen()
    
    def initOpeningScreen(self):
        print('Starting opening')
        self.spacePushed = False
        self.currentTrial += 1
        
        self.label = QLabel("Press space to start")
        self.label.setAlignment(Qt.AlignCenter | Qt.AlignHCenter)
        self.installEventFilter(self)
        
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        
        self.show()

    def initTask(self):
        print('Starting task')
        self.removeEventFilter(self)
        
        self.images = pick_stimuli(self.allImages, self.nStimuli)
        self.grid = example_grid.generate_grid(self.images, 
                                               self.nrow, self.ncol)

        # Create the example grid
        self.createMasterGrid()
        
        # self.layout = QVBoxLayout()
        self.layout.addWidget(self.masterGrid)
        self.setLayout(self.layout)
        
        self.show()
        self.runTimer()
        
    def createMasterGrid(self):
        self.masterGrid = QGroupBox("Grid", self)
        layout = QGridLayout()

        self.emptyGridLayout()
        layout.addWidget(self.emptyGridBox, 0, 0)
        layout.addWidget(self.emptyGridBox, 0, 1)
        layout.addWidget(self.emptyGridBox, 0, 2)
        
        self.exampleGridLayout()
        layout.addWidget(self.exampleGridBox, 1, 0)

        layout.addWidget(self.emptyGridBox, 1, 1)
        
        self.copyGridLayout()
        layout.addWidget(self.copyGridBox, 1, 2)
        
        self.resourceGridLayout()
        layout.addWidget(self.resourceGridBox, 2, 2) #, 1, 3)
        
        self.masterGrid.setLayout(layout)
        self.masterGrid.setTitle('')
        self.masterGrid.setStyleSheet(self.styleStr)
            
    def emptyGridLayout(self):
        self.emptyGridBox = QGroupBox("Grid", self)
        layout = QGridLayout()

        self.emptyGridBox.setLayout(layout)
        self.emptyGridBox.setTitle('')
        self.emptyGridBox.setSizePolicy(self.sizePolicy)
        self.emptyGridBox.setStyleSheet(self.styleStr)
    
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

                    exampleDict = pd.DataFrame({
                        'x': x,
                        'y': y,
                        'Name': '',
                        'shouldBe': image.name,
                        'Correct': False,
                        'Time': None,
                        'dragDuration': None,
                        'dragDistance': None,
                        'Trial': self.currentTrial
                    }, index=[0])
                    self.copiedImages = self.copiedImages.append(exampleDict, ignore_index=True)
                    
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
                label = CustomLabel('', self, x, y, self.currentTrial)
                label.setFrameStyle(QFrame.Panel)
                label.resize(self.imageWidth, self.imageWidth)
                label.setAlignment(QtCore.Qt.AlignCenter)
                layout.addWidget(label, x, y)
        
        self.copyGridBox.setLayout(layout)
        self.copyGridBox.setTitle('')
        self.copyGridBox.setSizePolicy(self.sizePolicy)
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
                    label.setAlignment(QtCore.Qt.AlignCenter)
                    
                    if i % 3 == 0:
                        row += 1
                        col = 0

                    layout.addWidget(label, row, col)
                    i += 1
                    col += 1
        
        self.resourceGridBox.setLayout(layout)
        self.resourceGridBox.setTitle('')
        self.resourceGridBox.setSizePolicy(self.sizePolicy)
        # self.resourceGridBox.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.resourceGridBox.setStyleSheet(self.styleStr)
        # self.resourceGridBox.setWindowFlags(QtCore.Qt.FramelessWindowHint)
