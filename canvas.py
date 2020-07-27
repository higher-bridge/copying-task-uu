"""
Created on Wed Feb 26 19:10:04 2020
Author: Alex Hoogerbrugge (@higher-bridge)
"""

import json
import time
from random import shuffle, gauss

import numpy as np
import pandas as pd
from pathlib import Path
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QCursor
from PyQt5.QtWidgets import (QFrame, QGridLayout, QGroupBox, QLabel,
                             QSizePolicy, QVBoxLayout, QWidget)

import example_grid
from stimulus import pick_stimuli
from custom_functions import customTimer, CustomLabel, DraggableLabel


class Canvas(QWidget):
    def __init__(self, images:list, nStimuli:int, imageWidth:int, nrow:int, ncol:int,
                 conditions:list, conditionOrder:list, nTrials:int, 
                 useCustomTimer:bool=False, addNoise=True,
                 left:int=50, top:int=50, width:int=1920, height:int=1080):
        
        super().__init__()
        # Set window params
        self.title = 'Copying task TEST'
        self.left = left
        self.top = top
        self.width = width * 1.5
        self.height = height * 1.5

        self.sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.sizePolicy.setHeightForWidth(True)
        self.setSizePolicy(self.sizePolicy)
        self.styleStr = "background-color:transparent"
        
        # Set stimuli params
        self.nStimuli = nStimuli
        self.allImages = images        
        self.imageWidth = imageWidth
        self.nrow = nrow
        self.ncol = ncol
        
        # Set experiment params
        self.nTrials = nTrials
        self.conditions = conditions
        self.nConditions = len(conditions)
        self.conditionOrder = conditionOrder
        
        self.currentTrial = 0
        self.conditionOrderIndex = 0
        self.currentConditionIndex = self.conditionOrder[self.conditionOrderIndex]

        self.useCustomTimer = useCustomTimer
        self.visibleTime = 1000
        self.occludedTime = 0
        self.addNoise = addNoise
        self.spacePushed = False

        # Set tracking vars
        self.timer = QTimer(self)
        self.mouse = QCursor()
        self.dragStartTime = None
        self.dragStartPos = None
        
        # Init tracking dataframes
        self.copiedImages = pd.DataFrame(columns=['x', 'y', 'Name', 'shouldBe', 
                                                  'Correct', 'Time', 
                                                  'dragDuration', 'dragDistance', 
                                                  'Trial', 'Condition', 'visibleTime'])
        self.mouseTracker = pd.DataFrame(columns=['x', 'y', 'Time'])

        
        self.ppNumber = None
        self.setParticipantNumber()


        self.inOpeningScreen = True
        self.initUI()

    # =============================================================================
    # TRACKING FUNCTIONS    
    # =============================================================================
    def setParticipantNumber(self):
        # Get input
        number = input('Enter participant number or name:\n')
        
        # If input is none, try again (recursively)
        if len(number) < 1 or number == None:
            print('Invalid input, try again')
            self.setParticipantNumber()
        
        # Read already used numbers
        with open('results/usedNumbers.txt', 'r') as f:
            usedNumbers = json.load(f)
        
        # If not in use, save and return
        if number not in usedNumbers:
            self.ppNumber = number
            usedNumbers.append(number)
            
            with open('results/usedNumbers.txt', 'w') as f:
                json.dump(usedNumbers, f)
            
            return
        
        # If already in use, recursively run again
        else:
            print(f'{number} is already in use! Use another number or name')
            self.setParticipantNumber()
                
            
            
    
    def writeCursorPosition(self):
        e = self.mouse.pos()
        movementDF = pd.DataFrame({
            'x': e.x(),
            'y': e.y(),
            'Time': time.time()
        }, index=[0])

        self.mouseTracker = self.mouseTracker.append(movementDF, ignore_index=True)
    
    def updateTimer(self):
        # print(self.inOpeningScreen)
        if self.inOpeningScreen:
          return
        
        self.writeCursorPosition()
        now = round(time.time() * 1000)
        
        shouldUpdate = False
        
        if self.useCustomTimer:
            shouldUpdate = customTimer(self, now)
        else:            
            if self.exampleGridBox.isVisible():
                if now - self.start >= self.visibleTime:
                    shouldUpdate = True
            else:
                if now - self.start >= self.occludedTime:
                    shouldUpdate = True
        
        if shouldUpdate:
            self.showHideExampleGrid()
            self.start = now
            
            self.checkIfFinished()
            
            # print(f'Update took {round(time.time() * 1000) - now}ms')
        
    def runTimer(self):
        self.timer.setInterval(1)
        self.timer.timeout.connect(self.updateTimer)
        
        self.start = round(time.time() * 1000)
        self.timer.start()
        
    def disconnectTimer(self):
        try:
            self.timer.stop()
        except AttributeError:
            pass # Happens if the timer has never been started yet

    def checkIfFinished(self):
        copiedTemp = self.copiedImages.loc[self.copiedImages['Trial'] == self.currentTrial]
        copiedTemp = copiedTemp.loc[copiedTemp['Condition'] == self.currentConditionIndex]
        allCorrect = np.all(copiedTemp['Correct'].values)

        # print(copiedTemp)

        if len(copiedTemp) > 0 and allCorrect:
            print(f'All correct: {allCorrect}')
            print(copiedTemp)

            self.copiedImages.to_csv(Path(f'results/{self.ppNumber}-stimulusPlacements.csv'))
            self.mouseTracker.to_csv(Path(f'results/{self.ppNumber}-mouseTracking-trial{self.currentTrial}-condition{self.currentConditionIndex}.csv'))
            self.mouseTracker = pd.DataFrame(columns=['x', 'y', 'Time'])

            self.clearScreen()
            self.initOpeningScreen()
    
    def showHideExampleGrid(self):
        # print('Updating grid')
        if self.exampleGridBox.isVisible():
            self.exampleGridBox.setVisible(False)
        else:
            self.exampleGridBox.setVisible(True)
    
    def eventFilter(self, widget, e):
        if e.type() == QtCore.QEvent.KeyPress:
            key = e.key()
            if key == QtCore.Qt.Key_Space and not self.spacePushed:
                # print('Space')
                
                # Set spacePushed to true and remove all widgets
                self.spacePushed = True
                self.clearScreen()

                # Start the task
                self.initTask()
                return True

        return QWidget.eventFilter(self, widget, e)

    def setConditionTiming(self):
        # Try to retrieve condition timing. If index out of range (IndexError), conditions are exhausted.
        # visibleTime and occludedTime are pulled from conditions, and assigned to class after mutation has been done
        try:
            visibleTime, occludedTime = self.getConditionTiming()
        except IndexError:
            pass
            #TODO implement breaking out of the program
        
        if self.addNoise:
            if occludedTime != 0:
                sumDuration = visibleTime + occludedTime

                # Generating a noise and its invert keeps the sum duration the same as without permutation
                noise = gauss(mu=1.0, sigma=0.05)
                
                self.visibleTime = int(self.visibleTime * noise)
                self.occludedTime = sumDuration - self.visibleTime
        else:
            self.visibleTime = visibleTime
            self.occludedTime = occludedTime
        
        print(f'Moving to condition {self.currentConditionIndex}: ({self.visibleTime}, {self.occludedTime})')
    
    def getConditionTiming(self):
        # conditionOrderIndex to retrieve a condition number from conditionOrder
        self.currentConditionIndex = self.conditionOrder[self.conditionOrderIndex]
        
        # Use the condition number retrieved from conditionOrder to retrieve the actual condition to use
        visibleTime = self.conditions[self.currentConditionIndex][0]
        occludedTime = self.conditions[self.currentConditionIndex][1]

        return visibleTime, occludedTime
        
    # =============================================================================
    #     INITIALIZATION OF SCREENS
    # =============================================================================
    def clearScreen(self):
        for i in reversed(range(self.layout.count())): 
            self.layout.itemAt(i).widget().setParent(None)
    
    def initUI(self):
        # print('Starting UI')
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setStyleSheet("background-color:rgb(128, 128, 128)")
        self.layout = QVBoxLayout()

        self.initOpeningScreen()
    
    def initOpeningScreen(self):
        # print('Starting opening')
        self.disconnectTimer()
        self.inOpeningScreen = True
        
        # If all trials are done, increment condition counter and 
        # reset trial counter to 0
        if self.currentTrial >= self.nTrials:
            self.conditionOrderIndex += 1
            self.currentTrial = 0
        
        self.spacePushed = False
        self.currentTrial += 1
        
        if self.currentTrial > 1:
            self.label = QLabel("End of trial. Press space to continue")
        else:
            self.label = QLabel("Press space to start")
        
        self.label.setAlignment(Qt.AlignCenter | Qt.AlignHCenter)
        self.installEventFilter(self)
        
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        
        self.show()

    def initTask(self):
        # print('Starting task')
        self.removeEventFilter(self)
        
        self.images = pick_stimuli(self.allImages, self.nStimuli)
        self.grid = example_grid.generate_grid(self.images, 
                                               self.nrow, self.ncol)
        self.setConditionTiming() # Set slightly different timing for each trial

        # Create the example grid
        self.createMasterGrid()
        
        self.layout.addWidget(self.masterGrid)
        self.setLayout(self.layout)
        
        self.show()
        
        self.inOpeningScreen = False
        self.runTimer()
        
    # =============================================================================
    #    GENERATE GRIDS     
    # =============================================================================
    def createMasterGrid(self):
        self.masterGrid = QGroupBox("Grid", self)
        layout = QGridLayout()
        
        
        masterGridRows = 3
        masterGridCols = 6
        gridLocs = [(1, 1), (1,4), (2, 4)]
        
        self.emptyGridLayout()
        for row in range(masterGridRows):
            for col in range(masterGridCols):
                if (row, col) not in gridLocs:
                    layout.addWidget(self.emptyGridBox, row, col)

        self.exampleGridLayout()
        layout.addWidget(self.exampleGridBox, gridLocs[0][0], gridLocs[0][1])
        
        self.copyGridLayout()
        layout.addWidget(self.copyGridBox, gridLocs[1][0], gridLocs[1][1])
        
        self.resourceGridLayout()
        layout.addWidget(self.resourceGridBox, gridLocs[2][0], gridLocs[2][1])
        
        self.masterGrid.setLayout(layout)
        self.masterGrid.setTitle('')
        self.masterGrid.setStyleSheet(self.styleStr)# + "; border:0px")
            
    def emptyGridLayout(self):
        self.emptyGridBox = QGroupBox("Grid", self)
        layout = QGridLayout()

        self.emptyGridBox.setLayout(layout)
        self.emptyGridBox.setTitle('')
        # self.emptyGridBox.setSizePolicy(self.sizePolicy)
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

                    exampleDict = pd.DataFrame({
                        'x': x,
                        'y': y,
                        'Name': '',
                        'shouldBe': image.name,
                        'Correct': False,
                        'Time': None,
                        'dragDuration': None,
                        'dragDistance': None,
                        'Trial': self.currentTrial,
                        'Condition': self.currentConditionIndex,
                        'visibleTime': self.visibleTime
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
                label = CustomLabel('', self, x, y, self.currentTrial, self.currentConditionIndex)
                label.setFrameStyle(QFrame.Panel)
                label.resize(self.imageWidth, self.imageWidth)
                label.setAlignment(QtCore.Qt.AlignCenter)
                label.setSizePolicy(self.sizePolicy)
                layout.addWidget(label, x, y)
        
        self.copyGridBox.setLayout(layout)
        self.copyGridBox.setTitle('')
        # self.copyGridBox.setSizePolicy(self.sizePolicy)
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
                    label.setSizePolicy(self.sizePolicy)
                    
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
