"""
Created on Wed Feb 26 19:10:04 2020
Author: Alex Hoogerbrugge (@higher-bridge)
"""

import json
import time
from random import shuffle, gauss
import os

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QCursor, QFont
from PyQt5.QtWidgets import (QFrame, QGridLayout, QGroupBox, QLabel,
                             QSizePolicy, QVBoxLayout, QWidget)

from pygaze import libscreen
from pygaze.eyetracker import EyeTracker

import example_grid
from stimulus import pick_stimuli
from custom_functions import customTimer, CustomLabel, DraggableLabel


class Canvas(QWidget):
    def __init__(self, images:list, nStimuli:int, imageWidth:int, nrow:int, ncol:int,
                 conditions:list, conditionOrder:list, nTrials:int, 
                 useCustomTimer:bool=False, trialTimeOut:int=10000, addNoise=True,
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
        
        self.currentTrial = 1
        self.conditionOrderIndex = 0
        self.currentConditionIndex = self.conditionOrder[self.conditionOrderIndex]

        self.useCustomTimer = useCustomTimer
        self.visibleTime = 0
        self.occludedTime = 0
        self.addNoise = addNoise
        self.trialTimeOut = trialTimeOut

        self.spacePushed = False

        # Set tracking vars
        self.timer = QTimer(self)
        self.mouse = QCursor()
        self.dragStartTime = None
        self.dragStartPos = None
        
        # Track correct placements
        self.correctPlacements = pd.DataFrame(columns=['x', 'y', 'Name', 'shouldBe', 
                                                  'Correct', 'Time', 
                                                  'dragDuration', 'dragDistance', 
                                                  'Trial', 'Condition', 'visibleTime'])
        
        # Track placements including mistakes
        self.allPlacements = pd.DataFrame(columns=['x', 'y', 'Name', 'shouldBe',
                                                   'Correct', 'Time',  
                                                   'Trial', 'Condition', 'visibleTime'])
        
        # self.mouseTracker = pd.DataFrame(columns=['x', 'y', 'Time', 'Trial'])
        self.mouseTrackerDict = {key: [] for key in ['x', 'y', 'Time', 'TrackerTime', 'Trial', 'Condition']}

        self.eventTracker = pd.DataFrame(columns=['Time', 'TrackerTime', 'TimeDiff', 'Event', 'Condition', 'Trial'])

        self.ppNumber = None
        self.setParticipantNumber()        

        # self.disp = libscreen.Display()
        # self.tracker = EyeTracker(self.disp)
        # self.disp.close()
        self.disp = None
        self.tracker = None
        self.recordingSession = 0


        ### TEMP
        self.mouseUpdates = 0

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
                
            # Make dedicated folder for this person
            if not str(self.ppNumber) in os.listdir('results'):
                os.mkdir(f'results/{self.ppNumber}')
                
            return
        
        # If already in use, recursively run again
        else:
            print(f'{number} is already in use! Use another number or name')
            self.setParticipantNumber()
                
    def writeCursorPosition(self):
        e = self.mouse.pos()

        self.mouseUpdates += 1

        self.mouseTrackerDict['x'].append(e.x())
        self.mouseTrackerDict['y'].append(e.y())
        self.mouseTrackerDict['Time'].append(round(time.time() * 1000))
        self.mouseTrackerDict['TrackerTime'].append(self.getTrackerClock())
        self.mouseTrackerDict['Trial'].append(self.currentTrial)
        self.mouseTrackerDict['Condition'].append(self.currentConditionIndex)
        
    
    def updateTimer(self):
        if self.inOpeningScreen:
          return
        
        self.writeCursorPosition()

        now = round(time.time() * 1000)
        shouldUpdate = False

        # Check for timeout
        if now - self.globalTrialStart >= self.trialTimeOut:
            self.checkIfFinished(timeOut=True)
            return

        # Check if update necessary
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

        # Check if result needs checking, every 500ms to avoid too much processing
        if now - self.checkIfFinishedStart >= 500:
            self.checkIfFinished()
            self.checkIfFinishedStart = now
                    
    def runTimer(self):
        self.timer.setInterval(1) # 1 ms
        self.timer.timeout.connect(self.updateTimer)
        
        # Unfortunately we need three tracking vars to keep updates not too time-consuming
        self.start = round(time.time() * 1000)
        self.globalTrialStart = self.start
        self.checkIfFinishedStart = self.start
        
        self.timer.start()
        
    def disconnectTimer(self):
        try:
            self.timer.stop()
        except AttributeError:
            pass # Happens if the timer has never been started yet

    def getTrackerClock(self):
        try:
            trackerClock = self.tracker.get_eyelink_clock()
        except Exception:
            trackerClock = 0
            
        return trackerClock

    def writeEvent(self, msg):
        trackerClock = self.getTrackerClock()
            
        localTime = round(time.time() * 1000)
        timeDiff = localTime - trackerClock
        
        event = pd.DataFrame({'Time': localTime, 'TrackerTime': trackerClock, 
                              'TimeDiff': timeDiff, 'Event': msg,
                              'Condition': self.currentConditionIndex, 'Trial': self.currentTrial}, index=[0])
        self.eventTracker = self.eventTracker.append(event, ignore_index=True)

    def writeFiles(self):
        self.correctPlacements.to_csv(Path(f'results/{self.ppNumber}/{self.ppNumber}-correctPlacements.csv'))
        self.allPlacements.to_csv(Path(f'results/{self.ppNumber}/{self.ppNumber}-allPlacements.csv'))
        
        self.eventTracker.to_csv(Path(f'results/{self.ppNumber}/{self.ppNumber}-eventTracking.csv'))

    def writeMouseTracker(self):
        mouseTrackerDF = pd.DataFrame(self.mouseTrackerDict)
        mouseTrackerDF.to_csv(Path(f'results/{self.ppNumber}/{self.ppNumber}-mouseTracking-condition{self.currentConditionIndex}-trackingSession-{self.recordingSession}.csv'))
        
        # pickle.dump(self.mouseTrackerDict, 
        #             open(Path(f'results/{self.ppNumber}/{self.ppNumber}-mouseTracking-condition{self.currentConditionIndex}.p'),
        #             'wb'))        
        self.mouseTrackerDict = {key: [] for key in ['x', 'y', 'Time', 'TrackerTime', 'Trial', 'Condition']}
        
    def checkIfFinished(self, timeOut=False):        
        copiedTemp = self.correctPlacements.loc[self.correctPlacements['Trial'] == self.currentTrial]
        copiedTemp = copiedTemp.loc[copiedTemp['Condition'] == self.currentConditionIndex]
        
        allCorrect = np.all(copiedTemp['Correct'].values)


        if (len(copiedTemp) > 0 and allCorrect) or timeOut:
            print(f'All correct: {allCorrect}')
            
            self.writeEvent('Finished trial')

            self.clearScreen()
            
            self.writeFiles()
            self.currentTrial += 1
            
            self.initOpeningScreen(timeOut)
    
    def showHideExampleGrid(self):
        if self.exampleGridBox.isVisible():
            text = 'Showing'
            self.exampleGridBox.setVisible(False)
        else:
            text = 'Hiding'
            self.exampleGridBox.setVisible(True)

        self.writeEvent(f'{text} grid')
    
    def moveAndRenameTrackerFile(self):
        fromLocation = 'default.edf'
        toLocation = f'results/{self.ppNumber}/{self.ppNumber}-trackingSession-{self.recordingSession}.edf'
        os.rename(Path(fromLocation), Path(toLocation))
        
        self.writeMouseTracker()
        
        self.writeEvent(f'Writing eyetracker session {self.recordingSession}')

        print(f'Saved session {self.recordingSession} to {toLocation}')

    # def performDriftCorrection(self):
    #     self.clearScreen()
    #     self.showMaximized()

    #     try:
    #         self.tracker.stop_recording()
    #     #     self.tracker.close()
    #     #     self.moveAndRenameTrackerFile()
    #     #     self.disp = None
    #     #     self.tracker = None
    #     except Exception as e:
    #         print(e)
        

        
    #     # self.disp = libscreen.Display()
    #     # self.tracker = EyeTracker(self.disp)
        
    #     corrected = self.tracker.fix_triggered_drift_correction()

    #     event = pd.DataFrame({'Time': time.time(), 'Event': f'Drift correction (result={str(corrected)})',
    #                           'Condition': self.currentConditionIndex, 'Trial': self.currentTrial}, index=[0])
    #     self.eventTracker = self.eventTracker.append(event, ignore_index=True)
        
    #     print(f'Drift correction (result={str(corrected)})')
        
    #     self.disp.close()
    #     self.showFullScreen()
        
    #     self.recordingSession += 1
    #     self.tracker.start_recording()
        
    #     time.sleep(1)
    #     self.initOpeningScreen()
        

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

            elif key == QtCore.Qt.Key_Backspace:
                print('Backspace pressed')
                
                self.clearScreen()
                
                # Try to close the eyetracker
                try:
                    self.tracker.stop_recording()
                    self.tracker.close()
                    self.moveAndRenameTrackerFile()
                    self.disp = None
                    self.tracker = None
                    
                except Exception as e:
                    # There is no recording to stop
                    print(e)


                # Go into calibration
                if self.disp == None:                    
                    # Program crashes if both pygaze and this want to use fullscreen, so maximize instead of FS
                    self.showMaximized()
                    
                    # time.sleep(5)
                    self.disp = libscreen.Display()
                    self.tracker = EyeTracker(self.disp)
    
                    self.tracker.calibrate()
                    self.disp.close()
                    # self.disp = None
                    
                    self.showFullScreen()
    
                    # When done, start recording and init task
                    self.recordingSession += 1
                    self.tracker.start_recording()
                    
                    # Get the async between tracker and os
                    async_val = self.tracker._get_eyelink_clock_async()
                    self.writeEvent(f'Async {async_val}')
                    
                time.sleep(1)
                self.initOpeningScreen()
                
                return True
            
            # Trigger early exit
            elif key == QtCore.Qt.Key_Tab:
                try:
                    self.tracker.stop_recording()
                    self.tracker.close(full_close=True)
                    self.moveAndRenameTrackerFile()
                except Exception as e:
                    print(e)
                                
                self.writeEvent('Early exit')
                
                self.writeFiles()
                
                self.close()
                print('\nEarly exit')
                raise SystemExit(0)

        return QWidget.eventFilter(self, widget, e)

    def setConditionTiming(self):
        # Try to retrieve condition timing. If index out of range (IndexError), conditions are exhausted.
        # visibleTime and occludedTime are pulled from conditions, and assigned to class after mutation has been done
        try:
            visibleTime, occludedTime = self.getConditionTiming()
        except IndexError as e:
            self.tracker.stop_recording()
            self.tracker.close(full_close=True)
            self.moveAndRenameTrackerFile()
            
            self.writeEvent('Finished')
            
            self.writeFiles()
            
            self.close()
            print('\nNo more conditions, the experiment is finished!')
            raise SystemExit(0)
        
        if occludedTime != 0 and self.addNoise:
            sumDuration = visibleTime + occludedTime

            # Generating a noise and its invert keeps the sum duration the same as without permutation
            noise = gauss(mu=1.0, sigma=0.1)
            
            self.visibleTime = int(visibleTime * noise)
            self.occludedTime = sumDuration - visibleTime
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
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setStyleSheet("background-color:rgb(128, 128, 128)")
        self.layout = QVBoxLayout()

        self.writeEvent('UI init',)

        self.initOpeningScreen()
    
    def initOpeningScreen(self, timeOut=False):
        self.disconnectTimer()
        self.inOpeningScreen = True
        
        # If all trials are done, increment condition counter and 
        # reset trial counter to 0. Also reset the mouseTracker df
        if self.currentTrial > self.nTrials:
            self.conditionOrderIndex += 1
            self.currentTrial = 1
        
        self.spacePushed = False

        self.writeEvent('In starting screen')
        
        if self.currentTrial == 1:
            if self.conditionOrderIndex == 0:
                self.label = QLabel(\
"Welcome to the experiment.\n \
Throughout this experiment, you will be asked to copy the layout on the left side of the screen to the right side of the screen,\n \
by dragging the images in the lower right part of the screen to their correct positions. You are asked to do this as quickly \
and as accurately as possible.\n \
If you make a mistake, you can right-click the image to remove it from that location. \n \
Throughout the experiment, the example layout may disappear for brief periods of time. You are asked to keep\n \
performing the task as quickly and as accurately as possible.\n \n \
If you have any questions now or anytime during the experiment, please ask them straightaway.\n \
If you need to take a break, please tell the experimenter so.\n \n \
When you are ready to start the experiment, please tell the experimenter and we will start calibrating.\n \
Good luck!")

            elif self.conditionOrderIndex > 0:
                # self.performDriftCorrection()
                self.label = QLabel(\
f"End of block {self.conditionOrderIndex}. You may now take a break if you wish to do so.\n \
If you wish to carry on immediately, let the experimenter know.\n \
If you have taken a break, please wait for the experimenter to start the calibration procedure.")

        elif self.currentTrial > 1:
            
            addText = '\nNow may be a good time to re-calibrate.' if self.currentTrial % 10 == 0 else ''
            
            if timeOut:
                self.label = QLabel(f"You timed out. Press space to continue to the next trial. {addText}")
            else:
                self.label = QLabel(f"End of trial. Press space to continue to the next trial. {addText}")            

        self.label.setFont(QFont("Times", 12))
        self.label.setAlignment(Qt.AlignCenter | Qt.AlignHCenter)
        self.installEventFilter(self)
        
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        
        self.show()

    def initTask(self):
        self.removeEventFilter(self)
        
        self.images = pick_stimuli(self.allImages, self.nStimuli)
        self.grid = example_grid.generate_grid(self.images, 
                                               self.nrow, self.ncol)
        self.setConditionTiming() # Set slightly different timing for each trial

        # Create the example grid
        self.createMasterGrid()
        
        self.layout.addWidget(self.masterGrid)
        self.setLayout(self.layout)
        
        self.writeEvent('Task init')

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
                    self.correctPlacements = self.correctPlacements.append(exampleDict, ignore_index=True)
                    
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
