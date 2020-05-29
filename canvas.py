#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 19:10:04 2020
Author: Alex Hoogerbrugge (@higher-bridge)
"""

import time
from random import shuffle

import pandas as pd
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QFrame, QGridLayout, QGroupBox, QLabel,
                             QSizePolicy, QVBoxLayout, QWidget)

import example_grid
from custom_labels import CustomLabel, DraggableLabel


class Canvas(QWidget):
    # Generates a canvas with a Copygrid and an Examplegrid
    def __init__(self, images:list, image_width:int, nrow:int, ncol:int, 
                 left:int=10, top:int=10, width:int=640, height:int=480,
                 visible_time:int=1000, occluded_time:int=500):
        
        super().__init__()
        self.title = 'Copying task TEST'
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        
        self.images = images
        self.image_width = image_width
        self.nrow = nrow
        self.ncol = ncol
        
        self.visible_time = visible_time
        self.occluded_time = occluded_time

        # self.stylestr = "background-color:rgba(255, 255, 255, 0)" # doet niks
        # self.stylestr = "background-color:rgb(128, 128, 128)" # wordt steeds donkerder
        self.stylestr = "background-color:transparent"
        
        self.grid = example_grid.generate_grid(self.images, 
                                               self.nrow, self.ncol)
        
        self.sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.sizePolicy.setHeightForWidth(True)
        
        self.initUI()
        self.run_timer()

                
    def update_timer(self):
        should_update = False
        now = round(time.time() * 1000)
        
        if self.exampleGridBox.isVisible():
            if now - self.start >= self.visible_time:
                should_update = True
        else:
            if now - self.start >= self.occluded_time:
                should_update = True
        
        if should_update:
            self.showHideExampleGrid()
            print(f'Update took {now - self.start}ms')
            self.start = now
        

    def run_timer(self):
        timer = QTimer(self)
        timer.setInterval(2)
        timer.timeout.connect(self.update_timer)
        
        self.start = round(time.time() * 1000)
        timer.start()
        
    def showHideExampleGrid(self):
        if self.exampleGridBox.isVisible():
            self.exampleGridBox.setVisible(False)
        else:
            self.exampleGridBox.setVisible(True)
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setStyleSheet("background-color:rgb(128, 128, 128)")

        # Create the example grid
        self.createMasterGrid()
        
        windowLayout = QVBoxLayout()
        windowLayout.addWidget(self.masterGrid)
        self.setLayout(windowLayout)
        
        self.show()
        
        
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
        self.masterGrid.setStyleSheet(self.stylestr)
            
    def emptyGridLayout(self):
        self.emptyGridBox = QGroupBox("Grid", self)
        layout = QGridLayout()

        self.emptyGridBox.setLayout(layout)
        self.emptyGridBox.setTitle('')
        self.emptyGridBox.setSizePolicy(self.sizePolicy)
        self.emptyGridBox.setStyleSheet(self.stylestr)
    
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
                    i += 1

                layout.addWidget(label, x, y)

        self.exampleGridBox.setLayout(layout)
        self.exampleGridBox.setTitle('')
        self.exampleGridBox.setSizePolicy(self.sizePolicy)
        self.exampleGridBox.setStyleSheet(self.stylestr)
        
    def copyGridLayout(self):
        self.copyGridBox = QGroupBox("Grid", self)
        layout = QGridLayout()
                
        for x in range(self.nrow):
            for y in range(self.ncol):
                label = CustomLabel('', self)
                label.setFrameStyle(QFrame.Panel)
                label.resize(self.image_width, self.image_width)
                label.setAlignment(QtCore.Qt.AlignCenter)
                layout.addWidget(label, x, y)
        
        
        self.copyGridBox.setLayout(layout)
        self.copyGridBox.setTitle('')
        self.copyGridBox.setSizePolicy(self.sizePolicy)
        self.copyGridBox.setStyleSheet(self.stylestr)
        
    def resourceGridLayout(self):
        self.resourceGridBox = QGroupBox("Grid", self)
        layout = QGridLayout()
        
        shuffled_images = self.images
        shuffle(shuffled_images)

        i = 0
        row = 0
        col = 0
        for x in range(self.nrow):
            for y in range(self.ncol):
                if self.grid[x, y]:
                    # label = CustomLabel('', self)
                    image = shuffled_images[i]
                    label = DraggableLabel(self, image.qimage)
                    # pixmap = QPixmap.fromImage(image.qimage)
                    # label.setPixmap(pixmap)
                    label.setAlignment(QtCore.Qt.AlignCenter)
                    # label.setStyleSheet(self.stylestr)
                    
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
        self.resourceGridBox.setStyleSheet(self.stylestr)
        # self.resourceGridBox.setWindowFlags(QtCore.Qt.FramelessWindowHint)
