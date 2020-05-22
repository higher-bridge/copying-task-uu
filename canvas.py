#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 19:10:04 2020
Author: Alex Hoogerbrugge (@higher-bridge)
"""

from PyQt5.QtWidgets import QWidget, QLabel, QGroupBox, QGridLayout, QVBoxLayout, QFrame, QSizePolicy
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore
import example_grid
from random import shuffle


class Canvas(QWidget):
    # Generates a canvas with a Copygrid and an Examplegrid
    def __init__(self, images:list, image_width:int, nrow:int, ncol:int, 
                 left:int=10, top:int=10, width:int=640, height:int=480):
        
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

        self.stylestr = "background-color:rgb(128, 128, 128)"
        
        self.grid = example_grid.generate_grid(self.images, 
                                               self.nrow, self.ncol)
        
        self.sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.sizePolicy.setHeightForWidth(True)
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setStyleSheet(self.stylestr)

        # Create the example grid
        self.createMasterGrid()
        
        windowLayout = QVBoxLayout()
        windowLayout.addWidget(self.horizontalGroupBox)
        self.setLayout(windowLayout)
        
        # TODO: Create the copy grid
        # TODO: Create the draggable images

        self.show()
        
    def createMasterGrid(self):
        self.horizontalGroupBox = QGroupBox("Grid")
        self.horizontalGroupBox.setStyleSheet(self.stylestr)
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
        
        self.horizontalGroupBox.setLayout(layout)
        self.horizontalGroupBox.setTitle('')
            
    def emptyGridLayout(self):
        self.emptyGridBox = QGroupBox("Grid")
        self.emptyGridBox.setStyleSheet(self.stylestr)
        layout = QGridLayout()

        self.emptyGridBox.setLayout(layout)
        self.emptyGridBox.setTitle('')
        self.emptyGridBox.setSizePolicy(self.sizePolicy)
    
    def exampleGridLayout(self):
        self.exampleGridBox = QGroupBox("Grid")
        self.exampleGridBox.setStyleSheet(self.stylestr)
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
        
    def copyGridLayout(self):
        self.copyGridBox = QGroupBox("Grid")
        self.copyGridBox.setStyleSheet(self.stylestr)
        layout = QGridLayout()
                
        for x in range(self.nrow):
            for y in range(self.ncol):
                label = QLabel(self)
                label.setFrameStyle(QFrame.Panel)
                label.resize(self.image_width, self.image_width)
                layout.addWidget(label, x, y)
        
        
        self.copyGridBox.setLayout(layout)
        self.copyGridBox.setTitle('')
        self.copyGridBox.setSizePolicy(self.sizePolicy)
        
    def resourceGridLayout(self):
        self.resourceGridBox = QGroupBox("Grid")
        self.resourceGridBox.setStyleSheet(self.stylestr)
        layout = QGridLayout()
        
        shuffled_images = self.images
        shuffle(shuffled_images)

        i = 0
        row = 0
        col = 0
        for x in range(self.nrow):
            for y in range(self.ncol):
                if self.grid[x, y]:
                    label = QLabel(self)
                    image = shuffled_images[i]
                    pixmap = QPixmap.fromImage(image.qimage)
                    label.setPixmap(pixmap)
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




    
