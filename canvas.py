#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 19:10:04 2020
Author: Alex Hoogerbrugge (@higher-bridge)
"""

from PyQt5.QtWidgets import QWidget, QLabel, QGroupBox, QGridLayout, QVBoxLayout, QFrame, QSizePolicy
from PyQt5.QtGui import QPixmap
import example_grid


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
        
        self.grid = example_grid.generate_grid(self.images, 
                                               self.nrow, self.ncol)
        
        self.sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.sizePolicy.setHeightForWidth(True)
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

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
        
        layout = QGridLayout()
        
        self.exampleGridLayout()
        layout.addWidget(self.exampleGridBox, 0, 0)
        
        self.copyGridLayout()
        layout.addWidget(self.copyGridBox, 0, 1)
        
        # pass
        # layout.addWidget(label, 1, 0)
        
        self.resourceGridLayout()
        layout.addWidget(self.resourceGridBox, 2, 0, 1, 2)
        
        self.horizontalGroupBox.setLayout(layout)
        self.horizontalGroupBox.setTitle('')
            
    def exampleGridLayout(self):
        self.exampleGridBox = QGroupBox("Grid")
        layout = QGridLayout()
        
        i = 0
        for x in range(self.nrow):
            for y in range(self.ncol):
                if self.grid[x, y]:
                    label = QLabel(self)
                    label.setFrameStyle(QFrame.Panel)
                    image = self.images[i]
                    pixmap = QPixmap.fromImage(image.qimage)
                    label.setPixmap(pixmap)
                    layout.addWidget(label, x, y)
                    i += 1
          
        self.exampleGridBox.setLayout(layout)
        self.exampleGridBox.setTitle('')
        self.exampleGridBox.setSizePolicy(self.sizePolicy)
        
    def copyGridLayout(self):
        self.copyGridBox = QGroupBox("Grid")
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
        layout = QGridLayout()
        
        i = 0
        for x in range(self.nrow):
            for y in range(self.ncol):
                if self.grid[x, y]:
                    label = QLabel(self)
                    image = self.images[i]
                    pixmap = QPixmap.fromImage(image.qimage)
                    label.setPixmap(pixmap)
                    layout.addWidget(label, 0, y)
                    i += 1
          
        self.resourceGridBox.setLayout(layout)
        self.resourceGridBox.setTitle('')
        self.resourceGridBox.setSizePolicy(self.sizePolicy)




    
