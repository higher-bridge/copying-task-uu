#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 20:03:30 2020

@author: mba
"""

from PyQt5.QtWidgets import QApplication
import os
import sys
from stimulus import pick_stimuli
from canvas import Canvas

IMAGE_WIDTH = 100

if __name__ == '__main__':
    path = f'{os.getcwd()}/images/'
    images = pick_stimuli(path, 6, IMAGE_WIDTH)
    # grid = generate_grid(images, 3, 3)
        
    app = QApplication(sys.argv)
    ex = Canvas(images, IMAGE_WIDTH, 4, 4, width=1000, height=800)
    sys.exit(app.exec_())

