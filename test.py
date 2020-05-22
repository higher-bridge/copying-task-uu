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

NUM_STIMULI = 6
IMAGE_WIDTH = 50
ROW_COL_NUM = 3


if __name__ == '__main__':
    path = f'{os.getcwd()}/stimuli/'
    images = pick_stimuli(path, NUM_STIMULI, IMAGE_WIDTH, extension='.jpg')
        
    app = QApplication(sys.argv)
    ex = Canvas(images, IMAGE_WIDTH, ROW_COL_NUM, ROW_COL_NUM, width=800, height=800)
    sys.exit(app.exec_())

