# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:16:56 2020

@author: pytest
"""

from pygaze import libscreen
from pygaze.eyetracker import EyeTracker
import time
from constants import *


disp = libscreen.Display()

tracker = EyeTracker(disp) #resolution=(2560, 1440), screensize=(59.8, 33.6)
tracker.calibrate()

disp.close()

start = time.time()
tracker.start_recording()


print(tracker._get_eyelink_clock_async(time.time()))

while time.time() - start < 20:
    if round(time.time() - start, 2) % 1 == 0:
        # print(tracker.sample())
        # print(tracker._get_eyelink_clock_async(time.time()))
        tracker.log('time {round(time.time() * 1000)}')
        pass
        
tracker.stop_recording()

tracker.close()
