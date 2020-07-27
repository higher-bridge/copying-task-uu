# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:16:56 2020

@author: pytest
"""

from psychopy.iohub import launchHubServer
from psychopy.core import getTime, wait


iohub_config = {'eyetracker.hw.sr_research.eyelink.EyeTracker':
                {'name': 'tracker',
                 'model_name': 'EYELINK 1000 DESKTOP',
                 'runtime_settings': {'sampling_rate': 1000,
                                      'track_eyes': 'RIGHT'}
                 }
                }
io = launchHubServer(**iohub_config)

# Get the eye tracker device.
tracker = io.devices.tracker

# run eyetracker calibration
r = tracker.runSetupProcedure()


# Check for and print any eye tracker events received...
tracker.setRecordingState(True)

stime = getTime()
while getTime()-stime < 2.0:
    for e in tracker.getEvents():
        print(e)
        
        
# Check for and print current eye position every 100 msec.
stime = getTime()
while getTime()-stime < 5.0:
    print(tracker.getPosition())
    wait(0.1)

tracker.setRecordingState(False)

# Stop the ioHub Server
io.quit()