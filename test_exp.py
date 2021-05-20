from pygaze import libscreen
from pygaze import eyetracker

import time

disp = libscreen.Display()
tracker = eyetracker.EyeTracker(disp, trackertype='dummy')

fixscreen = libscreen.Screen()
fixscreen.draw_fixation()

tracker.calibrate()

start = time.time()

disp.fill(fixscreen)
disp.show()

while (time.time() - start) < 5:
    print(round(time.time(), 4), tracker.sample())
    time.sleep(0.01)

tracker.stop_recording()
tracker.close()
disp.close()
