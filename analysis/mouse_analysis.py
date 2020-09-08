#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 13:52:14 2020

@author: alexos
"""

import pandas as pd
import numpy as np
from math import sqrt

def euclidean_distance(vector1, vector2):
    dist = [(a - b)**2 for a, b in zip(vector1, vector2)]
    dist = sqrt(sum(dist))
    return dist

class MouseFixations():
    def __init__(self):
        self.fix_dict = self.init_fixation_dict()

    def init_fixation_dict(self):
        keys = ['start_x', 'start_y', 'end_x', 'end_y', 'dist', 'velocity',
                'avx', 'avy', 'start', 'end', 'duration']
        
        fix_dict = {key: [] for key in keys}
        return fix_dict
           
    def add_to_fixation_dict(self, start_location:tuple, end_location:tuple, start, end):
        dist = euclidean_distance(start_location, end_location)
        dur = end - start
        velocity = dist / dur
        
        self.fix_dict['start_x'].append(start_location[0])
        self.fix_dict['start_y'].append(start_location[1])
        
        self.fix_dict['end_x'].append(end_location[0])
        self.fix_dict['end_y'].append(end_location[1])
        
        self.fix_dict['dist'].append(dist)
        self.fix_dict['velocity'].append(velocity)
        
        self.fix_dict['avx'].append((start_location[0] + end_location[0]) / 2)
        self.fix_dict['avy'].append((start_location[1] + end_location[1]) / 2)
        
        self.fix_dict['start'].append(start)
        self.fix_dict['end'].append(end)
        self.fix_dict['duration'].append(dur)
        
    def get_fix_dict(self, astype:str='dataframe'):
        if astype == 'dataframe':
            return pd.DataFrame(self.fix_dict)
        elif astype == 'dict':
            return self.fix_dict
        else:
            print(f'Could not convert fix_dict to {astype}. Try dataframe or dict.')
            return None

        
# class MouseSaccade():
#     def __init__(self, start_location:tuple, end_location:tuple, start, end):
#         self.start_x = start_location[0]
#         self.start_y = start_location[1]
        
#         self.end_x = end_location[0]
#         self.end_y = end_location[1]
        
#         self.start = start
#         self.end = end
#         self.duration = end - start
        
        
def get_fixation_events(xdata:list, ydata:list, timedata:list, max_deviation:int=10, 
                        min_duration:int=80, max_duration:int=10000):
    mf = MouseFixations()
    i = 0
    
    while i < len(timedata):
        # Set the start variables for this possible fixation        
        start_location = (xdata[i], ydata[i])
        start_time = timedata[i]
        
        # We assume a fixation just to get in the while loop. If distance is too great,
        # it will break out immediately
        fixating = True
        time_lim_reached = False
        
        while fixating:
            try:
                # Take the next data point
                i += 1
                new_location = (xdata[i], ydata[i])
                new_time = timedata[i]
                
                # Calculate distance between now and start loc, if too great, break out of while loop
                if euclidean_distance(start_location, new_location) > max_deviation:
                    fixating = False
                
                # Min duration must have been reached to count as fixation
                if new_time - start_time > min_duration:
                    time_lim_reached = True
            
            except IndexError: # i becomes too large
                fixating = False
                time_lim_reached = True
            
            # If minimum time reached and no longer fixating, write
            if not fixating and time_lim_reached:
                mf.add_to_fixation_dict(start_location, (xdata[i-1], ydata[i-1]), 
                                        start_time, timedata[i-1])
    
    events = mf.get_fix_dict()
    events = events.loc[events['duration'] < max_duration] # Remove all that is longer than max_duration            
    
    return events
                
                
                
            
        
        
        
        