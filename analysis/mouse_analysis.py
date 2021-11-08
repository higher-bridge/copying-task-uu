"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from math import sqrt

import numpy as np
import pandas as pd


def euclidean_distance(vector1, vector2):
    dist = [(a - b)**2 for a, b in zip(vector1, vector2)]
    dist = sqrt(sum(dist))
    return dist

class MouseEvents():
    def __init__(self):
        self.event_dict = self.init_event_dict()

    def init_event_dict(self):
        keys = ['type', 'start_x', 'start_y', 'end_x', 'end_y', 
                'dist', 'velocity', 'peak_velocity',
                'avg_xpos', 'avg_ypos', 
                'start', 'end', 'duration',
                'Trial', 'Condition', 'Session', 'Dragging',
                'indices']
        
        event_dict = {key: [] for key in keys}
        return event_dict
           
    def add_to_event_dict(self, kind:str, start_location:tuple, end_location:tuple,
                             dist:float, start:int, end:int, trial:int, condition:int, session:int,
                             dragging:bool,
                             velocity:float=np.nan, peak_velocity:float=np.nan,
                             indices=[]):        
        self.event_dict['type'].append(kind)
        self.event_dict['start_x'].append(start_location[0])
        self.event_dict['start_y'].append(start_location[1])
        
        self.event_dict['end_x'].append(end_location[0])
        self.event_dict['end_y'].append(end_location[1])
        
        self.event_dict['dist'].append(dist)
        self.event_dict['velocity'].append(velocity)
        self.event_dict['peak_velocity'].append(peak_velocity)
        
        self.event_dict['avg_xpos'].append((start_location[0] + end_location[0]) / 2)
        self.event_dict['avg_ypos'].append((start_location[1] + end_location[1]) / 2)
        
        self.event_dict['start'].append(start)
        self.event_dict['end'].append(end)
        self.event_dict['duration'].append(end - start)
        self.event_dict['indices'].append(indices)
        
        self.event_dict['Trial'].append(trial)
        self.event_dict['Condition'].append(condition)
        self.event_dict['Session'].append(session)
        self.event_dict['Dragging'].append(dragging)
        
    def get_event_dict(self, astype:str='dataframe'):
        if astype == 'dataframe':
            df = pd.DataFrame(self.event_dict)
            df = df.drop('indices', axis=1)
            df = df.sort_values(by=['start'], ignore_index=True, kind='mergesort')
            return df
        elif astype == 'dict':
            return self.event_dict
        else:
            print(f'Could not convert fix_dict to {astype}. Try dataframe or dict.')
            return None
        
        
def _get_fixation_events(me, xdata:list, ydata:list, timedata:list, trials:list, conditions:list, sessions:list,
                         dragging:list,
                         max_deviation:int, 
                         min_duration:int, max_duration:int):

    xdata = np.array(xdata)
    ydata = np.array(ydata)
    timedata = np.array(timedata)
    
    i = 0
    while i < len(timedata):
        start_i = i
        
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
                dist = euclidean_distance(start_location, new_location)
                if dist > max_deviation:
                    fixating = False
                
                # Min duration must have been reached to count as fixation
                if new_time - start_time > min_duration:
                    time_lim_reached = True
            
            except IndexError: # i becomes too large
                fixating = False
                time_lim_reached = True
            
            # If minimum time reached and no longer fixating, write
            if not fixating and time_lim_reached:
                end_location = (xdata[i-1], ydata[i-1])
                me.add_to_event_dict(kind='fixation', 
                                     start_location=start_location, 
                                     end_location=end_location,
                                     dist=euclidean_distance(start_location, end_location),
                                     start=start_time, 
                                     end=timedata[i-1],
                                     trial=trials[i-1],
                                     condition=conditions[i-1],
                                     session=sessions[i-1],
                                     dragging=dragging[i-1],
                                     indices=np.arange(start_i, i - 1))
    
    return me
                
def _get_saccade_events(me, xdata:list, ydata:list, timedata:list, trials:list, conditions:list, sessions:list,
                        dragging:list,
                        min_deviation:int,
                        min_duration:int, max_duration:int):

    xdata = np.array(xdata)
    ydata = np.array(ydata)
    timedata = np.array(timedata)
    
    # Discard all data in xdata, ydata and timedata where a fixation was measured
    valid = np.empty(len(timedata), dtype=bool)
    valid[:] = True
    
    fix_dict = me.get_event_dict('dict')

    indices = fix_dict['indices']
    for ind in indices:
        valid[ind] = False

    i = 0
    while i < len(timedata):
        velocities = []
        distances = []
        
        start_time = timedata[i]
        end_time = timedata[i]
        
        start_location = (xdata[i], ydata[i])
        end_location  = (xdata[i], ydata[i])
        
        saccade_measured = False
        val = valid[i]        
        while val:
            x = xdata[i]
            y = ydata[i]
            t = timedata[i]
            
            timediff = t - timedata[i - 1]
            distance = euclidean_distance((xdata[i - 1], ydata[i - 1]), (x, y))
            distances.append(distance)
            
            velocity = distance / (timediff / 1000) if timediff > 0 else 0.01
            velocities.append(velocity)
            
            end_location = (x, y)
            end_time = t
            
            trial = trials[i]
            condition = conditions[i]
            session = sessions[i]
            drag = dragging[i]

            saccade_measured = True
            
            try:
                i += 1
                val = valid[i]
            except IndexError:
                val = False
        
        if saccade_measured:
            # print('Saccade measured')
            total_distance = sum(distances)
            if total_distance > min_deviation:
                # print('Distance within limits')
                if ((end_time - start_time) > min_duration) and ((end_time - start_time) < max_duration):
                    # print('Duration within limits')
                    me.add_to_event_dict(kind='saccade', 
                                         start_location=start_location, 
                                         end_location=end_location,
                                         trial=trial,
                                         condition=condition,
                                         session=session,
                                         dragging=drag,
                                         dist=total_distance,
                                         velocity=np.mean(velocities),
                                         peak_velocity=max(velocities),
                                         start=start_time, 
                                         end=end_time)
                
        i += 1 # If we can't get in the while loop, i += 1
    
    
    return me
                
            
        
def get_mouse_events(xdata:list, ydata:list, timedata:list, trials:list, conditions:list, sessions:list,
                     dragging:list,
                     fix_max_deviation:int=10, 
                     fix_min_duration:int=80, fix_max_duration:int=5000,
                     sac_min_duration:int=5, sac_max_duration:int=3000):
    
    me = MouseEvents()
    
    me = _get_fixation_events(me, xdata, ydata, timedata, trials, conditions, sessions, dragging,
                             fix_max_deviation, fix_min_duration, fix_max_duration)
    
    me = _get_saccade_events(me, xdata, ydata, timedata, trials, conditions, sessions, dragging,
                            fix_max_deviation, sac_min_duration, sac_max_duration)
    
    events = me.get_event_dict()
    
    return events
        
        