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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from celluloid import Camera

import helperfunctions as hf

ID = '003'
condition = 0
trial = 3

mousedata_files = sorted([f for f in hf.getListOfFiles('../results') if (f'/{ID}-' in f) and (f'-mouseTracking-condition{condition}' in f)])
mouse_data = pd.concat([pd.read_csv(f) for f in mousedata_files], ignore_index=True)

mouse_data = mouse_data.loc[mouse_data['Condition'] == condition]
mouse_data = mouse_data.loc[mouse_data['Trial'] == trial]
# print(mouse_data.head())

all_gaze_data = pd.read_csv(f'../results/{ID}/{ID}-allFixations.csv')
gaze_data = all_gaze_data.loc[all_gaze_data['Condition'] == condition]
gaze_data = gaze_data.loc[gaze_data['Trial'] == trial]
# print(gaze_data.head())

events = pd.read_csv(f'../results/{ID}/{ID}-allEvents.csv')
events = events.loc[events['Condition'] == condition]
events = events.loc[events['Trial'] == trial]
# print(events.head())

start_row = events.loc[events['Event'] == 'Task init']
start_time = start_row['TrackerTime'].values[0]
# print(start_time)

print(list(mouse_data['TrackerTime'])[:10])

gaze_data['start'] = gaze_data['start'].apply(lambda x: x - start_time)
gaze_data['end'] = gaze_data['end'].apply(lambda x: x - start_time)
mouse_data['TrackerTime'] = mouse_data['TrackerTime'].apply(lambda x: x - start_time)

print(list(mouse_data['TrackerTime'])[:10])



