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
from celluloid import Camera

import helperfunctions as hf
import constants_analysis as constants

ID = '003'
condition = 0
trial = 3

# Load mouse tracking data
mousedata_files = sorted([f for f in hf.getListOfFiles('../results') if (f'/{ID}-' in f) and (f'-mouseTracking-condition{condition}' in f)])
mouse_data = pd.concat([pd.read_csv(f) for f in mousedata_files], ignore_index=True)
mouse_data = mouse_data.loc[mouse_data['Condition'] == condition]
mouse_data = mouse_data.loc[mouse_data['Trial'] == trial]
# print(mouse_data.head())

# Load all samples
all_gaze_data = pd.read_csv(f'../results/{ID}/{ID}-allSamples.csv')
gaze_data = all_gaze_data.loc[all_gaze_data['Condition'] == condition]
gaze_data = gaze_data.loc[gaze_data['Trial'] == trial]
# print(gaze_data.head())

# Load all events
events = pd.read_csv(f'../results/{ID}/{ID}-allEvents.csv')
events = events.loc[events['Condition'] == condition]
events = events.loc[events['Trial'] == trial]
# print(events.head())

# Find trial start and finish
start_row = events.loc[events['Event'] == 'Task init']
start_time = start_row['TrackerTime'].values[0]
print(start_time)

end_row = events.loc[events['Event'] == 'Finished trial']
end_time = end_row['TrackerTime'].values[0] - start_time
print(end_time)

# Subtract the start time from gaze and mouse data (therefore: start at 0 ms)
gaze_data['time'] = gaze_data['time'].apply(lambda x: int(x - start_time))
mouse_data['TrackerTime'] = mouse_data['TrackerTime'].apply(lambda x: int(x - start_time))
print(list(gaze_data['time'])[:10], len(gaze_data))

# 'gx_right' 'gy_right' 'time'

framerate = 60

fig = plt.figure(figsize=(7.5, 5), dpi=600)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor((.5, .5, .5))

camera = Camera(fig)

last_mouse_loc = (0, 0)

for t in np.arange(len(gaze_data), step=1000/framerate):
    # TODO: show stimuli in example grid and resource grid
    # TODO: add dragging motion from resource grid to workspace grid

    gd = gaze_data.iloc[int(t)]
    x, y = gd['gx_right'], gd['gy_right']

    m_idx = hf.find_nearest_index(mouse_data['TrackerTime'], t)
    md = mouse_data.iloc[m_idx]
    mx, my = md['x'], md['y']

    gaze, = plt.plot(x, y, 'bo')
    mouse, = plt.plot(mx, my, 'r+')
    plt.xlim((0, constants.RESOLUTION[0]))
    plt.ylim((constants.RESOLUTION[1], 0))

    plt.legend([gaze, mouse], ['gaze', 'mouse'])

    camera.snap()

animation = camera.animate()
animation.save(f'../results/plots/animation-{ID}-{condition}-{trial}.mp4', fps=framerate)

