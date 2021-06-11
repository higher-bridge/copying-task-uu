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

import time
from pathlib import Path, PureWindowsPath

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from celluloid import Camera

import helperfunctions as hf
import constants_analysis as constants

ID = '003'
condition = 0
trial = 3

# Load mouse tracking data
mousedata_files = sorted(
    [f for f in hf.getListOfFiles('../results') if (f'/{ID}-' in f) and (f'-mouseTracking-condition{condition}' in f)])
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

# Load all placement info
placements = pd.read_csv(f'../results/{ID}/{ID}-allCorrectPlacements.csv')
placements = placements.loc[placements['Condition'] == condition]
placements = placements.loc[placements['Trial'] == trial]

stimulus_paths = ['../' + PureWindowsPath(x).as_posix() for x in list(placements['shouldBe'])]
stimuli = [mpimg.imread(p) for p in stimulus_paths]
image_boxes = [OffsetImage(s, zoom=.1) for s in stimuli]

annotation_boxes = [AnnotationBbox(im, constants.all_example_locations[x + y * 3], frameon=False) for x, y, im in
                    zip(list(placements['x']),
                        list(placements['y']),
                        image_boxes)]

# stimulus_locations = [(constants.all_example_locations[x + y * 3],
#                        stim,
#                        im_box) for x, y, stim, im_box in zip(list(placements['x']),
#                                                              list(placements['y']),
#                                                              stimuli,
#                                                              image_boxes)]

# Find trial start and finish
start_row = events.loc[events['Event'] == 'Task init']
start_time = start_row['TrackerTime'].values[0]
print(start_time)

end_row = events.loc[events['Event'] == 'Finished trial']
end_time = end_row['TrackerTime'].values[0] - start_time
print(end_time)

time_diff = start_row['TimeDiff'].values[0]
print(time_diff)

# Subtract the start time from gaze and mouse data (therefore: start at 0 ms)
gaze_data['time'] = gaze_data['time'].apply(lambda x: int(x - start_time))
mouse_data['TrackerTime'] = mouse_data['TrackerTime'].apply(lambda x: int(x - start_time))
print(list(gaze_data['time'])[:10], len(gaze_data))

# 'gx_right' 'gy_right' 'time'

gx, gy = list(gaze_data['gx_right']), list(gaze_data['gy_right'])

framerate = 30
patch_width = 130
patch_height = 140

last_mouse_loc = (0, 0)

start = time.time()
fig = plt.figure(figsize=(8, 4.5), dpi=400)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor((.5, .5, .5))

camera = Camera(fig)

for t in np.arange(len(gaze_data), step=1000 / framerate):
    # TODO: show stimuli in resource grid
    # TODO: add dragging motion from resource grid to workspace grid

    ### PLOT EMPTY GRIDS ###
    all_patch_locations = constants.all_example_locations + constants.all_workspace_locations
    for loc in all_patch_locations:  # Draw empty grids
        l = (loc[0] - (patch_width / 2),
             loc[1] - (patch_height / 2))
        ax.add_patch(Rectangle(l, patch_width, patch_height,
                               edgecolor='black', facecolor='none',
                               linewidth=.8))

    ### PLOT EMPTY RESOURCE BOX ###
    ax.add_patch(Rectangle(constants.resource_grid_corner,  # Draw empty resource box
                           constants.resource_grid_size[0], constants.resource_grid_size[1],
                           edgecolor='black', facecolor='none',
                           linewidth=.8))

    ### PLOT EXAMPLE GRID STIMULI ###
    [ax.add_artist(ab) for ab in annotation_boxes]
    # for (loc, im, box) in stimulus_locations:
    #     ab = AnnotationBbox(box, loc, frameon=False)
    #     ax.add_artist(ab)

    ### PLOT MOUSE AND GAZE ###
    x, y = gx[int(t)], gy[int(t)]

    m_idx = hf.find_nearest_index(mouse_data['TrackerTime'], t)
    md = mouse_data.iloc[m_idx]
    mx, my = md['x'], md['y']

    gaze, = plt.plot(x, y, 'bo')
    mouse, = plt.plot(mx, my, 'r+')

    ### DO FORMATTING ###
    plt.xlim((0, constants.RESOLUTION[0]))
    plt.ylim((constants.RESOLUTION[1], 0))
    plt.legend([gaze, mouse], ['gaze', 'mouse'])

    camera.snap()

animation = camera.animate()
animation.save(f'../results/plots/animation-{ID}-{condition}-{trial}.mp4', fps=framerate)
# animation.save(f'../results/plots/animation-{ID}-{condition}-{trial}.gif', fps=framerate)

print(f'Animating took {round(time.time() - start, 1)} seconds')
