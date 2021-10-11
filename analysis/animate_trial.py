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
from pathlib import Path

import constants_analysis as constants
import helperfunctions as hf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from celluloid import Camera
from matplotlib.patches import Rectangle

ID = '003'
condition = 1
trial = 3

### Load mouse tracking data
mousedata_files = sorted(
    [f for f in hf.getListOfFiles('../results') if (f'/{ID}-' in f) and (f'-mouseTracking-condition{condition}' in f)])
mouse_data = pd.concat([pd.read_csv(f) for f in mousedata_files], ignore_index=True)
mouse_data = hf.locate_trial(mouse_data, condition, trial)
# print(mouse_data.head())

### Load all samples
all_gaze_data = pd.read_csv(f'../results/{ID}/{ID}-allSamples.csv')
gaze_data = hf.locate_trial(all_gaze_data, condition, trial)
gx, gy = list(gaze_data['gx_right']), list(gaze_data['gy_right'])

### Load all events
events = pd.read_csv(f'../results/{ID}/{ID}-allEvents.csv')
events = hf.locate_trial(events, condition, trial)

### Load all placement info
placements = pd.read_csv(f'../results/{ID}/{ID}-allCorrectPlacements.csv')
# placements = pd.read_excel(f'../results/{ID}/{ID}-allCorrectPlacements.xlsx')
placements = hf.locate_trial(placements, condition, trial)

# Temporary, cameFromX/Y was not recorded in earlier experiment versions. These are manually specified for 003-0-3
placements['cameFromX'] = [1926, 2070, 1911, 2031, 1761, 1800]
placements['cameFromY'] = [1100, 1259, 1293, 1093, 1111, 1273]

placements = placements.sort_values(by='Time', ignore_index=True)

### Find trial start and finish
start_row = events.loc[events['Event'] == 'Task init']
start_time = start_row['TrackerTime'].values[0]

end_row = events.loc[events['Event'] == 'Finished trial']
end_time = end_row['TrackerTime'].values[0] - start_time
print(f'Trial duration: {end_time} ms')

time_diff = start_row['TimeDiff'].values[0]

### Subtract the start time from gaze and mouse data (therefore: start at 0 ms)
gaze_data['time'] = gaze_data['time'].apply(lambda x: int(x - start_time))
mouse_data['TrackerTime'] = mouse_data['TrackerTime'].apply(lambda x: int(x - start_time))
placements['TrackerTime'] = placements['Time'].apply(lambda x: int(x - time_diff - start_time))
events['ZeroTime'] = events['TrackerTime'].apply(lambda x: int(x - start_time))

### Retrieve when grid was shown or hidden
show_hide_grid = []
is_showing = 'Grid'
for i in range(int(end_time)):
    row = events.loc[events['ZeroTime'] == i]
    if len(row) > 0:
        if row['Event'].values[0] == 'Showing grid':
            is_showing = 'Grid'
        elif row['Event'].values[0] == 'Hiding grid':
            is_showing = 'None'
        elif row['Event'].values[0] == 'Showing hourglass':
            is_showing = 'Hourglass'
        else:
            pass

    show_hide_grid.append(is_showing)

### Convert example stimuli to annotationBoxes (needs x/y)
example_stimuli = hf.prepare_stimuli(list(placements['shouldBe']),
                                     list(placements['x']), list(placements['y']),
                                     constants.all_example_locations)

### Find when stimuli were placed in workspace grid and convert them to annotationBoxes
workspace_placed = dict()
for i, row in placements.iterrows():
    timestamp = row['TrackerTime']

    box = hf.prepare_stimuli([row['shouldBe']],
                             [row['x']],
                             [row['y']],
                             constants.all_workspace_locations)

    workspace_placed[timestamp] = box[0]

### Convert resource grid stimuli to annotationBoxes
resource_stimuli = hf.prepare_stimuli(list(placements['shouldBe']),
                                      list(placements['cameFromX']),
                                      list(placements['cameFromY']),
                                      constants.all_resource_locations,
                                      in_pixels=True)

### Convert hourglass.png to annotationBox
hourglass = hf.prepare_stimuli([Path('pictograms/hourglass.png')],
                               [650], [720],
                               [(650, 720)],
                               in_pixels=True, zoom=.03)
hourglass = hourglass[0]

# Set vars
framerate = 30
dpi = 200
patch_width = 130
patch_height = 140

start = time.time()
fig = plt.figure(figsize=(8, 4.5), dpi=dpi)
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor((.5, .5, .5))

camera = Camera(fig)

# TODO: add brief opening screen

for t in np.arange(len(gaze_data), step=1000 / framerate):
    # TODO: Stimulus is placed in wrong resource grid location sometimes. Might be related to manual cameFromX/Y spec
    # TODO: add dragging motion from resource grid to workspace grid

    ### PLOT EMPTY WORKSPACE GRID ###
    for loc in constants.all_workspace_locations:  # Draw empty grids
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

    ### PLOT STIMULI IN RESOURCE BOX ###
    [ax.add_artist(ab) for ab in resource_stimuli]

    ### PLOT STIMULI, HOURGLASS, OR NOTHING
    if show_hide_grid[int(t)] == 'Grid':
        # Plot empty example grid
        for loc in constants.all_example_locations:  # Draw empty grids
            l = (loc[0] - (patch_width / 2),
                 loc[1] - (patch_height / 2))
            ax.add_patch(Rectangle(l, patch_width, patch_height,
                                   edgecolor='black', facecolor='none',
                                   linewidth=.8))

        # Plot example stimuli
        [ax.add_artist(ab) for ab in example_stimuli]
    elif show_hide_grid[int(t)] == 'Hourglass':
        # Plot hourglass
        ax.add_artist(hourglass)
    else:
        pass

    ### PLOT CORRECTLY PLACED ITEMS
    for k in list(workspace_placed.keys()):
        if t >= k:
            ax.add_artist(workspace_placed[k])

    ### PLOT MOUSE AND GAZE ###
    x, y = gx[int(t)], gy[int(t)]

    m_idx = hf.find_nearest_index(mouse_data['TrackerTime'], t)
    md = mouse_data.iloc[m_idx]
    mx, my = md['x'], md['y']

    gaze, = plt.plot(x, y, 'bo', zorder=50)  # zorder makes sure gaze and mouse are drawn on top of everything else
    mouse, = plt.plot(mx, my, 'r+', zorder=51)

    ### DO FORMATTING ###
    plt.xlim((0, constants.RESOLUTION[0]))
    plt.ylim((constants.RESOLUTION[1], 0))

    plt.xticks([])
    plt.yticks([])

    ### Show legend with x/y coords for gaze and mouse. zfill(4) so text is same length in each frame
    x, y = str(round(x)).zfill(4), str(round(y)).zfill(4)
    mx, my = str(mx).zfill(4), str(my).zfill(4)
    plt.legend([gaze, mouse], [f'Gaze ({x}, {y})', f'Mouse ({mx}, {my})'])

    camera.snap()

# TODO: add brief post-trial screen

animation = camera.animate()
animation.save(f'../results/plots/animation-pp{ID}-c{condition}-t{trial}.mp4', fps=framerate)
# animation.save(f'../results/plots/animation-{ID}-{condition}-{trial}.gif', fps=framerate)

print(f'Animating took {round(time.time() - start, 1)} seconds, {round((time.time() - start) / 60, 1)} minutes')
