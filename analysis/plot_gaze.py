#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:46:31 2020

@author: alexos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def find_nearest_index(array, timestamp):
    array = np.asarray(array)
    idx = (np.abs(array - timestamp)).argmin()
    return idx

filename = 'xytest'

# Load data
samples_loc = f'../results/{filename}-trackingSession-1-samples.csv'
events_loc = f'../results/{filename}-trackingSession-1-events.csv'

samples = pd.read_csv(samples_loc)
events = pd.read_csv(events_loc)

# Filter only fixations
fixations = events.loc[events['type'] == 'fixation']

# Plot all fixations
plt.figure()
sns.scatterplot('gavx', 'gavy', data=fixations)
plt.xlim((0, 2560))
plt.ylim((1440, 0)) # Note that the y-axis needs to be flipped
plt.title('All trials')
plt.show()

# Get start & end times of first trial
task_events = pd.read_csv(f'../results/{filename}-eventTracking.csv')

start_times = task_events.loc[task_events['Event'] == 'Task init']['TrackerTime']
end_times = task_events.loc[task_events['Event'] == 'Finished trial']['TrackerTime']

for trial_num in range(len(start_times)):

    start = list(start_times)[trial_num]
    end = list(end_times)[trial_num]
    
    start_idx = find_nearest_index(fixations['start'], start)
    end_idx = find_nearest_index(fixations['end'], end)
    
    fixations_trial = fixations.iloc[start_idx:end_idx]
    
    # Plot
    plt.figure()
    sns.scatterplot('gavx', 'gavy', data=fixations_trial)
    plt.xlim((0, 2560))
    plt.ylim((1440, 0)) # Note that the y-axis needs to be flipped
    plt.title(f'Trial {trial_num + 1}')
    plt.show()
