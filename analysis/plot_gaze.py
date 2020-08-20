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

# Load data
samples_loc = '../results/001-trackingSession-1-samples.csv'
events_loc = '../results/001-trackingSession-1-events.csv'

samples = pd.read_csv(samples_loc)
events = pd.read_csv(events_loc)

# Filter only fixations
fixations = events.loc[events['type'] == 'fixation']

# Get start & end times of first trial
task_events = pd.read_csv('../results/001-eventTracking.csv')

start_times = task_events.loc[task_events['Event'] == 'Task init']['Time']
start = list(start_times)[0]
start_idx = find_nearest_index(fixations['start'], start)

end_times = task_events.loc[task_events['Event'] == 'Finished trial']['Time']
end = list(end_times)[0]
end_idx = find_nearest_index(fixations['start'], end)


plt.figure()
sns.scatterplot('gavx', 'gavy', data=fixations)
plt.show()
