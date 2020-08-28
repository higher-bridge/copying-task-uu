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

filename = '001-2'
trackingsession = 2
PLOT = False


# Load samples
# samples_loc = f'../results/{filename}/{filename}-trackingSession-{trackingsession}-samples.csv'
# samples = pd.read_csv(samples_loc)

# Load events
events_loc = f'../results/{filename}/{filename}-trackingSession-{trackingsession}-events.csv'
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
task_events = pd.read_csv(f'../results/{filename}/{filename}-eventTracking.csv')

trial_list = np.empty(len(events))
trial_list[:] = np.nan

condition_list = np.empty(len(events))
condition_list[:] = np.nan

for condition in list(task_events['Condition'].unique()):
    condition_df = task_events.loc[task_events['Condition'] == condition]
    
    for trial_num in list(condition_df['Trial'].unique()):
        trial_df = condition_df.loc[condition_df['Trial'] == trial_num]
        
        start_times = trial_df.loc[trial_df['Event'] == 'Task init']['TrackerTime']
        end_times = trial_df.loc[trial_df['Event'] == 'Finished trial']['TrackerTime']
        
        start = list(start_times)[0]
        end = list(end_times)[0]
        
        start_idx = find_nearest_index(fixations['start'], start)
        end_idx = find_nearest_index(fixations['end'], end)
        
        if start_idx != end_idx:    
            fixations_trial = fixations.iloc[start_idx:end_idx]
            
            trial_list[start_idx:end_idx] = trial_num
            condition_list[start_idx:end_idx] = condition
            
            # Plot
            if PLOT:
                plt.figure()
                sns.scatterplot('gavx', 'gavy', data=fixations_trial)
                plt.xlim((0, 2560))
                plt.ylim((1440, 0)) # Note that the y-axis needs to be flipped
                plt.title(f'Condition {condition}, trial {trial_num + 1}')
                plt.show()

            
events['Trial'] = trial_list
events['Condition'] = condition_list
events.to_csv(events_loc)
