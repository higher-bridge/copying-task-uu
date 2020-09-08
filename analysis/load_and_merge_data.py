#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:46:31 2020

@author: alexos
"""

import pandas as pd
import numpy as np
import os
import sys

import helperfunctions as hf
import mouse_analysis


PLOT = False

base_location = '../results'
all_IDs = sorted([f for f in os.listdir(base_location) if not '.' in f]) # Ignores actual files, only finds folders

ID_dict = hf.write_IDs_to_dict(all_IDs)
    
for i, ID in enumerate(ID_dict.keys()):
    task_event_files = hf.find_files(ID, ID_dict[ID], base_location, '-eventTracking.csv')
    eventfiles = hf.find_files(ID, ID_dict[ID], base_location, '-events.csv')
    mousefiles = hf.find_files(ID, ID_dict[ID], base_location, '-mouseTracking')
    
    # Concatenate all separate session files into one larger file
    task_events = hf.concat_event_files(task_event_files)
    events = hf.concat_event_files(eventfiles)  
    mousedata = hf.concat_event_files(mousefiles) #.drop('Unnamed: 0')
    
    # Filter only fixations
    fixations = events.loc[events['type'] == 'fixation']
    
    # Plot all fixations
    # hf.scatterplot_fixations(fixations, 'gavx', 'gavy', title=f'{ID}: All trials')

    # Create two empty lists
    trial_list = np.empty(len(events))
    trial_list[:] = np.nan
    
    condition_list = np.empty(len(events))
    condition_list[:] = np.nan
    
    mouse_valid = np.zeros(len(mousedata), dtype=bool)
    mouse_valid[:] = False
    
    # For each condition, each trial, get the start and end times and relate it to eye events
    for condition in list(task_events['Condition'].unique()):
        condition_df = task_events.loc[task_events['Condition'] == condition]
        
        for trial_num in list(condition_df['Trial'].unique()):
            trial_df = condition_df.loc[condition_df['Trial'] == trial_num]
            
            # Retrieve where eventTracking has written trial init and trial finish
            start_times = trial_df.loc[trial_df['Event'] == 'Task init']['TrackerTime']
            start = list(start_times)[0]
            
            end_times = trial_df.loc[trial_df['Event'] == 'Finished trial']['TrackerTime']
            end = list(end_times)[0]
            
            # Match the timestamp to the closest timestamp in the fixations df
            start_idx = hf.find_nearest_index(fixations['start'], start)
            end_idx = hf.find_nearest_index(fixations['end'], end)
            
            mouse_start_idx = hf.find_nearest_index(mousedata['TrackerTime'], start)
            mouse_end_idx = hf.find_nearest_index(mousedata['TrackerTime'], end)
            
            if start_idx != end_idx: 
                # Set everything between start and end of trial with that condition/trial
                trial_list[start_idx:end_idx] = trial_num
                condition_list[start_idx:end_idx] = condition
                mouse_valid[mouse_start_idx:mouse_end_idx] = True
                
                # Plot
                if PLOT:
                    fixations_trial = fixations.iloc[start_idx:end_idx]
                    hf.scatterplot_fixations(fixations_trial, 'gavx', 'gavy', title=f'ID {ID}, condition {condition}, trial {trial_num + 1}')
    
    
    # Append trial/condition info to eye events df            
    events['Trial'] = trial_list
    events['Condition'] = condition_list
    events.to_csv(f'../results/{ID}/{ID}-allFixations.csv')
    
    mousedata['Valid'] = mouse_valid
    mousedata = mousedata.loc[mousedata['Valid'] == True]
    
    mouse_fixations = mouse_analysis.get_fixation_events(list(mousedata['x']), list(mousedata['y']),
                                                          list(mousedata['TrackerTime']))
    mouse_fixations.to_csv(f'../results/{ID}/{ID}-mouseFixations.csv')
    
    print(f'Parsed {i + 1} of {len(ID_dict.keys())} files')
    sys.stdout.write("\033[F")
    
    
    
    
