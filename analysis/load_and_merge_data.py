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
from constants import base_location, MIDLINE


PLOT = False

all_IDs = sorted([f for f in os.listdir(base_location) if not '.' in f]) # Ignores actual files, only finds folders

ID_dict = hf.write_IDs_to_dict(all_IDs)
pp_info = pd.read_excel('../results/participant_info.xlsx')
pp_info['ID'] = [str(x).zfill(3) for x in list(pp_info['ID'])]


trials_b1 = []
trials_b2 = []
trials_b3 = []
trials_b4 = []
    
for i, ID in enumerate(list(pp_info['ID'].unique())):     
    task_event_files = hf.find_files(ID, ID_dict[ID], base_location, '-eventTracking.csv')
    eventfiles = hf.find_files(ID, ID_dict[ID], base_location, '-events.csv')
    mousefiles = hf.find_files(ID, ID_dict[ID], base_location, '-mouseTracking-')
    placements = hf.find_files(ID, ID_dict[ID], base_location, '-allPlacements.csv')
    c_placements = hf.find_files(ID, ID_dict[ID], base_location, '-correctPlacements.csv')
    
    # Concatenate all separate session files into one larger file
    task_events = hf.concat_event_files(task_event_files)
    events = hf.concat_event_files(eventfiles)  
    # mousedata = hf.concat_event_files(mousefiles) #.drop('Unnamed: 0')
    all_placements = hf.concat_event_files(placements)
    correct_placements = hf.concat_event_files(c_placements)
    
    # Order the df in the proper condition, retrieved from participant_info.xlsx
    # (I didn't design the mousetracker filenames with chronological ordering)
    condition_order = hf.get_condition_order(pp_info, ID)
    # mousedata = hf.order_by_condition(mousedata, condition_order)
    
    # Filter only fixations
    fixations = events.loc[events['type'] == 'fixation']
    
    # Plot all fixations
    if PLOT:
        hf.scatterplot_fixations(fixations, 'gavx', 'gavy', title=f'{ID}: All trials',
                                 savestr=f'../results/{ID}/{ID}-fixationPlotAll.png')

    # Create two empty lists in which we will fill in the appropriate trials/condition value
    trial_list = np.empty(len(events), dtype=int)
    trial_list[:] = 999
    
    condition_list = np.empty(len(events), dtype=int)
    condition_list[:] = 999
    
    dragging_list = np.empty(len(events), dtype=bool)
    dragging_list[:] = False
    
    # Create a list to track which rows in mousedata are valid
    # mouse_valid = np.zeros(len(mousedata), dtype=bool)
    # mouse_valid[:] = False
    
    # For each condition, each trial, get the start and end times and relate it to eye events
    for x, condition in enumerate(list(task_events['Condition'].unique())):
        condition_df = task_events.loc[task_events['Condition'] == condition]
        correct_condition = correct_placements.loc[correct_placements['Condition'] == condition]
        
        all_trials = list(condition_df['Trial'].unique())
        for trial_num in all_trials:
            trial_df = condition_df.loc[condition_df['Trial'] == trial_num]
            correct_trial = correct_condition.loc[correct_condition['Trial'] == trial_num]
            
            # Retrieve where eventTracking has written trial init and trial finish            
            start_times = trial_df.loc[trial_df['Event'] == 'Task init']['TrackerTime']
            start = list(start_times)[0]
            
            end_times = trial_df.loc[trial_df['Event'] == 'Finished trial']['TrackerTime']
            end = list(end_times)[0]
            
            # Match the timestamp to the closest timestamp in the fixations df
            start_idx = hf.find_nearest_index(events['end'], start)
            end_idx = hf.find_nearest_index(events['start'], end)
            
            # Find during which events items were being dragged with the mouse
            task_init = trial_df.loc[trial_df['Event'] == 'Task init']
            timediff = task_init['TimeDiff'].values[0]
            
            for index, row in correct_trial.iterrows():
                end_drag = row['Time'] - timediff
                start_drag = end_drag - row['dragDuration']
                start_drag_idx = hf.find_nearest_index(events['start'], start_drag)
                end_drag_idx = hf.find_nearest_index(events['end'], end_drag)
                
                dragging_list[start_drag_idx:end_drag_idx] = True
            
            # mouse_start_idx = hf.find_nearest_index(mousedata['TrackerTime'], start)
            # mouse_end_idx = hf.find_nearest_index(mousedata['TrackerTime'], end)
            
            if start_idx != end_idx: 
                # Set everything between start and end of trial with that condition/trial
                trial_list[start_idx:end_idx] = trial_num
                condition_list[start_idx:end_idx] = condition
                # mouse_valid[mouse_start_idx:mouse_end_idx] = True
                
                # Plot
                if PLOT:
                    events_trial = events.iloc[start_idx:end_idx]
                    fixations_trial = events_trial.loc[events_trial['type'] == 'fixation']
                    num_crossings = hf.get_midline_crossings(list(fixations_trial['gavx']), midline=MIDLINE)
                    hf.scatterplot_fixations(fixations_trial, 'gavx', 'gavy', 
                                             title=f'ID {ID}, condition {condition}, trial {trial_num}, crossings={num_crossings}',
                                             plot_line=False,
                                             save=False,
                                             savestr=f'../results/{ID}/{ID}-fixationPlot-{condition}-{trial_num}.png')
        
    # Append trial/condition info to eye events df            
    events['Trial'] = trial_list
    events['Condition'] = condition_list
    events['Dragging'] = dragging_list
    
    num_trials = hf.get_num_trials(events)
    trials_b1.append(int(num_trials[0]))
    trials_b2.append(int(num_trials[1]))
    trials_b3.append(int(num_trials[2]))
    trials_b4.append(int(num_trials[3]))
    
    
    # mousedata['Valid'] = mouse_valid
    # mousedata = mousedata.loc[mousedata['Valid'] == True]
    
    # mouse_fixations = mouse_analysis.get_fixation_events(list(mousedata['x']), list(mousedata['y']),
    #                                                       list(mousedata['TrackerTime']))
    
    # mouse_fixations.to_csv(f'../results/{ID}/{ID}-mouseFixations.csv')
    events.to_csv(f'../results/{ID}/{ID}-allFixations.csv')
    task_events.to_csv(f'../results/{ID}/{ID}-allEvents.csv')
    all_placements.to_csv(f'../results/{ID}/{ID}-allAllPlacements.csv')
    correct_placements.to_csv(f'../results/{ID}/{ID}-allCorrectPlacements.csv')
    
    print(f'Parsed {i + 1} of {len(ID_dict.keys())} files')
    sys.stdout.write("\033[F")
        
pp_info['Trials condition 0'] = trials_b1
pp_info['Trials condition 1'] = trials_b2
pp_info['Trials condition 2'] = trials_b3
pp_info['Trials condition 3'] = trials_b4

pp_info.to_excel('../results/participant_info.xlsx')
    
  
    
