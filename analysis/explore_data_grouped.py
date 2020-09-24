#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:47:07 2020

@author: mba
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import helperfunctions as hf
from constants import EXCLUDE_TRIALS, MIDLINE, base_location


pp_info = pd.read_excel('../results/participant_info.xlsx')
pp_info['ID'] = [str(x).zfill(3) for x in list(pp_info['ID'])]

pp_info = hf.remove_from_pp_info(pp_info, [f'Trials condition {i}' for i in range(4)])


fixations_files = [f for f in hf.getListOfFiles(base_location) if '-allFixations.csv' in f]
task_events_files = [f for f in hf.getListOfFiles(base_location) if 'allEvents.csv' in f]
all_placements_files = [f for f in hf.getListOfFiles(base_location) if '-allAllPlacements.csv' in f]
correct_placements_files = [f for f in hf.getListOfFiles(base_location) if '-allCorrectPlacements.csv' in f]

features = ['Completion time (ms)', 
            'Number of crossings', 
            'Number of left-side fixations', 
            'Total dwell time left side (ms)',
            'Completion time minus dwell time']
cols = ['ID', 'Condition', 'Trial']
[cols.append(f) for f in features]
results = pd.DataFrame(columns=cols)

# =============================================================================
# WRITE VARIABLES TO ROW FOR EACH ID, CONDITION AND TRIAL
# =============================================================================
# Data outside of trial start-end are marked with 999 so will be filtered in the next steps
for ID in list(pp_info['ID'].unique()): 
    fix_filenames = [f for f in fixations_files if ID in f]
    fix_filename = fix_filenames[0]
    
    task_filenames = [f for f in task_events_files if ID in f]
    task_filename = task_filenames[0]    
    
    fix_df = pd.read_csv(fix_filename)
    fix_df = fix_df.loc[fix_df['type'] == 'fixation']
    
    task_df = pd.read_csv(task_filename)    
    
    for condition in list(fix_df['Condition'].unique()):
        fix_df_c = fix_df.loc[fix_df['Condition'] == condition]
        task_df_c = task_df.loc[task_df['Condition'] == condition]
        
        for trial in list(fix_df_c['Trial'].unique()): 
            if trial not in EXCLUDE_TRIALS:
                fix_df_t = fix_df_c.loc[fix_df_c['Trial'] == trial]
                task_df_t = task_df_c.loc[task_df_c['Trial'] == trial]

                start_times = task_df_t.loc[task_df_t['Event'] == 'Task init']['TrackerTime']
                start = list(start_times)[0]
                
                end_times = task_df_t.loc[task_df_t['Event'] == 'Finished trial']['TrackerTime']
                end = list(end_times)[0]
                                
                completion_time = end - start
                num_crossings = hf.get_midline_crossings(list(fix_df_t['gavx']), midline=MIDLINE)
                num_fixations = hf.get_left_side_fixations(list(fix_df_t['gavx']), midline=MIDLINE)
                dwell_times = hf.get_dwell_times(list(fix_df_t['gavx']),
                                                   list(fix_df_t['start']),
                                                   list(fix_df_t['end']), 
                                                   midline=MIDLINE)                
                
                r = pd.DataFrame({'ID': ID,
                                  'Condition': int(condition),
                                  'Trial': int(trial),
                                  'Completion time (ms)': float(completion_time),
                                  'Number of crossings': float(num_crossings),
                                  'Number of left-side fixations': float(num_fixations),
                                  'Total dwell time left side (ms)': float(dwell_times),
                                  'Completion time minus dwell time': float(completion_time - dwell_times)},
                                 index=[0])
                results = results.append(r, ignore_index=True)
                
                # hf.scatterplot_fixations(fix_df_t, 'gavx', 'gavy', 
                #                           title=f'ID {ID}, condition {condition}, trial {trial}, crossings={num_crossings}',
                #                           plot_line=True,
                #                           save=False,
                #                           savestr='')
                
results = results.dropna()

# Group by ID and Condition, use median
results_grouped = results.groupby(['ID', 'Condition']).agg({f: ['median'] for f in features}).reset_index()
results_grouped.columns = results_grouped.columns.get_level_values(0)
results_grouped.to_csv(f'{base_location}/results-grouped-ID-condition.csv')

for f in features:
    plt.figure(figsize=(4, 5))
    sns.barplot('Condition', f, data=results_grouped, capsize=.1, errwidth=1.5)
    plt.title(f)
    plt.tight_layout()
    plt.savefig(f'{base_location}/plots/grouped {f} bar.png', dpi=500)
    plt.show()
    
    plt.figure()
    for c in list(results_grouped['Condition'].unique()):
        plot_df = results_grouped.loc[results_grouped['Condition'] == c]
        sns.distplot(plot_df[f], label=c)
    
    plt.title(f)
    plt.legend(title='Condition')
    plt.savefig(f'{base_location}/plots/grouped {f} dist.png', dpi=500)
    plt.show()

    hf.test_anova(results_grouped, f, list(results_grouped['Condition'].unique()))

                
 