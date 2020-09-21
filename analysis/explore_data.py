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

EXCLUDE_TRIALS = list(range(1, 6))
EXCLUDE_TRIALS.append(999)

base_location = '../results'

pp_info = pd.read_excel('../results/participant_info.xlsx')
pp_info['ID'] = [str(x).zfill(3) for x in list(pp_info['ID'])]


fixations_files = [f for f in hf.getListOfFiles(base_location) if '-allFixations.csv' in f]
task_events_files = [f for f in hf.getListOfFiles(base_location) if 'allEvents.csv' in f]
all_placements_files = [f for f in hf.getListOfFiles(base_location) if '-allAllPlacements.csv' in f]
correct_placements_files = [f for f in hf.getListOfFiles(base_location) if '-allCorrectPlacements.csv' in f]
    

# =============================================================================
# EXPLORE TOTAL/CORRECT/INCORRECT NUMBER OF TRIALS
# =============================================================================
total_trials = 0
correct_trials = 0
incorrect_trials = 0
no_data_loss = 0
no_data_loss_correct = 0

for i, ID in enumerate(list(pp_info['ID'].unique())): 
    filenames = [f for f in correct_placements_files if ID in f]
    filename = filenames[0]
    
    fix_filenames = [f for f in fixations_files if ID in f]
    fix_filename = fix_filenames[0]
    
    df = pd.read_csv(filename)
    fix_df = pd.read_csv(fix_filename)
    
    for condition in list(df['Condition'].unique()):
        df_c = df.loc[df['Condition'] == condition]
        fix_df_c = fix_df.loc[fix_df['Condition'] == condition]
        
        for trial in list(df_c['Trial'].unique()):
            if trial not in EXCLUDE_TRIALS:
                total_trials += 1
                
                df_t = df_c.loc[df_c['Trial'] == trial]
                fix_df_t = fix_df_c.loc[fix_df_c['Trial'] == trial]
                
                all_correct = np.all(df_t['Correct'].values)
                data_loss = len(fix_df_t) == 0
                
                if all_correct:
                    correct_trials += 1
                else:
                    incorrect_trials += 1
                    
                if not data_loss:
                    no_data_loss += 1
                    
                    if all_correct:
                        no_data_loss_correct += 1

print(f'Total: {total_trials} (average per person = {round(total_trials/len(correct_placements_files), 3)})')
print(f'Correct: {correct_trials} (ratio={round(correct_trials/total_trials, 3)})')
print(f'Incorrect: {incorrect_trials} (ratio={round(incorrect_trials/total_trials, 3)})')
print('')
print(f'No loss: {no_data_loss} (ratio={round(no_data_loss/total_trials, 3)})')
print(f'No loss + correct: {no_data_loss_correct} (ratio={round(no_data_loss_correct/total_trials, 3)})\n')


# =============================================================================
# EXPLORE COMPLETION TIMES PER CONDITION
# =============================================================================
time_dict = {f'Condition {i}': [] for i in range(4)}

for i, ID in enumerate(list(pp_info['ID'].unique())): 
    filenames = [f for f in task_events_files if ID in f]
    filename = filenames[0]
    
    df = pd.read_csv(filename)
    
    for condition in list(df['Condition'].unique()):
        df_c = df.loc[df['Condition'] == condition]
        
        for trial in list(df_c['Trial'].unique()): 
            if trial not in EXCLUDE_TRIALS:
                df_t = df_c.loc[df_c['Trial'] == trial]
                
                start_times = df_t.loc[df_t['Event'] == 'Task init']['TrackerTime']
                start = list(start_times)[0]
                
                end_times = df_t.loc[df_t['Event'] == 'Finished trial']['TrackerTime']
                end = list(end_times)[0]
                
                duration = end - start
                if duration < 20000:
                    time_dict[f'Condition {condition}'].append(duration)
                else:
                    time_dict[f'Condition {condition}'].append(np.nan)

# Melt the dataframe
all_times = pd.DataFrame(time_dict)
all_times_melt = all_times.melt(value_vars=['Condition 0',
                                         'Condition 1',
                                         'Condition 2',
                                         'Condition 3'],
                                var_name='Condition',
                                value_name='Time (ms)')

# Distplot of trial times
plt.figure()
for condition in list(time_dict.keys()):
    df = all_times_melt.loc[all_times_melt['Condition'] == condition]
    sns.distplot(df['Time (ms)'], label=condition)

plt.legend()
plt.xlim((0, 20100))
plt.savefig(f'{base_location}/plots/trial-time-dist.png', dpi=500)
plt.show()

# Barplot of mean trial times
plt.figure()
sns.catplot('Condition', 'Time (ms)', data=all_times_melt, kind='bar')
plt.tight_layout()
plt.savefig(f'{base_location}/plots/trial-time-bar.png', dpi=500)
plt.show()

# Calculate mean and SD trial times
mean_times = pd.DataFrame(columns=['Mean', 'SD', 'Condition'])

for condition in list(time_dict.keys()):
    times = time_dict[condition]
    
    mt = pd.DataFrame({'Mean'     : round(np.nanmean(times), 1),
                       'SD'       : round(np.nanstd(times), 1),
                       'Condition': condition},
                      index=[0])
    mean_times = mean_times.append(mt, ignore_index=True)
    
print(mean_times)

# =============================================================================
# EXPLORE NUMBER OF RIGHT->LEFT CROSSINGS
# =============================================================================
crossings = pd.DataFrame(columns=['Crossings', 'Condition', 'Trial', 'ID'])

# Data outside of trial start-end are marked with 999 so will be filtered in the next steps
for i, ID in enumerate(list(pp_info['ID'].unique())): 
    filenames = [f for f in fixations_files if ID in f]
    filename = filenames[0]
    
    df = pd.read_csv(filename)
    df = df.loc[df['type'] == 'fixation']
    
    for condition in list(df['Condition'].unique()):
        df_c = df.loc[df['Condition'] == condition]
        
        for trial in list(df_c['Trial'].unique()): 
            if trial not in EXCLUDE_TRIALS:
                df_t = df_c.loc[df_c['Trial'] == trial]
                
                num_crossings = hf.get_midline_crossings(list(df_t['gavx']))
                
                c = pd.DataFrame({'Crossings': num_crossings,
                                  'Condition': condition,
                                  'Trial'    : trial,
                                  'ID'       : ID},
                                 index=[0])
                
                crossings = crossings.append(c, ignore_index=True)


# Distplot of number of crossings
plt.figure()
for condition in sorted(list(crossings['Condition'].unique())):
    df = crossings.loc[crossings['Condition'] == condition]
    sns.distplot(df['Crossings'], label=condition, bins=15)
    
plt.legend()
plt.xlim((-1, 8))
plt.xlabel('Midline crossings (right to left)')
plt.savefig(f'{base_location}/plots/crossings-dist.png', dpi=500)
plt.show()

# Barplot of number of crossings
plt.figure()
sns.catplot('Condition', 'Crossings', data=crossings, kind='bar')
plt.ylabel('Midline crossings (right to left)')
plt.tight_layout()
plt.savefig(f'{base_location}/plots/crossings-bar.png', dpi=500)
plt.show()

# =============================================================================
# EXPLORE FIXATIONS ON THE EXAMPLE GRID PER CONDITION
# =============================================================================
left_fixations = pd.DataFrame(columns=['Fixations', 'Condition', 'Trial', 'ID'])

# Data outside of trial start-end are marked with 999 so will be filtered in the next steps
for i, ID in enumerate(list(pp_info['ID'].unique())): 
    filenames = [f for f in fixations_files if ID in f]
    filename = filenames[0]
    
    df = pd.read_csv(filename)
    df = df.loc[df['type'] == 'fixation']
    
    for condition in list(df['Condition'].unique()):
        df_c = df.loc[df['Condition'] == condition]
        
        for trial in list(df_c['Trial'].unique()): 
            if trial not in EXCLUDE_TRIALS:
                df_t = df_c.loc[df_c['Trial'] == trial]
                
                num_fixations = hf.get_left_side_fixations(list(df_t['gavx']))
                
                c = pd.DataFrame({'Fixations': num_fixations,
                                  'Condition': condition,
                                  'Trial'    : trial,
                                  'ID'       : ID},
                                 index=[0])
                
                left_fixations = left_fixations.append(c, ignore_index=True)


# Distplot of number of crossings
plt.figure()
for condition in sorted(list(left_fixations['Condition'].unique())):
    df = left_fixations.loc[left_fixations['Condition'] == condition]
    sns.distplot(df['Fixations'], label=condition, bins=15)
    
plt.legend()
plt.xlim((-1, 20))
plt.xlabel('Left-side fixations')
plt.savefig(f'{base_location}/plots/leftFixations-dist.png', dpi=500)
plt.show()

# Barplot of number of crossings
plt.figure()
sns.catplot('Condition', 'Fixations', data=left_fixations, kind='bar')
plt.ylabel('Left-side fixations')
plt.tight_layout()
plt.savefig(f'{base_location}/plots/leftFixations-bar.png', dpi=500)
plt.show()

# =============================================================================
# EXPLORE FIXATIONS ON THE EXAMPLE GRID AS RATIO OF TOTAL
# =============================================================================
ratio_fixations = pd.DataFrame(columns=['Fixations', 'Condition', 'Trial', 'ID'])

# Data outside of trial start-end are marked with 999 so will be filtered in the next steps
for i, ID in enumerate(list(pp_info['ID'].unique())): 
    filenames = [f for f in fixations_files if ID in f]
    filename = filenames[0]
    
    df = pd.read_csv(filename)
    df = df.loc[df['type'] == 'fixation']
    
    for condition in list(df['Condition'].unique()):
        df_c = df.loc[df['Condition'] == condition]
        
        for trial in list(df_c['Trial'].unique()): 
            if trial not in EXCLUDE_TRIALS:
                df_t = df_c.loc[df_c['Trial'] == trial]
                
                num_fixations = hf.get_left_ratio_fixations(list(df_t['gavx']))
                
                c = pd.DataFrame({'Fixations': num_fixations,
                                  'Condition': condition,
                                  'Trial'    : trial,
                                  'ID'       : ID},
                                 index=[0])
                
                ratio_fixations = ratio_fixations.append(c, ignore_index=True)


# Distplot of number of crossings
plt.figure()
for condition in sorted(list(ratio_fixations['Condition'].unique())):
    df = ratio_fixations.loc[ratio_fixations['Condition'] == condition]
    sns.distplot(df['Fixations'], label=condition, bins=15)
    
plt.legend()
plt.xlim((-0.05, 1.05))
plt.xlabel('Left-side fixations as ratio of total')
plt.savefig(f'{base_location}/plots/ratioFixations-dist.png', dpi=500)
plt.show()

# Barplot of number of crossings
plt.figure()
sns.catplot('Condition', 'Fixations', data=ratio_fixations, kind='bar')
plt.ylabel('Left-side fixations as ratio of total')
plt.tight_layout()
plt.savefig(f'{base_location}/plots/ratioFixations-bar.png', dpi=500)
plt.show()

