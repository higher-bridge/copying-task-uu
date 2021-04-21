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
import seaborn as sns

import helperfunctions as hf
from constants_analysis import EXCLUDE_TRIALS, MIDLINE, base_location

PLOT = True

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


# # =============================================================================
# # EXPLORE COMPLETION TIMES PER CONDITION
# # =============================================================================
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

if PLOT:
    # Distplot of trial times
    plt.figure()
    for condition in list(time_dict.keys()):
        df = all_times_melt.loc[all_times_melt['Condition'] == condition]
        sns.distplot(df['Time (ms)'], label=condition)
    
    plt.legend()
    plt.xlim((0, 20100))
    plt.xlabel('Trial completion time (ms)')
    plt.savefig(f'{base_location}/plots/trial-time-dist.png', dpi=500)
    plt.show()
    
    # Barplot of mean trial times
    plt.figure()
    sns.catplot('Condition', 'Time (ms)', data=all_times_melt, kind='bar', estimator=np.median)
    plt.xlabel('Trial completion time (ms)')
    plt.tight_layout()
    plt.savefig(f'{base_location}/plots/trial-time-bar.png', dpi=500)
    plt.show()

# Calculate mean and SD trial times
mean_times = pd.DataFrame(columns=['Median', 'SD', 'Condition'])

for condition in list(time_dict.keys()):
    times = time_dict[condition]
    
    mt = pd.DataFrame({'Median'     : round(np.nanmedian(times), 1),
                        'SD'       : round(np.nanstd(times), 1),
                        'Condition': condition},
                      index=[0])
    mean_times = mean_times.append(mt, ignore_index=True)
 
print('\nMedian trial durations:')
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
                
                num_crossings = hf.get_midline_crossings(list(df_t['gavx']),
                                                          midline=MIDLINE)
                
                c = pd.DataFrame({'Crossings': num_crossings,
                                  'Condition': condition,
                                  'Trial'    : trial,
                                  'ID'       : ID},
                                  index=[0])
                
                crossings = crossings.append(c, ignore_index=True)

print('\nANOVA test Crossings')
hf.test_anova(crossings, 'Crossings', list(crossings['Condition'].unique()))

if PLOT:
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
    sns.catplot('Condition', 'Crossings', data=crossings, kind='bar', estimator=np.median)
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
                
                num_fixations = hf.get_left_side_fixations(list(df_t['gavx']),
                                                            midline=MIDLINE)
                
                c = pd.DataFrame({'Fixations': num_fixations,
                                  'Condition': condition,
                                  'Trial'    : trial,
                                  'ID'       : ID},
                                  index=[0])
                
                left_fixations = left_fixations.append(c, ignore_index=True)
                
                # hf.scatterplot_fixations(df_t, 'gavx', 'gavy', f'{num_fixations} left fix', save=False, savestr='')

print('\nANOVA test left fixations')
hf.test_anova(left_fixations, 'Fixations', list(left_fixations['Condition'].unique()))
    
if PLOT:
    # Distplot of number of left fixations
    plt.figure()
    for condition in sorted(list(left_fixations['Condition'].unique())):
        df = left_fixations.loc[left_fixations['Condition'] == condition]
        sns.distplot(df['Fixations'], label=condition, bins=15)
        
    plt.legend()
    plt.xlim((-1, 20))
    plt.xlabel('Number of left-side fixations')
    plt.savefig(f'{base_location}/plots/leftFixations-dist.png', dpi=500)
    plt.show()
    
    # Barplot of number of left fixations
    plt.figure()
    sns.catplot('Condition', 'Fixations', data=left_fixations, kind='bar', estimator=np.median)
    plt.ylabel('Median of number of left-side fixations')
    plt.tight_layout()
    plt.savefig(f'{base_location}/plots/leftFixations-bar.png', dpi=500)
    plt.show()

# =============================================================================
# EXPLORE DWELL TIME FIXATIONS ON THE EXAMPLE GRID PER CONDITION
# =============================================================================
dwell_times = pd.DataFrame(columns=['Dwell Time', 'Condition', 'Trial', 'ID'])

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
                
                num_fixations = hf.get_dwell_times(list(df_t['gavx']),
                                                    list(df_t['start']),
                                                    list(df_t['end']),
                                                    midline=MIDLINE)
                
                c = pd.DataFrame({'Dwell Time': num_fixations,
                                  'Condition': condition,
                                  'Trial'    : trial,
                                  'ID'       : ID},
                                  index=[0])
                
                dwell_times = dwell_times.append(c, ignore_index=True)
                
dwell_times = dwell_times.dropna(axis=0)

print('\nANOVA test dwell time')
hf.test_anova(dwell_times, 'Dwell Time', list(dwell_times['Condition'].unique()))
    
if PLOT:
    # Distplot of number of left fixations
    plt.figure()
    for condition in sorted(list(dwell_times['Condition'].unique())):
        df = dwell_times.loc[dwell_times['Condition'] == condition]
        sns.distplot(df['Dwell Time'], label=condition, bins=15)
        
    plt.legend()
    plt.xlim((-50, 4100))
    plt.xlabel('Total dwell time on left side of screen (ms)')
    plt.savefig(f'{base_location}/plots/dwellTime-dist.png', dpi=500)
    plt.show()
    
    # Barplot of number of left fixations
    plt.figure()
    sns.catplot('Condition', 'Dwell Time', data=dwell_times, kind='bar', estimator=np.median)
    plt.ylabel('Median of total dwell time on left side of screen (ms)')
    plt.tight_layout()
    plt.savefig(f'{base_location}/plots/dwellTime-bar.png', dpi=500)
    plt.show()

# =============================================================================
# EXPLORE VISIBLE TIME DISTRIBUTIONS
# =============================================================================
vis_times = pd.DataFrame(columns=['Visible Time', 'Condition', 'Trial', 'ID'])

# Data outside of trial start-end are marked with 999 so will be filtered in the next steps
for i, ID in enumerate(list(pp_info['ID'].unique())): 
    filenames = [f for f in correct_placements_files if ID in f]
    filename = filenames[0]
    
    df = pd.read_csv(filename)
    
    for condition in list(df['Condition'].unique()):
        df_c = df.loc[df['Condition'] == condition]
        
        for trial in list(df_c['Trial'].unique()): 
            if trial not in EXCLUDE_TRIALS:
                df_t = df_c.loc[df_c['Trial'] == trial]
                
                vt = list(df_t['visibleTime'])[0]
                
                c = pd.DataFrame({'Visible Time': vt,
                                  'Condition': condition,
                                  'Trial'    : trial,
                                  'ID'       : ID},
                                 index=[0])
                
                vis_times = vis_times.append(c, ignore_index=True)
                
if PLOT:
    plt.figure()
    for condition in sorted(list(vis_times['Condition'].unique())):
        if condition != 0:
            df = vis_times.loc[vis_times['Condition'] == condition]
            sns.distplot(df['Visible Time'], label=condition)
        
    plt.legend(title='Condition')
    plt.savefig(f'{base_location}/plots/visibleTime-dist.png', dpi=500)
    plt.show()
    
    # Barplot of number of left fixations
    plt.figure()
    sns.catplot('Condition', 'Visible Time', data=vis_times, kind='bar', estimator=np.median)
    plt.tight_layout()
    plt.savefig(f'{base_location}/plots/visibleTime-bar.png', dpi=500)
    plt.show()


