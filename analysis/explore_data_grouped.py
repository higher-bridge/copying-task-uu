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

import constants_analysis as constants
import helperfunctions as hf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

pp_info = pd.read_excel('../results/participant_info.xlsx', engine='openpyxl')
pp_info['ID'] = [str(x) for x in list(pp_info['ID'])]

# pp_info = hf.remove_from_pp_info(pp_info, [f'Trials condition {i}' for i in range(4)])


fixations_files = [f for f in hf.getListOfFiles(constants.base_location) if '-allFixations.csv' in f]
task_events_files = [f for f in hf.getListOfFiles(constants.base_location) if '-allEvents.csv' in f]
all_placements_files = [f for f in hf.getListOfFiles(constants.base_location) if '-allCorrectPlacements.csv' in f]

features = [
            'Number of crossings',
            'Dwell time per crossing (ms)',
            'Completion time (s)', 
            'Fixations per second',
            'Saccade velocity',
            'Peak velocity'
            # 'Errors'
            ]

cols = ['ID', 'Session', 'Condition', 'Session-condition', 'Trial']
[cols.append(f) for f in features]
results = pd.DataFrame(columns=cols)

# =============================================================================
# WRITE VARIABLES TO ROW FOR EACH ID, SESSION, CONDITION AND TRIAL
# =============================================================================
# Data outside of trial start-end are marked with 999 so will be filtered in the next steps
for ID in list(pp_info['ID'].unique()): 
    fix_filenames = [f for f in fixations_files if ID in f]
    fix_filename = fix_filenames[0]
    
    task_filenames = [f for f in task_events_files if ID in f]
    task_filename = task_filenames[0]

    placements_filenames = [f for f in all_placements_files if ID in f]
    placements_filename = placements_filenames[0]    
    
    fix_df = pd.read_csv(fix_filename)

    task_df = pd.read_csv(task_filename)
    placement_df = pd.read_csv(placements_filename)

    for session in list(placement_df['Session'].unique()):
        fix_df_s = fix_df.loc[fix_df['Session'] == session]
        task_df_s = task_df.loc[task_df['Session'] == session]
        placement_df_s = placement_df.loc[placement_df['Session'] == session]
    
        for condition in list(placement_df_s['Condition'].unique()):
            if condition != 999:
                fix_df_c = fix_df_s.loc[fix_df_s['Condition'] == condition]
                task_df_c = task_df_s.loc[task_df_s['Condition'] == condition]
                placement_df_c = placement_df_s.loc[placement_df_s['Condition'] == condition]

                for trial in list(placement_df_c['Trial'].unique()):
                    if trial not in constants.EXCLUDE_TRIALS:
                        fix_df_t = fix_df_c.loc[fix_df_c['Trial'] == trial]
                        task_df_t = task_df_c.loc[task_df_c['Trial'] == trial]
                        placement_df_t = placement_df_c.loc[placement_df_c['Trial'] == trial]

                        fixations = fix_df_t.loc[fix_df_t['type'] == 'fixation']
                        saccades = fix_df_t.loc[fix_df_t['type'] == 'saccade']

                        start_times = task_df_t.loc[task_df_t['Event'] == 'Task init']['TrackerTime']
                        start = list(start_times)[0]

                        end_times = task_df_t.loc[task_df_t['Event'] == 'Finished trial']['TrackerTime']
                        end = list(end_times)[0]

                        completion_time = end - start
                        num_crossings = hf.get_midline_crossings(list(fixations['gavx']))
                        num_fixations = hf.get_left_side_fixations(list(fixations['gavx']))
                        dwell_times = hf.get_dwell_times(list(fixations['gavx']),
                                                         list(fixations['start']),
                                                         list(fixations['end']))
                        dwell_time_pc = hf.get_dwell_time_per_crossing(list(fixations['gavx']),
                                                                       list(fixations['start']),
                                                                       list(fixations['end']))
                        errors = len(placement_df_t.loc[placement_df_t['Correct'] != True])

                        r = pd.DataFrame({'ID': ID,
                                          'Session': session,
                                          'Condition': int(condition),
                                          'Session-condition': f'{session}-{condition}',
                                          'Trial-session': f'{trial}-{session}',
                                          'Trial': int(trial),
                                          'Number of crossings': float(num_crossings),
                                          'Dwell time per crossing (ms)': float(np.median(dwell_time_pc)),
                                          'Completion time (s)': float(completion_time / 1000),
                                          'Fixations per second': float(len(fixations) / (completion_time / 1000)),
                                          'Saccade velocity': float(np.median(saccades['avel'])),
                                          'Peak velocity': float(np.median(saccades['pvel']))
                                          # 'Errors':float(errors)
                                          },
                                         index=[0])
                        results = results.append(r, ignore_index=True)

# =============================================================================
# AGGREGATE BY MEDIAN                
# =============================================================================
results = results.dropna()

# Group by ID and Condition, use median. If there is a session 2, we're working with patient data and should use
# Condition-session as grouping variable
if 2 in list(results['Session']):
    condition_var = 'Session-condition'
    label = condition_var
else:
    condition_var = 'Condition'
    label = 'Reliability of visual access'

results_grouped = results.groupby(['ID', condition_var]).agg({f: ['median'] for f in features}).reset_index()
results_grouped.columns = results_grouped.columns.get_level_values(0)

trial_counts = results.groupby(['ID', condition_var]).agg('count').reset_index()
results_grouped['Trial count'] = trial_counts['Trial']

# Calculate mean for the Errors value
# errors_grouped = results.groupby(['ID', 'Condition']).agg({f: ['mean'] for f in features}).reset_index()
# errors_grouped.columns = errors_grouped.columns.get_level_values(0)

# Drop median errors from results_grouped and append mean errors from errors_grouped
# results_grouped = results_grouped.drop(['Errors'], axis=1)
# results_grouped['Errors'] = errors_grouped['Errors']

# Rename condition variables if necessary
if condition_var == 'Condition':
    results_grouped['Condition number'] = results_grouped[condition_var].apply(hf.condition_number_to_name)
else:
    results_grouped['Condition number'] = results_grouped[condition_var]

results_grouped.to_csv(f'{constants.base_location}/results-grouped-ID-condition.csv')

# =============================================================================
# SEPARATE PLOTS
# =============================================================================
# colors = sns.color_palette("Blues")[2:]
#
# for f in features:
#     plt.figure(figsize=(4, 5))
#     sns.boxplot('Condition', f, data=results_grouped, #capsize=.1, errwidth=1.5,
#                 palette=colors)
#     plt.title(f)
#     plt.tight_layout()
#     plt.savefig(f'{constants.base_location}/plots/grouped {f} box.png', dpi=500)
#     plt.show()
#
#     plt.figure()
#     for c in list(results_grouped['Condition'].unique()):
#         plot_df = results_grouped.loc[results_grouped['Condition'] == c]
#         sns.distplot(plot_df[f], label=c)
#
#     plt.title(f)
#     plt.legend(title='Condition')
#     plt.savefig(f'{constants.base_location}/plots/grouped {f} dist.png', dpi=500)
#     plt.show()
#
#     print('\n###########################')
#     hf.test_friedman(results_grouped, 'Condition', f)
#     hf.test_posthoc(results_grouped, f, list(results_grouped['Condition'].unique()))

# =============================================================================
# COMBINED BOXPLOTS                
# =============================================================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times']
rcParams['font.size'] = 11

# y_lims = [(1.5, 7.5), (150, 1000), (4, 15), (2.5, 5), (100, 240), (210, 430)]

nrows, ncols = 2, 3
f = plt.figure(figsize=(7.5, 5))
axes = [f.add_subplot(nrows, ncols, s) for s in range(1, len(features) + 1)]

for i, feat in enumerate(features):
    sns.boxplot(x='Condition number', y=feat, data=results_grouped, #capsize=.1, errwidth=1.5,
                palette='Blues', ax=axes[i])
    axes[i].set_xlabel('')
    axes[i].set_ylabel(feat, fontsize=13)
    # axes[i].set_ylim(y_lims[i])
    
    if i < (len(features) - ncols):
        # Only place xticks on the bottom row
        axes[i].set_xticks([])
        
    if i == 4:
        axes[i].set_xlabel(label, fontsize=13)

f.tight_layout() #(pad=1, w_pad=0.2)
f.savefig(f'{constants.base_location}/plots/combined-boxplots.png', dpi=700)
plt.show()
    

# =============================================================================
# COMBINED DISTPLOTS                
# =============================================================================
ls = ['-', '--', '-.', ':']

nrows, ncols = 2, 3
f = plt.figure(figsize=(7.5, 5))
axes = [f.add_subplot(nrows, ncols, s) for s in range(1, len(features) + 1)]

for i, feat in enumerate(features):
    for j, c in enumerate(list(results_grouped['Condition number'].unique())):
        plot_df = results_grouped.loc[results_grouped['Condition number'] == c]
        sns.kdeplot(plot_df[feat], label=c,
                    ax=axes[i], linestyle=ls[j])
    
    axes[i].set_yticks([])

    if i % ncols != 0:
        axes[i].set_ylabel('')

    if i == 2:
        axes[i].legend(title=label)

f.tight_layout()
f.savefig(f'{constants.base_location}/plots/combined-distplots.png', dpi=500, bbox_inches='tight')
plt.show()




