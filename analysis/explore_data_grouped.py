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
import textwrap

pp_info = pd.read_excel('../results/participant_info.xlsx', engine='openpyxl')
pp_info['ID'] = [str(x) for x in list(pp_info['ID'])]

# pp_info = hf.remove_from_pp_info(pp_info, [f'Trials condition {i}' for i in range(4)])


fixations_files = [f for f in hf.getListOfFiles(constants.base_location) if '-allFixations.csv' in f]
task_events_files = [f for f in hf.getListOfFiles(constants.base_location) if '-allEvents.csv' in f]
all_placements_files = [f for f in hf.getListOfFiles(constants.base_location) if '-allCorrectPlacements.csv' in f]

# Add column names for additional measures
features = [
    'Number of crossings',
    'Total dwell time per crossing (ms)',
    'Completion time (s)',
    'Fixations per second',
    'Saccade velocity',
    'Peak velocity',
    'Incorrect placements',
    'Correct placements',
    'Wrong per correct',
    'Crossings per correct item',
    'Hourglass shown duration (s)',
    'Hourglass fixated duration (s)',
    'Total dwell time at grid (s)',
    'Dwell time per crossing (ms)',
    'Dwell time at grid per correct item (s)'
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
                        dwell_times_at_grid = hf.get_dwell_time_at_grid(list(fixations['gavx']),
                                                                        list(fixations['gavy']),
                                                                        list(fixations['start']),
                                                                        list(fixations['end']))
                        hourglass_fixated_duration = hf.get_fixated_hourglass_duration(task_df_t,
                                                                               list(fixations['gavx']),
                                                                               list(fixations['gavy']),
                                                                               list(fixations['start']),
                                                                               list(fixations['end']))
                        total_hourglass_duration = hf.get_hourglass_duration(task_df_t)

                        # Compute additional outcome measures here
                        errors = np.nan
                        incorrect_placements = hf.number_of_incorrect_placements_per_trial(task_df_t)
                        correct_placements = hf.number_of_correct_placements_per_trial(placement_df_t)
                        wrong_per_correct = incorrect_placements / correct_placements if correct_placements > 0 else incorrect_placements
                        crossing_per_correct = num_crossings / correct_placements if correct_placements > 0 else 0
                        grid_visible_dwell = dwell_times_at_grid - hourglass_fixated_duration
                        grid_visible_dwell_per_crossing = grid_visible_dwell / num_crossings if num_crossings > 0 else np.nan
                        grid_visible_dwell_per_correct = grid_visible_dwell / correct_placements if correct_placements > 0 else 0

                        # Add additional outcome measures here (as specified in features var, line 35)
                        r = pd.DataFrame({'ID': ID,
                                          'Session': session,
                                          'Condition': int(condition),
                                          'Session-condition': f'{session}-{condition}',
                                          'Trial-session': f'{trial}-{session}',
                                          'Trial': int(trial),
                                          'Number of crossings': float(num_crossings),
                                          'Total dwell time per crossing (ms)': float(np.median(dwell_time_pc)),
                                          'Completion time (s)': float(completion_time / 1000),
                                          'Fixations per second': float(len(fixations) / (completion_time / 1000)),
                                          'Saccade velocity': float(np.median(saccades['avel'])),
                                          'Peak velocity': float(np.median(saccades['pvel'])),
                                          'Incorrect placements': float(incorrect_placements),
                                          'Correct placements': float(correct_placements),
                                          'Wrong per correct': float(wrong_per_correct),
                                          'Crossings per correct item': float(crossing_per_correct),
                                          'Hourglass shown duration (s)': float(total_hourglass_duration / 1000),
                                          'Hourglass fixated duration (s)': float(hourglass_fixated_duration / 1000),
                                          'Total dwell time at grid (s)': float(grid_visible_dwell / 1000),
                                          'Dwell time per crossing (ms)': float(grid_visible_dwell_per_crossing),
                                          'Dwell time at grid per correct item (s)': float(
                                              grid_visible_dwell_per_correct / 1000)
                                          },
                                         index=[0])
                        results = results.append(r, ignore_index=True)

# =============================================================================
# AGGREGATE BY MEDIAN                
# =============================================================================
features = [
    'Number of crossings',
    'Dwell time per crossing (ms)',
    'Completion time (s)',
    'Fixations per second',
    'Saccade velocity',
    'Peak velocity',
    # 'Incorrect placements',
    # 'Correct placements',
    # 'Wrong per correct',
    # 'Crossings per correct item',
    # 'Hourglass shown duration (s)',
    # 'Hourglass fixated duration (s)',
    # 'Total dwell time at grid (s)',
    # 'Total dwell time per crossing (ms)',
    # 'Dwell time at grid per correct item (s)'
]

results = results.dropna()

# Group by ID and Condition, use median. If there is a session 2, we're working with patient data and should use
# Condition-session as grouping variable
if 2 in list(results['Session']):
    condition_var = 'Condition'  # Usually 'Session-condition'
    label = condition_var

    blue = (0 / 256, 170 / 256, 155 / 256)
    orange = (241 / 256, 142 / 256, 0 / 256)
    sns.set_palette([blue, orange, blue, orange])

else:
    condition_var = 'Condition'
    label = 'Reliability of visual access'
    sns.set_palette(sns.color_palette('tab10'))

results_grouped = results.groupby(['ID', condition_var]).agg({f: ['median'] for f in features}).reset_index()
results_grouped.columns = results_grouped.columns.get_level_values(0)

trial_counts = results.groupby(['ID', condition_var]).agg('count').reset_index()
results_grouped['Trial count'] = trial_counts['Trial']

# Rename condition variables if necessary
if condition_var == 'Condition':
    results_grouped['Condition number'] = results_grouped[condition_var].apply(hf.condition_number_to_name)
else:
    results_grouped['Condition number'] = results_grouped[condition_var]

results_grouped.to_csv(f'{constants.base_location}/results-grouped-ID-condition.csv')

# =============================================================================
# SEPARATE PLOTS
# =============================================================================
rcParams['font.size'] = 14

# Define two linestyles
ls = ['-', '--']

# Make a plot for each feature
for f in features:
    fig = plt.figure(figsize=(7.5, 5))
    axes = fig.subplots(1, 2, gridspec_kw=dict(wspace=0))

    # Boxplot
    sns.boxplot(x='Condition number', y=f, data=results_grouped,
                fliersize=0,
                ax=axes[0])

    # Stripplot (add individual points)
    sns.swarmplot(x='Condition number', y=f, data=results_grouped,
                  color='black', #jitter=False,
                  ax=axes[0])

    axes[0].set_xlabel(label)

    # Add kdeplot (histogram/kernel density estimation) to the side
    condition_list = list(results_grouped['Condition number'].unique())
    if condition_var == 'Session-condition':
        condition_list = sorted(condition_list)

    for i, cond in enumerate(condition_list):
        df_c = results_grouped.loc[results_grouped['Condition number'] == cond]
        sns.kdeplot(y=f, data=df_c,
                    color=sns.color_palette()[i],
                    fill=True, alpha=.25,
                    linestyle=ls[int(cond[0]) - 1] if condition_var == 'Session-condition' else ls[0],
                    clip=axes[0].get_ylim(),
                    label=cond,
                    ax=axes[1])

    # Set parameters for kdeplot
    axes[1].set_ylabel('')
    axes[1].set_yticks([])
    axes[1].set_ylim(axes[0].get_ylim())
    axes[1].set_xticks([])
    axes[1].set_xlabel('Distribution density')

    if label == 'Session-condition':
        axes[1].legend(title='Session-\ncondition', fontsize=12)
    else:
        # axes[1].legend(title=label, fontsize=12)
        pass

    # Set overall parameters and save
    plt.tight_layout()
    plt.savefig(f'{constants.base_location}/plots/grouped {f} box.png', dpi=500)
    plt.show()

#     print('\n###########################')
#     hf.test_friedman(results_grouped, 'Condition number', f)
#     hf.test_posthoc(results_grouped, f, list(results_grouped['Condition number'].unique()))

# =============================================================================
# COMBINED BOXPLOTS
# =============================================================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times']
rcParams['font.size'] = 11

colors = sns.color_palette('tab10')
sns.set_palette(colors)

# Adjust nrows and ncols as needed such that nrows * ncols == len(features)
nrows, ncols = 2, 3
f = plt.figure(figsize=(7.5, 5))
axes = [f.add_subplot(nrows, ncols, s) for s in range(1, len(features) + 1)]

for i, feat in enumerate(features):
    sns.boxplot(x='Condition number', y=feat, data=results_grouped,
                palette='Blues', fliersize=0, ax=axes[i])
    sns.swarmplot(x='Condition number', y=feat, data=results_grouped,
                  color='black', ax=axes[i])
    axes[i].set_xlabel('')
    axes[i].set_ylabel(feat, fontsize=13)

    if i < (len(features) - ncols):
        # Only place xticks on the bottom row
        axes[i].set_xticks([])

    if i == 4:
        axes[i].set_xlabel(label, fontsize=13)

f.tight_layout()  # (pad=1, w_pad=0.2)
f.savefig(f'{constants.base_location}/plots/combined-boxplots.png', dpi=700)
plt.show()
