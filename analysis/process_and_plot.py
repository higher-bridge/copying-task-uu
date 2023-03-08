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

import textwrap

import constants_analysis as constants
import helperfunctions as hf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch
from scipy.stats import spearmanr

pp_info = pd.read_excel('../results/participant_info.xlsx', engine='openpyxl')
pp_info['ID'] = pp_info['ID'].astype(str)

exclude_IDs = []

fixations_files = [f for f in hf.getListOfFiles(constants.RESULT_DIR) if '-allFixations.csv' in f]
task_events_files = [f for f in hf.getListOfFiles(constants.RESULT_DIR) if '-allEvents.csv' in f]
all_placements_files = [f for f in hf.getListOfFiles(constants.RESULT_DIR) if '-allCorrectPlacements.csv' in f]

# Add column names for additional measures
features = [
    'Number of all crossings',
    'Number of crossings',
    'Total dwell time per crossing (ms)',
    'Completion time total (s)',
    'Fixations per second',
    'Saccade velocity',
    'Peak velocity',
    'Errors per trial',
    'Correct placements',
    'Wrong per correct',
    'Crossings per correct item',
    'Items placed after crossing',
    'Hourglass shown duration (s)',
    'Time spent waiting (s)',
    'Proportion spent waiting overall',
    'Proportion spent waiting',
    'Total dwell time at grid (s)',
    'Dwell time per crossing (ms)',
    'Dwell time at grid per correct item (s)',
    'Completion time (s)',
    'Errors per second',
    'Errors while occluded',
    'Fixations at grid',
    'Inspections without placement',
    'Correct per inspection if placed',
    # 'Correct placement times',
    # 'Incorrect placement times',
    # 'Correct streak'
]

cols = ['ID', 'Session', 'Condition', 'Session-condition', 'Trial']
[cols.append(f) for f in features]
results = pd.DataFrame(columns=cols)

# =============================================================================
# WRITE VARIABLES TO ROW FOR EACH ID, SESSION, CONDITION AND TRIAL
# =============================================================================
# Data outside of trial start-end are marked with 999 so will be filtered in the next steps
valid_IDs = [x for x in list(pp_info['ID'].unique()) if x not in exclude_IDs]
print(f'N={len(valid_IDs)}')

cross_times = {'ID': [],
               'Condition': [],
               'Trial': [],
               'Timestamp': []}

all_placements = []

for ID in valid_IDs:
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
                        try:
                            fix_df_t = fix_df_c.loc[fix_df_c['Trial'] == trial]
                            task_df_t = task_df_c.loc[task_df_c['Trial'] == trial]
                            placement_df_t = placement_df_c.loc[placement_df_c['Trial'] == trial]

                            fixations = fix_df_t.loc[fix_df_t['label'] == 'FIXA']
                            saccades = fix_df_t.loc[fix_df_t['label'] == 'SACC']

                            start_times = task_df_t.loc[task_df_t['Event'] == 'Task init']['TrackerTime']
                            start = list(start_times)[0]

                            end_times = task_df_t.loc[task_df_t['Event'] == 'Finished trial']['TrackerTime']
                            end = list(end_times)[0]

                            completion_time = end - start
                            num_crossings = hf.get_midline_crossings(list(fixations['avg_x']))
                            crossings = hf.get_midline_cross_times(list(fixations['avg_x']),
                                                                   list(fixations['onset']))
                            num_fixations = hf.get_left_side_fixations(list(fixations['avg_x']))
                            dwell_times = hf.get_dwell_times(list(fixations['avg_x']),
                                                             list(fixations['onset']),
                                                             list(fixations['offset']))
                            dwell_time_pc = hf.get_dwell_time_per_crossing(list(fixations['avg_x']),
                                                                           list(fixations['onset']),
                                                                           list(fixations['offset']))
                            dwell_times_at_grid = hf.get_dwell_time_at_grid(list(fixations['avg_x']),
                                                                            list(fixations['avg_y']),
                                                                            list(fixations['onset']),
                                                                            list(fixations['offset']))
                            hourglass_fixated_duration = hf.get_fixated_hourglass_duration(task_df_t,
                                                                                           list(fixations['avg_x']),
                                                                                           list(fixations['avg_y']),
                                                                                           list(fixations['onset']),
                                                                                           list(fixations['offset']))
                            total_hourglass_duration = hf.get_hourglass_duration(task_df_t)
                            useful_crossings, _, _ = hf.get_useful_crossings(task_df_t,
                                                                             list(fixations['avg_x']),
                                                                             list(fixations['onset']),
                                                                             list(fixations['offset']))
                            fix_at_grid = hf.get_fixations_at_grid(list(fixations['avg_x']),
                                                                   list(fixations['avg_y']),
                                                                   list(fixations['onset']),
                                                                   list(fixations['offset']))
                            errors_occluded = hf.get_errors_while_occluded(task_df_t)

                            correct_per_crossing, insp_no_place, correct_if_placement, placements = hf.get_placements_per_inspection(
                                task_df_t,
                                placement_df_t,
                                fixations['avg_x'],
                                fixations['onset'],
                                fixations['offset'])

                            placements['ID'] = [ID] * len(placements)
                            placements['Condition'] = [condition] * len(placements)
                            placements['Trial'] = [trial] * len(placements)
                            all_placements.append(placements)

                            errors = np.nan
                            incorrect_placements = hf.number_of_incorrect_placements_per_trial(task_df_t)
                            correct_placements = hf.number_of_correct_placements_per_trial(placement_df_t)
                            wrong_per_correct = incorrect_placements / correct_placements if correct_placements > 0 else incorrect_placements
                            crossing_per_correct = useful_crossings / correct_placements if correct_placements > 0 else 0
                            correct_per_crossing = correct_placements / useful_crossings if useful_crossings > 0 else 0
                            grid_visible_dwell = dwell_times_at_grid - hourglass_fixated_duration
                            grid_visible_dwell_per_crossing = grid_visible_dwell / useful_crossings if useful_crossings > 0 else np.nan
                            grid_visible_dwell_per_correct = grid_visible_dwell / correct_placements if correct_placements > 0 else 0
                            prop_spent_waiting = hourglass_fixated_duration / completion_time
                            prop_hourglass = total_hourglass_duration / completion_time
                            rel_prop_spent_waiting = prop_spent_waiting / prop_hourglass

                            # Add additional outcome measures here (as specified in features var, line 35)
                            r = pd.DataFrame({'ID': ID,
                                              'Session': session,
                                              'Condition': int(condition),
                                              'Session-condition': f'{session}-{condition}',
                                              'Trial-session': f'{trial}-{session}',
                                              'Trial': int(trial),
                                              'Number of all crossings': float(num_crossings),
                                              'Number of crossings': float(useful_crossings),
                                              'Total dwell time per crossing (ms)': float(np.nanmedian(dwell_time_pc)),
                                              'Completion time total (s)': float(completion_time / 1000),
                                              'Fixations per second': float(len(fixations) / (completion_time / 1000)),
                                              'Saccade velocity': float(np.nanmedian(saccades['avg_vel'])),
                                              'Peak velocity': float(np.nanmedian(saccades['peak_vel'])),
                                              'Errors per trial': float(incorrect_placements),
                                              'Correct placements': float(correct_placements),
                                              'Wrong per correct': float(wrong_per_correct),
                                              'Crossings per correct item': float(crossing_per_correct),
                                              'Items placed after crossing': float(correct_per_crossing),
                                              'Hourglass shown duration (s)': float(total_hourglass_duration / 1000),
                                              'Time spent waiting (s)': float(hourglass_fixated_duration / 1000),
                                              'Proportion spent waiting overall': float(prop_spent_waiting),
                                              'Proportion spent waiting': float(rel_prop_spent_waiting),
                                              'Total dwell time at grid (s)': float(grid_visible_dwell / 1000),
                                              'Dwell time per crossing (ms)': float(grid_visible_dwell_per_crossing),
                                              'Dwell time at grid per correct item (s)': float(
                                                  grid_visible_dwell_per_correct / 1000),
                                              'Completion time (s)': float(
                                                  completion_time - hourglass_fixated_duration) / 1000,
                                              'Errors per second': float(incorrect_placements / (
                                                      (completion_time - hourglass_fixated_duration) / 1000)),
                                              'Errors while occluded': float(errors_occluded),
                                              'Fixations at grid': float(len(fix_at_grid['x']) / useful_crossings),
                                              'Inspections without placement': float(insp_no_place),
                                              'Correct per inspection if placed': float(correct_if_placement),
                                              # 'Correct placement times': float(placement_times_correct),
                                              # 'Incorrect placement times': float(placement_times_incorrect),
                                              # 'Correct streak': float(streak)
                                              },
                                             index=[0])
                            results = pd.concat([results, r], ignore_index=True)

                            for cross in crossings:
                                cross_times['ID'].append(ID)
                                cross_times['Condition'].append(condition)
                                cross_times['Trial'].append(trial)
                                cross_times['Timestamp'].append(cross - start)

                        except ZeroDivisionError:
                            pass

# Define: [0] the features that we use, [1] the aggregate function, [2] tick marks, [3] tick labels
features = [
    ('Number of crossings', np.nanmean, [2, 3, 4, 5, 6, 7, 8, 9], [' 2', ' 3', ' 4', ' 5', ' 6', ' 7', ' 8', ' 9']),
    ('Fixations at grid', np.nanmean, [1, 1.5, 2, 2.5, 3, 3.5], [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]),
    ('Items placed after crossing', np.nanmean, [0.5, 1.0, 1.5, 2.0, 2.5], [0.5, 1.0, 1.5, 2.0, 2.5]),
    ('Completion time (s)', np.nanmedian, [9, 10, 12, 14, 16, 18, 20], ['', 10, 12, 14, 16, 18, 20]),
    ('Errors per trial', np.nanmean, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
    ('Proportion spent waiting', np.nanmean, [0, 0.05, 0.1, 0.15, 0.2], [0, 0.05, 0.1, 0.15, 0.2]),
    ('Inspections without placement', np.nanmean, [], []), ('Correct per inspection if placed', np.nanmean, [], []),
    ('Hourglass shown duration (s)', np.nanmedian, [], [])]

# =============================================================================
# REMOVE OUTLIERS / OTHER OPTIONAL SCALING
# =============================================================================
for f_ in features:
    f = f_[0]

    idxs = hf.remove_outliers(results.copy(), f, return_indices=True, high_perc=95)
    results[f].iloc[idxs] = np.nan

# =============================================================================
# AGGREGATE BY MEDIAN
# =============================================================================
results.to_csv(f'{constants.RESULT_DIR}/results-pregrouped.csv')
# results = results.dropna()

condition_var = 'Condition'
label = 'Availability of external information'

# Group variables by the specified aggregate function
results_grouped = results.groupby(['ID', condition_var]).agg({f[0]: [f[1]] for f in features}).reset_index()
results_grouped.columns = results_grouped.columns.get_level_values(0)

# Add a count column
trial_counts = results.groupby(['ID', condition_var]).agg('count').reset_index()
results_grouped['Trial count'] = trial_counts['Trial']

# Convert condition names
results_grouped['Condition number'] = results_grouped[condition_var].apply(hf.condition_number_to_name)
results_grouped.to_csv(f'{constants.RESULT_DIR}/results-grouped-ID-condition.csv')

# Pivot table for analysis
medians = {'Feature': [],
           'Condition': [],
           'Mdn': [],
           'MAD': []
           }

for f in features:
    feat = f[0]
    df_pivot = pd.pivot(results_grouped, columns=['Condition number'], values=feat, index='ID')
    df_pivot.to_csv(constants.RESULT_DIR / f'pivot_{feat}.csv')

    for col in list(df_pivot.columns):
        mdn = df_pivot[col].median().round(3)
        mad = df_pivot[col].mad().round(3)

        medians['Feature'].append(feat)
        medians['Condition'].append(col)
        medians['Mdn'].append(mdn)
        medians['MAD'].append(mad)

medians = pd.DataFrame(medians)
medians.to_excel(constants.RESULT_DIR / 'medians.xlsx')

# Remove the inspections without placement variable from the list
features = features[:-3]
# =============================================================================
# COMBINED BOXPLOTS (FIGURE 2 IN MANUSCRIPT)
# =============================================================================
rcParams['font.family'] = 'sans-serif'
rcParams['font.serif'] = 'Helvetica'
rcParams['font.size'] = 9

# Define two linestyles
ls = ['-', '--']

colors = list(sns.color_palette('Blues', 4))

# Adjust nrows and ncols as needed such that nrows * ncols == len(features)
nrows, ncols = 2, 3
f = plt.figure(figsize=(7.5, 5))
axes = [f.add_subplot(nrows, ncols, s) for s in range(1, len(features) + 1)]

for i, feat_ in enumerate(features):
    feat = feat_[0]

    # Make boxplot
    sns.boxplot(x='Condition', y=feat, data=results_grouped,
                palette=colors,
                fliersize=0,
                linewidth=1,
                boxprops={'alpha': .7},
                ax=axes[i])

    # Add lines and markers for individuals
    for ID in list(results_grouped['ID'].unique()):
        results_ID = results_grouped.loc[results_grouped['ID'] == ID]
        sns.lineplot(x='Condition', y=feat, data=results_ID,
                     color='gray',
                     linestyle='--',
                     linewidth=.4,
                     marker='.', markersize=4,
                     markerfacecolor='gray', markeredgecolor='black',
                     ax=axes[i],
                     zorder=50)

    # Retrieve y-axis info, then load the post-hoc ttest results
    ylim = axes[i].get_ylim()
    yticks = axes[i].get_yticks()

    height = ylim[1]
    sigs = hf.get_ttest(results_grouped, feat, 'Condition')

    ss = 0
    for s in range(len(sigs['comb'])):
        comb = sigs['comb'][s]
        sig = sigs['sig'][s]

        # If significant, add a line and asterisks. We do some messing around with the height of the figure and the
        # lines to get everything to line up somewhat nicely
        if sig != '':
            h = height * (1 + ss / 25)
            axes[i].plot([comb[0], comb[1]], [h, h],
                         linestyle='-', color='black', linewidth=.8)
            axes[i].text(comb[0], h, sig,
                         ha='left', va='center',
                         fontsize=7)
            ss += 1

    new_height = axes[i].get_ylim()[1]
    axes[i].set_ylim((feat_[2][0], new_height))

    axes[i].set_yticks(feat_[2], feat_[3])
    axes[i].set_ylabel(hf.rename_features(feat), fontsize=10)

    if i < (len(features) - ncols):
        axes[i].set_xticks([])
    else:
        axes[i].set_xticks([0, 1, 2, 3], ['Always', 'High', 'Medium', 'Low'], fontsize=9)

    if i == 4:
        axes[i].set_xlabel(label, fontsize=10)
    else:
        axes[i].set_xlabel('')

    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)

f.tight_layout()
f.savefig(f'{constants.RESULT_DIR}/plots/Figure 2.pdf', dpi=600)
plt.show()

################
# Other plots
################
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.serif'] = 'Helvetica'
# rcParams['font.size'] = 9
#
# colors = sns.color_palette('Blues', 4)

# all_placements = pd.concat(all_placements, ignore_index=True)
# all_placements['Condition'] = all_placements['Condition'].apply(hf.condition_number_to_name)
# all_placements = all_placements.loc[all_placements['Placements'] == 0]
# all_placements = all_placements.loc[all_placements['Time since start'] < 42000]

# plt.figure(figsize=(7.5, 4))
# sns.boxplot(data=all_placements, x='Condition', y='Proportion since start',
#             palette=colors)
# plt.show()


# plt.figure(figsize=(7.5, 4))
# sns.histplot(data=all_placements, x='Proportion since start', hue='Condition',
#              kde=True,
#              # stat='proportion',
#              binwidth=0.02,
#              palette=colors)
# plt.ylabel('Frequency of inspections without placement')
# plt.show()


# all_placements_grouped = all_placements.groupby(['ID', 'Condition', 'Trial']).agg('count')
#
# plt.figure(figsize=(7.5, 4))
# sns.lineplot(data=all_placements_grouped, x='Trial', y='Proportion since start', hue='Condition',
#              # kde=True,
#              # stat='proportion',
#              # binwidth=0.02,
#              palette=colors)
# plt.ylabel('Number of inspections without placement')
# plt.show()
