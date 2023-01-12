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

import os
from pathlib import Path

import helperfunctions as hf
import numpy as np
import pandas as pd
from constants_analysis import EXCLUDE_EXCEL_BASED, RESULT_DIR
from joblib import Parallel, delayed


def load_and_merge(ID, ID_dict, pp_info, base_location, exclude_trials):
    # ID = str(int(ID))
    task_event_files = hf.find_files(ID, ID_dict[ID], base_location, '-eventTracking.csv')
    eventfiles = hf.find_files(ID, ID_dict[ID], base_location, '-events.csv')
    c_placements = hf.find_files(ID, ID_dict[ID], base_location, '-correctPlacements.csv')

    # Concatenate all separate session files into one larger file
    task_events = hf.concat_event_files(task_event_files)
    events = hf.concat_event_files(eventfiles)
    correct_placements = hf.concat_event_files(c_placements)

    # Create two empty lists in which we will fill in the appropriate trials/condition value
    trial_list = np.empty(len(events), dtype=int)
    trial_list[:] = 999

    condition_list = np.empty(len(events), dtype=int)
    condition_list[:] = 999

    session_list = np.empty(len(events), dtype=int)
    session_list[:] = 999

    # For each condition, each trial, get the start and end times and relate it to eye events
    for session in list(task_events['Session'].unique()):
        session_df = task_events.loc[task_events['Session'] == session]
        correct_session = correct_placements.loc[correct_placements['Session'] == session]

        for x, condition in enumerate(list(session_df['Condition'].unique())):
            condition_df = session_df.loc[session_df['Condition'] == condition]
            correct_condition = correct_session.loc[correct_session['Condition'] == condition]

            all_trials = list(condition_df['Trial'].unique())
            for trial_num in all_trials:
                try:

                    # Find a row in exclude_trials where all variables match. If the length of the resulting df
                    # equals 0, this trial was not in the exclusion list
                    if EXCLUDE_EXCEL_BASED and len(exclude_trials) > 0:
                        # exclusion = exclude_trials.loc[exclude_trials['ID'] == ID]
                        # exclusion = exclusion.loc[exclusion['session'] == str(session)]
                        # exclusion = exclusion.loc[exclusion['condition'] == str(condition)]
                        # exclusion = exclusion.loc[exclusion['trial'] == str(trial_num)]

                        print(f'Excluded {ID}-{session} c{condition} t{trial_num} based on exclusion-excel')
                    else:
                        trial_df = condition_df.loc[condition_df['Trial'] == trial_num]
                        correct_trial = correct_condition.loc[correct_condition['Trial'] == trial_num]

                        # Retrieve where eventTracking has written trial init and trial finish
                        start_times = trial_df.loc[trial_df['Event'] == 'Task init']['TrackerTime']
                        start = list(start_times)[0]

                        end_times = trial_df.loc[trial_df['Event'] == 'Finished trial']['TrackerTime']
                        end = list(end_times)[0]

                        # Match the timestamp to the closest timestamp in the fixations df
                        events_ = events.loc[events['Session'] == session]
                        start_idx = hf.find_nearest_index(events_['offset'], start, keep_index=True)
                        end_idx = hf.find_nearest_index(events_['onset'], end, keep_index=True)

                        if start_idx < end_idx:
                            # Set everything between start and end of trial with that condition/trial
                            trial_list[start_idx:end_idx] = trial_num
                            condition_list[start_idx:end_idx] = condition
                            session_list[start_idx:end_idx] = session
                except Exception as e:
                    print(f'Could not parse {ID}, s{session}, c{condition}, t{trial_num}: {e}')

    # Append trial/condition info to eye events df            
    events['Trial'] = trial_list
    events['Condition'] = condition_list
    events['Session'] = session_list

    if ID not in os.listdir(f'../results'):
        os.mkdir(f'../results/{ID}')

    # Write everything to csv
    events.to_csv(f'../results/{ID}/{ID}-allFixations.csv')
    task_events.to_csv(f'../results/{ID}/{ID}-allEvents.csv')
    correct_placements.to_csv(f'../results/{ID}/{ID}-allCorrectPlacements.csv')

    # Retrieve how many valid trials were recorded per condition
    num_trials = hf.get_num_trials(events)
    while len(num_trials) < 4:
        num_trials.append(np.nan)

    return num_trials


if __name__ == '__main__':
    path = RESULT_DIR
    all_IDs = list(path.glob('*-*-*/'))
    all_IDs = sorted(all_IDs)

    ID_dict_temp = hf.write_IDs_to_dict(all_IDs)
    pp_info = pd.read_excel('../results/participant_info.xlsx', engine='openpyxl')
    print(pp_info.head())
    pp_info['ID'] = pp_info['ID'].astype(int)
    pp_info['ID'] = pp_info['ID'].astype(str)
    ID_list = [str(ID) for ID in list(pp_info['ID'].unique())]

    ID_dict = {ID: value for ID, value in ID_dict_temp.items() if ID in ID_list}

    pp_exclude = pd.read_excel('../results/participant_exclude_trials.xlsx', engine='openpyxl').astype(str)

    # Sit back, this may take a while (multiprocessing)
    ID_dict_list = [ID_dict] * len(ID_list)
    pp_info_list = [pp_info] * len(ID_list)
    exclude_list = [pp_exclude] * len(ID_list)
    base_location_list = [RESULT_DIR] * len(ID_list)

    results = Parallel(n_jobs=4,
                       backend='loky',
                       verbose=True)(delayed(load_and_merge) \
                                         (ID, IDd, ppi, bll, el) for ID, IDd, ppi, bll, el in zip(ID_list,
                                                                                                  ID_dict_list,
                                                                                                  pp_info_list,
                                                                                                  base_location_list,
                                                                                                  exclude_list))

    # Sit back, this may take even longer (single core, uncomment in case multiprocessing doesn't work)
    # results = []
    # for ID in ID_list:
    #     results.append(load_and_merge(ID, ID_dict, pp_info, base_location, pp_exclude))
    #     print(f'Parsed {len(results)} of {len(ID_list)} files')

    pp_info['Trials condition 0'] = [b[0] for b in results]
    pp_info['Trials condition 1'] = [b[1] for b in results]
    pp_info['Trials condition 2'] = [b[2] for b in results]
    pp_info['Trials condition 3'] = [b[3] for b in results]

    try:
        pp_info = pp_info.drop(['Unnamed: 0'], axis=1)
    except KeyError as e:
        pass

    pp_info.to_excel('../results/participant_info.xlsx')
