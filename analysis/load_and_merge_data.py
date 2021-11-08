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
import mouse_analysis
import numpy as np
import pandas as pd
from constants_analysis import base_location, EXCLUDE_EXCEL_BASED
from joblib import Parallel, delayed


def load_and_merge(ID, ID_dict, pp_info, base_location, exclude_trials):
    task_event_files = hf.find_files(ID, ID_dict[ID], base_location, '-eventTracking.csv')
    eventfiles = hf.find_files(ID, ID_dict[ID], base_location, '-events.csv')
    mousefiles = hf.find_files(ID, ID_dict[ID], base_location, '-mouseTracking-')
    c_placements = hf.find_files(ID, ID_dict[ID], base_location, '-correctPlacements.csv')
    samplefiles = hf.find_files(ID, ID_dict[ID], base_location, '-samples.csv')

    if len(samplefiles) > 0:
        samples_present = True
    else:
        samples_present = False

    # Concatenate all separate session files into one larger file
    task_events = hf.concat_event_files(task_event_files)
    events = hf.concat_event_files(eventfiles).drop('trial', axis=1)
    correct_placements = hf.concat_event_files(c_placements)
    if samples_present:
        samples = hf.concat_event_files(samplefiles)

    # Order the df in the proper condition, retrieved from participant_info.xlsx
    # (I didn't design the mousetracker filenames with chronological ordering)

    mousedata = hf.concat_event_files(mousefiles)
    condition_order = hf.get_condition_order(pp_info, ID)

    if condition_order == [0, 1, 0, 1]:  # Patient study runs 0, 1, 0, 1
        mousedata = hf.add_sessions(mousedata)
    else:
        mousedata = hf.order_by_condition(mousedata, condition_order)
        mousedata['Session'] = [1] * len(mousedata)

    # Create two empty lists in which we will fill in the appropriate trials/condition value
    trial_list = np.empty(len(events), dtype=int)
    trial_list[:] = 999

    condition_list = np.empty(len(events), dtype=int)
    condition_list[:] = 999

    session_list = np.empty(len(events), dtype=int)
    session_list[:] = 999

    # Do the same for the sample df's
    if samples_present:
        trial_list_samp = np.empty(len(samples), dtype=int)
        trial_list_samp[:] = 999

        condition_list_samp = np.empty(len(samples), dtype=int)
        condition_list_samp[:] = 999

        session_list_samp = np.empty(len(samples), dtype=int)
        session_list_samp[:] = 999

    # Create empty list to track whether mouse is being dragged
    dragging_list = np.empty(len(events), dtype=bool)
    dragging_list[:] = False

    dragging_mouse = np.empty(len(mousedata), dtype=bool)
    dragging_mouse[:] = False

    # Create a list to track which rows in mousedata are valid
    mouse_valid = np.zeros(len(mousedata), dtype=bool)
    mouse_valid[:] = False

    # For each condition, each trial, get the start and end times and relate it to eye events
    for session in list(task_events['Session'].unique()):
        session_df = task_events.loc[task_events['Session'] == session]
        correct_session = correct_placements.loc[correct_placements['Session'] == session]

        for x, condition in enumerate(list(session_df['Condition'].unique())):
            condition_df = session_df.loc[session_df['Condition'] == condition]
            correct_condition = correct_session.loc[correct_session['Condition'] == condition]

            all_trials = list(condition_df['Trial'].unique())
            for trial_num in all_trials:

                # Find a row in exclude_trials where all variables match. If the length of the resulting df
                # equals 0, this trial was not in the exclusion list
                exclusion = exclude_trials.loc[exclude_trials['ID'] == ID]
                exclusion = exclusion.loc[exclusion['session'] == str(session)]
                exclusion = exclusion.loc[exclusion['condition'] == str(condition)]
                exclusion = exclusion.loc[exclusion['trial'] == str(trial_num)]

                if EXCLUDE_EXCEL_BASED and len(exclusion) > 0:
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
                    start_idx = hf.find_nearest_index(events_['end'], start, keep_index=True)
                    end_idx = hf.find_nearest_index(events_['start'], end, keep_index=True)

                    if samples_present:
                        start_idx_samp = hf.find_nearest_index(samples['time'], start)
                        end_idx_samp = hf.find_nearest_index(samples['time'], end)

                    # Find during which events items were being dragged with the mouse
                    task_init = trial_df.loc[trial_df['Event'] == 'Task init']
                    timediff = task_init['TimeDiff'].values[0]

                    for index, row in correct_trial.iterrows():
                        end_drag = row['Time'] - timediff
                        start_drag = end_drag - row['dragDuration']
                        start_drag_idx = hf.find_nearest_index(events['start'], start_drag)
                        end_drag_idx = hf.find_nearest_index(events['end'], end_drag)

                        dragging_list[start_drag_idx:end_drag_idx] = True

                        start_drag_mouse = hf.find_nearest_index(mousedata['TrackerTime'], start_drag)
                        end_drag_mouse = hf.find_nearest_index(mousedata['TrackerTime'], end_drag)

                        dragging_mouse[start_drag_mouse:end_drag_mouse] = True

                    mouse_start_idx = hf.find_nearest_index(mousedata['TrackerTime'], start)
                    mouse_end_idx = hf.find_nearest_index(mousedata['TrackerTime'], end)

                    if start_idx < end_idx:
                        # Set everything between start and end of trial with that condition/trial
                        trial_list[start_idx:end_idx] = trial_num
                        condition_list[start_idx:end_idx] = condition
                        session_list[start_idx:end_idx] = session
                        mouse_valid[mouse_start_idx:mouse_end_idx] = True

                        if samples_present:
                            trial_list_samp[start_idx_samp:end_idx_samp] = trial_num
                            condition_list_samp[start_idx_samp:end_idx_samp] = condition
                            session_list_samp[start_idx_samp:end_idx_samp] = session

                    if '2031' in ID and trial_num == 2 and condition == 0 and session == 1:
                        print(session, condition, trial_num, end - start, end_idx - start_idx)

    # Append trial/condition info to eye events df            
    events['Trial'] = trial_list
    events['Condition'] = condition_list
    events['Session'] = session_list
    events['Dragging'] = dragging_list

    if samples_present:
        samples['Trial'] = trial_list_samp
        samples['Condition'] = condition_list_samp
        samples['Session'] = session_list_samp

    if '2031' in ID:
        print('')

    # Retrieve how many valid trials were recorded per condition
    num_trials = hf.get_num_trials(events)
    b1 = int(num_trials[0])
    b2 = int(num_trials[1])

    try:
        b3 = int(num_trials[2])
        b4 = int(num_trials[3])
    except IndexError:
        b3, b4 = 0, 0

    # Remove invalid rows from mousetracker
    mousedata['Valid'] = mouse_valid
    mousedata['Dragging'] = dragging_mouse
    mousedata = mousedata.loc[mousedata['Valid'] == True]

    # Compute mouse 'fixations' and 'saccades' for modelling later on
    mouse_events = mouse_analysis.get_mouse_events(list(mousedata['x']), list(mousedata['y']),
                                                   list(mousedata['TrackerTime']),
                                                   list(mousedata['Trial']),
                                                   list(mousedata['Condition']),
                                                   list(mousedata['Session']),
                                                   list(mousedata['Dragging']))

    if ID not in os.listdir(f'../results'):
        os.mkdir(f'../results/{ID}')

    # Write everything to csv
    mouse_events.to_csv(f'../results/{ID}/{ID}-mouseEvents.csv')
    events.to_csv(f'../results/{ID}/{ID}-allFixations.csv')
    task_events.to_csv(f'../results/{ID}/{ID}-allEvents.csv')
    correct_placements.to_csv(f'../results/{ID}/{ID}-allCorrectPlacements.csv')
    if samples_present:
        samples.to_csv(f'../results/{ID}/{ID}-allSamples.csv')

    return [b1, b2, b3, b4]


if __name__ == '__main__':
    path = Path(base_location)
    all_IDs = list(path.glob('*-*-*/'))
    all_IDs = sorted(all_IDs)

    ID_dict_temp = hf.write_IDs_to_dict(all_IDs)
    pp_info = pd.read_excel('../results/participant_info.xlsx', engine='openpyxl')
    pp_info['ID'] = pp_info['ID'].astype(str)
    ID_list = [str(ID) for ID in list(pp_info['ID'].unique())]

    # print(list(ID_dict_temp.keys()))
    # print(ID_list)
    ID_dict = {ID: value for ID, value in ID_dict_temp.items() if ID in ID_list}

    pp_exclude = pd.read_excel('../results/participant_exclude_trials.xlsx', engine='openpyxl').astype(str)

    # Sit back, this will take a while (multiprocessing)
    # ID_dict_list = [ID_dict] * len(ID_list)
    # pp_info_list = [pp_info] * len(ID_list)
    # exclude_list = [pp_exclude] * len(ID_list)
    # base_location_list = [base_location] * len(ID_list)
    #
    # results = Parallel(n_jobs=-2,
    #                    backend='loky',
    #                    verbose=True)(delayed(load_and_merge) \
    #                                      (ID, IDd, ppi, bll, el) for ID, IDd, ppi, bll, el in zip(ID_list,
    #                                                                                               ID_dict_list,
    #                                                                                               pp_info_list,
    #                                                                                               base_location_list,
    #                                                                                               exclude_list))

    # Sit back, this wil take even longer (single core, uncomment in case multiprocessing doesn't work)
    results = []
    for ID in ID_list:
        results.append(load_and_merge(ID, ID_dict, pp_info, base_location, pp_exclude))
        print(f'Parsed {len(results)} of {len(ID_list)} files')

    pp_info['Trials condition 0'] = [b[0] for b in results]
    pp_info['Trials condition 1'] = [b[1] for b in results]
    pp_info['Trials condition 2'] = [b[2] for b in results]
    pp_info['Trials condition 3'] = [b[3] for b in results]

    try:
        pp_info = pp_info.drop(['Unnamed: 0'], axis=1)
    except Exception as e:
        print(e)

    pp_info.to_excel('../results/participant_info.xlsx')
