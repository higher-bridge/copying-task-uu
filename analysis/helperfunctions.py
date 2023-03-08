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
from math import sqrt
from pathlib import Path, PureWindowsPath

import constants_analysis as constants
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import pingouin as pg
import seaborn as sns
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return sorted(allFiles)


def rename_features(x):
    if x == 'Number of crossings':
        return 'Example grid inspections'
    elif x == 'Fixations at grid':
        return 'Fixations per inspection'
    elif x == 'Items placed after crossing':
        return 'Items placed per inspection'
    elif x == 'Completion time (s)':
        return 'Completion time (s)'
    elif x == 'Errors per trial':
        return 'Errors per trial'
    elif x == 'Proportion spent waiting':
        return 'Proportion spent waiting'
    else:
        return x


def euclidean_distance(loc1, loc2):
    dist = [(a - b) ** 2 for a, b in zip(loc1, loc2)]
    return sqrt(sum(dist))


def find_nearest_index(array, timestamp, keep_index=False):
    if keep_index:
        new_series = abs(array - timestamp)
        idx = new_series.nsmallest(1).index[0]
    else:
        array = np.asarray(array)
        idx = (np.abs(array - timestamp)).argmin()

    return idx


def find_nearest_location(loc1, locations):
    distances = np.asarray([euclidean_distance(loc1, loc2) for loc2 in locations])
    min_dist_idx = distances.argmin()

    return locations[min_dist_idx]


def normalize_column(df: pd.DataFrame,
                     feature: str,
                     groupby: str = 'ID',
                     groupby2: str = 'Condition') -> pd.DataFrame:
    # Split by group (ID)
    group_df = []
    groups = list(df[groupby].unique())
    for g in groups:
        df_ = df.loc[df[groupby] == g]

        # Split by second group (Condition)
        split_df = []
        splits = list(df_[groupby2].unique())
        for s in splits:
            df_s = df_.loc[df_[groupby2] == s]

            # Perform a rolling window over each ID/Condition pair
            df_s[feature] = df_s[feature].rolling(5, center=True).mean()
            split_df.append(df_s)

        # Combine conditions back into one df, and scale whole ID
        split_df = pd.concat(split_df, ignore_index=True)
        split_df[feature] = MinMaxScaler().fit_transform(np.array(split_df[feature]).reshape(-1, 1))
        # split_df[feature] = StandardScaler().fit_transform(np.array(split_df[feature]).reshape(-1, 1))
        group_df.append(split_df)

    # Combine all IDs back together
    group_df = pd.concat(group_df, ignore_index=True)
    return group_df


def remove_outliers(df: pd.DataFrame,
                    feature: str,
                    low_perc=0.0,
                    high_perc=99.0,
                    return_indices=False) -> pd.DataFrame:
    len_before = len(df)

    # Select column
    x_ = np.array(df[feature].astype(float))
    x = x_[~np.isnan(x_)]

    # Get low and high percentiles
    low = np.percentile(x, low_perc)
    high = np.percentile(x, high_perc)

    if not return_indices:
        # Only select data that is within the percentiles
        df = df.loc[df[feature] >= low]
        df = df.loc[df[feature] <= high]

        len_after = len(df)
        print(f'{len_before - len_after} trials dropped ({feature})')

        return df

    else:
        indices = np.argwhere((np.array(df[feature]) < low) |
                              (np.array(df[feature]) > high)).ravel()

        print(f'{feature}:'.ljust(26),
              f'Trials before={len_before}, '
              f'removed={len(indices)} '
              f'({round((len(indices) / len_before) * 100, 2)}%), '
              f'remaining={len_before - len(indices)}, '
              f'Per condition={(len_before - len(indices)) / 4}'
              )

        return indices


def prepare_stimuli(paths: list, x_locs: list, y_locs: list, locations: list, in_pixels=False, snap_location=True,
                    zoom=.1):
    stimulus_paths = ['../' + PureWindowsPath(x).as_posix() for x in paths]
    stimuli = [mpimg.imread(p) for p in stimulus_paths]
    image_boxes = [OffsetImage(s, zoom=zoom) for s in stimuli]

    if not in_pixels:
        annotation_boxes = [AnnotationBbox(im, locations[x + y * 3], frameon=False) for x, y, im in
                            zip(x_locs,
                                y_locs,
                                image_boxes)]
    if in_pixels and snap_location:
        annotation_boxes = [AnnotationBbox(im, find_nearest_location((x, y), locations), frameon=False) for x, y, im in
                            zip(x_locs,
                                y_locs,
                                image_boxes)]
    if in_pixels and not snap_location:
        annotation_boxes = [AnnotationBbox(im, (x, y), frameon=False) for x, y, im in
                            zip(x_locs,
                                y_locs,
                                image_boxes)]

    return annotation_boxes


def locate_trial(df, condition, trial, session=None):
    if session is not None:
        try:
            df = df.loc[df['Session'] == session]
        except:
            pass

    df = df.loc[df['Condition'] == condition]
    df = df.loc[df['Trial'] == trial]

    return df


def write_IDs_to_dict(all_IDs: list):
    """ Takes a list of IDs and writes to dict with sub-lists for
    barreled IDs (1001-2, etc.) """
    ID_dict = dict()

    for ID in all_IDs:
        if ID.is_dir():
            ID = str(ID.parts[-1])
            # temp_ID = ID[0:3]
            temp_ID = ID[0:4]

            if ID[5] != '0':

                if temp_ID not in ID_dict.keys():
                    ID_dict[temp_ID] = []
                    ID_dict[temp_ID].append(ID[5:])
                else:
                    # ID_dict[temp_ID].append(ID[-1])
                    ID_dict[temp_ID].append(ID[5:])

    return ID_dict


def find_files(ID: str, sessions: list, location: str, subfix: str):
    """ Returns a list of tuples (file, session) which match {ID}-{session} for all sessions """
    all_files = []

    # all_folders = [(f'{location}/{ID}', 0)]
    # if len(sessions) > 0:
    #     [all_folders.append((f'{location}/{ID}-{s}', s)) for s in sessions]

    all_folders = [(f'{location}/{ID}-{s}', s) for s in sessions if s != '' and '0' not in s]

    for folder, session in all_folders:
        file_list = sorted([f'{folder}/{f}' for f in os.listdir(folder)])
        for file in file_list:
            if subfix in file:
                all_files.append((file, session))

    return all_files


def concat_event_files(eventfiles: list):
    """ Reads and concatenates multiple dataframes """
    all_sessions = []

    for ef, session in eventfiles:
        # Load events
        sess = pd.read_csv(ef)
        sess['Session'] = [int(session[0])] * len(sess)
        all_sessions.append(sess)

    events = pd.concat(all_sessions, ignore_index=True)
    return events


def add_sessions(df):
    # Find where the first set of conditions [0, 1] stops and where te second set begins. Then assign session 1 and 2
    # respectively.
    session_list = np.empty(len(df), dtype=int)
    session_list[:] = 999

    conditions = np.array(df['Condition'])
    ones = np.argwhere(conditions == 1)

    middle_marker_idx = 0
    for i, j in zip(ones[0: len(session_list) - 1], ones[1: len(session_list)]):
        if j - i > 1:
            middle_marker_idx = i[0]
            break

    session_list[0:middle_marker_idx] = 1
    session_list[middle_marker_idx + 1:len(session_list)] = 2
    df['Session'] = session_list

    return df


def order_by_condition(df, condition_order: list):
    new_order = []

    for c in condition_order:
        df_c = df.loc[df['Condition'] == c]
        new_order.append(df_c)

    new_df = pd.concat(new_order, ignore_index=True)

    return new_df


def get_condition_order(df, ID: str, blocks: list = [1, 2, 3, 4]):
    df_id = df.loc[df['ID'] == ID]

    condition_order = []

    for condition in blocks:
        colname = f'Block {condition}'
        c = list(df_id[colname])
        c = c[0]
        condition_order.append(c)

    return condition_order


def get_num_trials(df, exclude=constants.EXCLUDE_TRIALS):
    num_trials = []

    sessions = sorted(list(df['Session'].unique()))

    for session in sessions:
        if session != 999:
            df_s = df.loc[df['Session'] == session]

            conditions = sorted(list(df_s['Condition'].unique()))

            for condition in conditions:
                if condition != 999:
                    df_c = df_s.loc[df_s['Condition'] == condition]
                    trials = [t for t in list(df_c['Trial'].unique()) if t not in exclude]

                    num_trials.append(len(trials))

    return num_trials


def remove_from_pp_info(df, cols_to_check: list, min_value: int = 10):
    IDs_to_remove = []

    for ID in list(df['ID'].unique()):
        df_id = df.loc[df['ID'] == ID]

        for col in cols_to_check:
            trials = list(df_id[col])[0]

            if trials < min_value:
                IDs_to_remove.append(ID)

    for ID in IDs_to_remove:
        df = df.loc[df['ID'] != ID]

    return df


def condition_number_to_name(x):
    if x == 0:
        return 'baseline'
    elif x == 1:
        return 'high'
    elif x == 2:
        return 'medium'
    elif x == 3:
        return 'low'
    else:
        return 999


def get_midline_crossings(xpos: list, midline=constants.MIDLINE):
    num_crossings = 0
    prev_x = constants.SCREEN_CENTER[0]

    for x in xpos:
        if prev_x > midline and x < midline:
            num_crossings += 1

        prev_x = x

    return num_crossings


def get_midline_cross_times(xpos: list, timestamp: list, midline=constants.MIDLINE):
    cross_time = []
    prev_x = constants.SCREEN_CENTER[0]

    for x, ts in zip(xpos, timestamp):
        if prev_x > midline and x < midline:
            cross_time.append(ts)

        prev_x = x

    return cross_time


def get_post_inspection_times(xpos: list, timestamp: list, midline=constants.MIDLINE):
    cross_time = []
    prev_x = constants.SCREEN_CENTER[0]

    for x, ts in zip(xpos, timestamp):
        if prev_x < midline and x > midline:
            cross_time.append(ts)

        prev_x = x

    return cross_time


def get_left_side_fixations(xpos: list, midline=constants.MIDLINE):
    return len([x for x in xpos if x < midline])


def get_left_ratio_fixations(xpos: list, midline=constants.MIDLINE):
    return len([x for x in xpos if x < midline]) / len(xpos)


def get_dwell_times(xpos: list, starts: list, ends: list, midline=constants.MIDLINE):
    dwell_times = []

    for x, start, end in zip(xpos, starts, ends):
        if x < midline:
            dwell_times.append(end - start)

    if len(dwell_times) > 0:
        return sum(dwell_times)
    else:
        return np.nan


def get_dwell_time_per_crossing(xpos: list, starts: list, ends: list, midline=constants.MIDLINE):
    dwell_times = []
    prev_x = 2560

    keep_track = False

    for x, s, e in zip(xpos, starts, ends):
        if prev_x > midline and x < midline:
            keep_track = True
            x_list = []
            s_list = []
            e_list = []
        elif prev_x < midline and x > midline and keep_track:
            keep_track = False
            dwell_times.append(get_dwell_times(x_list, s_list, e_list))

        if keep_track:
            x_list.append(x)
            s_list.append(s)
            e_list.append(e)

        prev_x = x

    return dwell_times


def get_fixations_at_grid(x_list, y_list, starts, ends):
    example_boundaries = constants.example_boundaries
    example_min, example_max = example_boundaries[0], example_boundaries[1]

    fixations_at_grid = {'x': [],
                         'y': [],
                         'start': [],
                         'end': [],
                         'duration': []}

    for x, y, start, end in zip(x_list, y_list, starts, ends):
        if example_min[0] < x < example_max[0]:
            if example_min[1] < y < example_max[1]:
                # Fixation was within grid boundaries
                fixations_at_grid['x'].append(x)
                fixations_at_grid['y'].append(y)
                fixations_at_grid['start'].append(start)
                fixations_at_grid['end'].append(end)
                fixations_at_grid['duration'].append(end - start)

    return fixations_at_grid


def get_dwell_time_at_grid(x_list, y_list, starts, ends):
    fixations_at_grid = get_fixations_at_grid(x_list, y_list, starts, ends)

    return sum(fixations_at_grid['duration'])


def get_useful_crossings(df, x_list, starts, ends, min_dur=120):
    starts = np.array(starts).astype(float)
    ends = np.array(ends).astype(float)

    # Find when grid was showing/hiding
    showing = df.loc[df['Event'] == 'Showing grid']
    hiding = df.loc[df['Event'] == 'Hiding grid']

    showing = pd.concat([showing, df.loc[df['Event'] == 'Task init']], ignore_index=True)

    if len(showing) > len(hiding):
        hiding = pd.concat([hiding, df.loc[df['Event'] == 'Finished trial']], ignore_index=True)
    elif len(showing) < len(hiding):
        showing = pd.concat([showing, df.loc[df['Event'] == 'Finished trial']], ignore_index=True)

    time_showing = np.array(showing['TrackerTime'].astype(int))
    time_hiding = np.array(hiding['TrackerTime'].astype(int))

    # Get start and end of crossing event
    prev_x = constants.MIDLINE
    dwellstarts, dwellends = [], []
    for i, (x, start, end) in enumerate(zip(x_list, starts, ends)):
        if prev_x >= constants.MIDLINE and x < constants.MIDLINE:
            dwellstarts.append(i)
        elif prev_x < constants.MIDLINE and x >= constants.MIDLINE:
            dwellends.append(i - 1)

        prev_x = x

    if len(dwellstarts) > len(dwellends):
        dwellends.append(dwellstarts[-1])
    elif len(dwellstarts) < len(dwellends):
        dwellends.pop()

    num_crossings = 0
    useful_starts = []
    useful_ends = []

    # For each dwell on the left, check if grid was visible during that time
    for start, end in zip(dwellstarts, dwellends):
        starttime = starts[start]
        endtime = ends[end]

        # Break out of loop, because we only need to know whether this dwell contained one event
        for showing, hiding in zip(time_showing, time_hiding):
            # If the grid started showing sometime during a dwell (but must be X ms before dwell ended)
            if starttime < showing < endtime - min_dur:
                num_crossings += 1
                useful_starts.append(starttime)
                useful_ends.append(endtime)
                break
            # If the grid was already showing, maybe it started hiding during a dwell. Must be X ms after dwell start
            elif starttime + min_dur < hiding < endtime:
                num_crossings += 1
                useful_starts.append(starttime)
                useful_ends.append(endtime)
                break
            # Maybe a dwell occurred only while the grid was showing. Count this too
            elif showing < starttime < hiding and showing < endtime < hiding:
                num_crossings += 1
                useful_starts.append(starttime)
                useful_ends.append(endtime)
                break

    return num_crossings, useful_starts, useful_ends


def get_fixated_hourglass_duration(df, x_list, y_list, starts, ends):
    fixations_at_grid = get_fixations_at_grid(x_list, y_list, starts, ends)

    hourglass_fixation_duration = 0

    showing = df.loc[df['Event'] == 'Showing hourglass']
    hiding = df.loc[df['Event'] == 'Hiding hourglass']

    if len(showing) == 0:
        return 0

    if len(showing) > len(hiding):
        hiding = pd.concat([hiding, df.loc[df['Event'] == 'Finished trial']], ignore_index=True)

    time_showing = showing['TrackerTime']
    time_hiding = hiding['TrackerTime']

    for show, hide in zip(time_showing, time_hiding):
        # Only count this hourglass duration if there was a fixation within the example grid
        for start, end in zip(fixations_at_grid['start'], fixations_at_grid['end']):
            if show < start < hide:
                fix_end = min(end, hide)
                duration = fix_end - start
            elif show < end < hide:  # This is not very likely to happen
                fix_start = max(start, show)
                duration = end - fix_start
            else:
                duration = 0

            hourglass_fixation_duration += duration

    return hourglass_fixation_duration


def get_hourglass_duration(df):
    hourglasstimer = 0

    showing = df.loc[df['Event'] == 'Showing hourglass']
    hiding = df.loc[df['Event'] == 'Hiding hourglass']

    if len(showing) == 0:
        return 0

    if len(showing) > len(hiding):
        hiding = pd.concat([hiding, df.loc[df['Event'] == 'Finished trial']], ignore_index=True)

    time_showing = showing['TrackerTime']
    time_hiding = hiding['TrackerTime']

    for show, hide in zip(time_showing, time_hiding):
        hourglass_shown = hide - show
        hourglasstimer += hourglass_shown

    return hourglasstimer


def get_errors_while_occluded(df):
    incorrect_idx = [i for i, x in enumerate(list(df['Event'])) if 'Incorrectly placed' in x]
    incorrect_ts = np.array(df['TrackerTime'])[incorrect_idx]

    showing = df.loc[df['Event'] == 'Showing hourglass']
    hiding = df.loc[df['Event'] == 'Hiding hourglass']

    if len(showing) == 0:
        return 0

    if len(showing) > len(hiding):
        hiding = pd.concat([hiding, df.loc[df['Event'] == 'Finished trial']], ignore_index=True)

    time_showing = np.array(showing['TrackerTime'])
    time_hiding = np.array(hiding['TrackerTime'])

    errors = 0
    for hide, show in zip(time_hiding[:-1], time_showing[1:]):
        for its in incorrect_ts:
            if hide < its <= show + 500:  # show + 500
                errors += 1

    return errors


def number_of_incorrect_placements_per_trial(df):
    incorrect_placements = 0

    for i, row in df.iterrows():
        event = row['Event']

        if 'Incorrectly placed' in event:
            incorrect_placements += 1

    return incorrect_placements


def number_of_correct_placements_per_trial(df):
    correct_placements = 0

    for i, row in df.iterrows():
        correct = row['Correct']

        if correct:
            correct_placements += 1

    return correct_placements


def get_placements_per_inspection(df_events, df_place, xpos: list, onsets: list, offsets):
    _, cross_times, back_times = get_useful_crossings(df_events, xpos, onsets, offsets)

    trial_start = df_events.loc[df_events['Event'] == 'Task init']
    trial_start = list(trial_start['TrackerTime'])[0]

    trial_end = df_events.loc[df_events['Event'] == 'Finished trial']
    trial_end = list(trial_end['TrackerTime'])[0]

    trial_dur = trial_end - trial_start

    if len(cross_times) <= len(back_times):
        cross_times.append(trial_end + 500)  # Add a few ms to make sure we catch the last placement(s)

    # Get list of timestamps of incorrect placements
    incorrect_idx = [i for i, x in enumerate(list(df_events['Event'])) if 'Incorrectly placed' in x]
    incorrect_ts = np.array(df_events['TrackerTime'])[incorrect_idx]

    # Add trackertime column to correct placement df
    timediff = list(df_events['TimeDiff'])[0]
    df_place['TrackerTime'] = np.array(df_place['Time']) - timediff

    # Get list of timestamps of correct placements
    df_place = df_place.loc[df_place['Correct'] == True]
    correct_ts = np.array(df_place['TrackerTime'])

    timestamps = pd.DataFrame(columns=['ts', 'correct'])
    timestamps['ts'] = np.concatenate([incorrect_ts, correct_ts])
    timestamps['correct'] = [False] * len(incorrect_ts) + [True] * len(correct_ts)
    timestamps = timestamps.sort_values(by='ts')

    placements = {'Timestamp': [np.nan] * len(back_times),
                  'Time since start': [np.nan] * len(back_times),
                  'Proportion since start': [np.nan] * len(back_times),
                  'Crossing': [i + 1 for i in range(len(back_times))],
                  'Inspection duration': [np.nan] * len(back_times),
                  'Placements': [0] * len(back_times),
                  'Correct placements': [0] * len(back_times),
                  'Streak': [0] * len(back_times)
                  }

    for j, (back, cross) in enumerate(zip(back_times, cross_times[1:])):
        counter = 0
        streak = 0
        correct_counter = 0

        placements['Timestamp'][j] = back
        placements['Time since start'][j] = back - trial_start
        placements['Proportion since start'][j] = (back - trial_start) / trial_dur
        placements['Inspection duration'][j] = back_times[j] - cross_times[j]

        # Loop through (timestamps of) placements
        for i, row in timestamps.iterrows():
            ts = row['ts']
            correct = row['correct']

            # If timestamp is between back and cross, a placement was made in this period
            if back <= ts <= cross:
                counter += 1
                placements['Placements'][j] = int(counter)

                if correct:
                    streak += 1
                    correct_counter += 1
                else:
                    streak = 0

                placements['Correct placements'][j] = int(correct_counter)
                placements['Streak'][j] = streak

    placements = pd.DataFrame(placements)

    crossings_no_placement = placements.loc[placements['Placements'] == 0]
    crossings_w_placement = placements.loc[placements['Placements'] > 0]

    return np.nanmean(placements['Correct placements']), \
           len(crossings_no_placement), \
           np.nanmean(crossings_w_placement['Correct placements']), \
           placements


def get_ttest(df: pd.DataFrame, dep_var: str, ind_var: str):
    tests = pd.read_excel(constants.RESULT_DIR / 'tests.xlsx', engine='openpyxl')
    testfeat = tests.loc[tests['Feature'] == dep_var]

    ind_vars = sorted(list(df[ind_var].unique()))

    iv_combinations = []
    for iv in ind_vars:
        for iv1 in ind_vars:
            if (iv != iv1) and ((iv, iv1) not in iv_combinations) and ((iv1, iv) not in iv_combinations):
                iv_combinations.append((iv, iv1))

    sigs = {'comb': [], 'p': [], 'sig': []}
    for comb in iv_combinations:
        # Locate only this condition + its comparison
        testf = testfeat.loc[testfeat['Condition'] == comb[0]]
        testf = testf.loc[testf['Comparison'] == comb[1]]

        p = testf['p'].values[0]

        if p < .001:
            sig = '***'
        elif p < .01:
            sig = '**'
        elif p < .05:
            sig = '*'
        else:
            sig = ''

        sigs['comb'].append(comb)
        sigs['p'].append(p)
        sigs['sig'].append(sig)

    return sigs


def compute_se(x, y):
    squared_error = (np.mean(x) - np.mean(y)) ** 2

    mms = MinMaxScaler()
    x = mms.fit_transform(np.array(x).reshape(-1, 1))
    y = mms.transform(np.array(y).reshape(-1, 1))

    norm_squared_error = (np.mean(x) - np.mean(y)) ** 2

    return squared_error, norm_squared_error
