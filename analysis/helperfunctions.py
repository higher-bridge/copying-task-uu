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
import pingouin as pg
import seaborn as sns
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.preprocessing import MinMaxScaler


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


def locate_trial(df, condition, trial):
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


def test_normality(df, dep_var, ind_vars):
    p_values = []

    for iv in ind_vars:
        df_iv = df.loc[df['Condition'] == iv]

        results = pg.normality(df_iv[dep_var])
        p = list(results['pval'])[0]

        p_values.append(p)

    print('')
    return p_values


def test_sphericity(df, dep_var, ind_var):
    p, W, _, _, pval = pg.sphericity(df, dep_var, ind_var, subject='ID')
    p = bool(p)

    print('')
    return p


def test_posthoc(df, dep_var, ind_vars, is_non_normal=None):
    # print(f'\n{dep_var}')
    ind_vars = sorted(ind_vars)

    if is_non_normal == None:
        normality_p = test_normality(df, dep_var, ind_vars)
        significants = [p for p in normality_p if p < 0.01]
        is_non_normal = len(significants) > 0

    iv_combinations = []

    for iv in ind_vars:
        for iv1 in ind_vars:
            if (iv != iv1) and ((iv, iv1) not in iv_combinations) and ((iv1, iv) not in iv_combinations):
                iv_combinations.append((iv, iv1))

    for comb in iv_combinations:
        x = df.loc[df['Condition'] == comb[0]][dep_var]
        y = df.loc[df['Condition'] == comb[1]][dep_var]

        try:
            if is_non_normal:
                # s, p = wilcoxon(x, y)
                results = pg.wilcoxon(x, y, 'one-sided')
                results = results.round(4)

                t = list(results['W-val'])[0]
                p = list(results['p-val'])[0]

                prefix = '   '

                if p < .05:
                    prefix = '*  '
                if p < .01:
                    prefix = '** '
                if p < .001:
                    prefix = '***'

                print(f'{prefix}{comb} Wilco: W={round(t, 2)}, p={round(p, 3)}')
            else:
                results = pg.ttest(x, y, paired=True, tail='one-sided')
                results = results.round(4)

                t = list(results['T'])[0]
                p = list(results['p-val'])[0]

                prefix = '   '

                if p < .05:
                    prefix = '*  '
                if p < .01:
                    prefix = '** '
                if p < .001:
                    prefix = '***'

                print(f'{prefix}{comb} Ttest: t={round(t, 2)}, p={round(p, 3)}')

        except Exception as e:
            print(f'Error in {comb}: {e}')

    return


def test_friedman(df, ind_var, dep_var, is_non_normal=None):
    print(f'\n{dep_var}:')
    test_df = pd.DataFrame()

    if is_non_normal == None:
        normality_p = test_normality(df, dep_var, list(df['Condition'].unique()))
        significants = [p for p in normality_p if p < 0.01]
        is_non_normal = len(significants) > 0

        sphericity_p = test_sphericity(df, dep_var, ind_var)

    for iv in list(df[ind_var].unique()):
        df_iv = df.loc[df[ind_var] == iv]

        dv = list(df_iv[dep_var])
        test_df[f'{dep_var} {iv}'] = dv
        print(f'{iv}: mean={round(np.mean(dv), 2)}, SD={round(np.std(dv), 2)}')

    if not is_non_normal and sphericity_p:

        print('\nRM ANOVA')
        results = pg.rm_anova(data=df, dv=dep_var, within=ind_var, subject='ID', correction=False, detailed=True)
        results = results.round(4)
        print(results)

    else:
        print('\nFriedman test')
        results = pg.friedman(data=df, dv=dep_var, within=ind_var, subject='ID')

        X2 = list(results['Q'])[0]
        N = len(list(df['ID'].unique()))
        k = len(list(df[ind_var].unique()))
        kendall_w = X2 / (N * (k - 1))

        results['Kendall'] = [kendall_w]

        results = results.round(3)
        print(results)


def compute_se(x, y):
    squared_error = (np.mean(x) - np.mean(y)) ** 2

    mms = MinMaxScaler()
    x = mms.fit_transform(np.array(x).reshape(-1, 1))
    y = mms.transform(np.array(y).reshape(-1, 1))

    norm_squared_error = (np.mean(x) - np.mean(y)) ** 2

    return squared_error, norm_squared_error


def scatterplot_fixations(data, x, y, title: str, plot_line=False, save=True, savestr: str = ''):
    # Plot fixations
    plt.figure()
    sns.scatterplot(x, y, data=data)
    if plot_line:
        sns.lineplot(x, y, data=data, sort=False)
    plt.xlim((0, 2560))
    plt.ylim((1440, 0))  # Note that the y-axis needs to be flipped
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.title(title)
    if save:
        plt.savefig(savestr, dpi=500)
    plt.show()
