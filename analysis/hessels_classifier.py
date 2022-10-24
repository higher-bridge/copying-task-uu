"""
visual_search
Copyright (C) 2022 Utrecht University, Alex Hoogerbrugge

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

------------------------------
Python implementation of slow/fast phase classifier as in Roy S. Hessels, Andrea J. van Doorn, Jeroen S. Benjamins,
Gijs A. Holleman & Ignace T. C. Hooge (2020). Task-related gaze control in human crowd navigation.
Attention, Perception, & Psychophysics 82, pp. 2482–2501. doi: 10.3758/s13414-019-01952-9

Original Matlab implementation (and remaining documentation) can be found at:
https://github.com/dcnieho/GlassesViewer/tree/master/user_functions/HesselsEtAl2020
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from constants_analysis import (HESSELS_LAMBDA, HESSELS_MAX_ITER,
                                HESSELS_MIN_AMP, HESSELS_MIN_FIX,
                                HESSELS_SAVGOL_LEN, HESSELS_THR,
                                HESSELS_WINDOW_SIZE, PX2DEG, SAMPLING_RATE,
                                STEP_SIZE)
from scipy.signal import savgol_filter


### STATISTICS FUNCTIONS ###
def _get_amplitudes(start_x: np.ndarray, start_y: np.ndarray,
                    end_x: np.ndarray, end_y: np.ndarray,
                    to_dva: bool = True) -> np.ndarray:
    amps = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

    if to_dva:
        amps *= PX2DEG

    return amps.round(3)


def _get_starts_ends(x: np.ndarray, y: np.ndarray, imarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                                                np.ndarray, np.ndarray,
                                                                                np.ndarray, np.ndarray]:
    start_x, end_x = [], []
    start_y, end_y = [], []
    avg_x, avg_y = [], []
    for start, end in zip(imarks[:-1], imarks[1:]):
        start_x.append(x[start])
        end_x.append(x[end])
        start_y.append(y[start])
        end_y.append(y[end])
        avg_x.append(np.nanmean(x[start:end]))
        avg_y.append(np.nanmean(y[start:end]))

    start_x.append(np.nan)
    end_x.append(np.nan)
    start_y.append(np.nan)
    end_y.append(np.nan)
    avg_x.append(np.nan)
    avg_y.append(np.nan)

    return np.array(start_x).round(2), np.array(start_y).round(2), \
           np.array(end_x).round(2), np.array(end_y).round(2), \
           np.array(avg_x).round(2), np.array(avg_y).round(2)


def _get_durations(smarks: np.ndarray) -> np.ndarray:
    durations = np.full(len(smarks), np.nan, dtype=float)
    durations[:-1] = smarks[1:] - smarks[:-1]
    return durations.round(3)


def _get_velocities(vel: np.ndarray, imarks: np.ndarray, stat: str, to_dva: bool = True) -> np.ndarray:
    velocities = []
    for start, end in zip(imarks[:-1], imarks[1:]):
        vels = vel[start:end]

        if stat == 'peak':
            velocities.append(np.max(vels))
        elif stat == 'mean':
            velocities.append(np.nanmean(vels))
        elif stat == 'median':
            velocities.append(np.nanmedian(vels))
        else:
            velocities.append(np.nan)
            raise UserWarning(f'Cannot compute {stat} of velocities')

    velocities.append(np.nan)
    velocities = np.array(velocities)

    if to_dva:
        velocities *= PX2DEG

    velocities *= SAMPLING_RATE

    return velocities.round(3)


### CLASSIFIER IMPLEMENTATION ###
def detect_velocity(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    :param p: array of position over time (x or y)
    :param t: array of timestamps
    :return:
    """
    delta_time_1 = t[1:-1] - t[0:-2]
    delta_pos_1 = p[1:-1] - p[0:-2]
    delta_time_2 = t[2:] - t[1:-1]
    delta_pos_2 = p[2:] - p[1:-1]

    # Compute velocities
    vel = ((delta_pos_1 / delta_time_1) + (delta_pos_2 / delta_time_2)) / 2

    # Initialize array of nan's and fill all (except first and last value) with the computed velocities
    velocities = np.full(len(delta_pos_1) + 2, np.nan, dtype=float)
    velocities[1:-1] = vel

    return velocities


def detect_velocity_python(x: np.ndarray, y: np.ndarray, ts: np.ndarray) -> np.ndarray:
    # Assert that measurement rate is constant.
    delta_time = ts[1:] - ts[:-1]
    sd = np.nanstd(delta_time)
    se = sd / np.sqrt(len(delta_time) + 1)  # Add 1 just in case we would otherwise divide by zero
    max_delta = np.max(delta_time).ravel()[0]

    assert se <= .1, f'Use detect_velocity() for non-constant measurement rates! (se = {se.round(3)})'
    assert max_delta < 5, f'Use detect_velocity() for non-constant measurement rates! (max_delta = {max_delta})'

    # Compile x/y arrays into tuples
    locs = [(x_, y_) for x_, y_ in zip(x, y)]

    # Create empty array with velocities
    vel = np.full(len(locs), np.nan, dtype=float)

    # Loop through locations + staggered locations (delta +1) and get the euclidean distance.
    # Since time delta is constant we don't need to take that into account
    for i, (xy, xyd) in enumerate(zip(locs[:-1], locs[1:])):
        d = [(loc1 - loc2) ** 2 for loc1, loc2 in zip(xy, xyd)]  # Get squared x and y deltas
        dist = np.sqrt(sum(d))  # sqrt of sum of distances (bonus trick: works in n dimensions)
        vel[i + 1] = (round(dist, 3))  # Measurement rate is constant, so distance == velocity

    return vel


def detect_switches(qvel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.full(len(qvel) + 2, False, dtype=bool)
    v[1:-1] = qvel
    v = v.astype(int)

    v0 = v[:-1]
    v1 = v[1:]
    switches = v0 - v1

    # If False - True (0 - 1): switch_on, if True - False (1 - 0): switch_off
    switch_on = np.argwhere(switches == -1)
    switch_off = np.argwhere(switches == 1) - 1  # Subtract 1 from each element

    return switch_on.ravel(), switch_off.ravel()


def merge_fix_candidates(idxs: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # idxs contains only starts, with slow/fast alternating
    fast_starts = idxs[1:-1:2]  # Take 1st element (0th is slow phase) and keep skipping two
    fast_ends = idxs[2::2] - 1  # Stagger by 1 so we get the start indices of the next slow phase, and remove 1 from all

    # Loop through start/end indices of the fast phases and determine if amplitude is too low
    remove_from_idxs = []
    for s, e in zip(fast_starts, fast_ends):
        amp = _get_amplitudes(x[s], y[s], x[e], y[e], to_dva=True)
        if amp < HESSELS_MIN_AMP:
            remove_from_idxs.append(s)
            remove_from_idxs.append(e + 1)  # We did ends - 1, so now we have to add it back to get a valid index

    mask = [i for i, x in enumerate(idxs) if x in remove_from_idxs]
    keep_idxs = np.delete(idxs, mask)

    return keep_idxs


def fmark(vel: np.ndarray, ts: np.ndarray, thr: np.ndarray) -> np.ndarray:
    qvel = vel < thr

    # Get indices of starts and ends of fixation candidates
    switch_on, switch_off = detect_switches(qvel)
    time_on, time_off = ts[switch_on], ts[switch_off]

    # Get durations of candidates and find at which indices they are long enough. Then select only those.
    time_deltas = time_off - time_on
    qfix = np.argwhere(time_deltas > HESSELS_MIN_FIX)
    time_on = time_on[qfix].ravel()
    time_off = time_off[qfix].ravel()

    # Combine the two lists and sort the timestamps
    times_sorted = sorted(np.concatenate([time_on, time_off]))

    return np.array(times_sorted)


def threshold(vel: np.ndarray) -> np.array:
    # Retrieve where vel is neither below threshold (and not nan), and get indices of those positions
    valid_idxs = np.argwhere(vel < HESSELS_THR).ravel()

    mean_vel = np.nanmean(vel[valid_idxs])
    std_vel = np.nanstd(vel[valid_idxs])

    if np.isnan(mean_vel):
        return np.array([np.nan] * len(vel))

    counter = 0
    prev_thr = HESSELS_THR

    while True:
        thr2 = mean_vel + (HESSELS_LAMBDA * std_vel)
        valid_idxs = np.argwhere(vel < thr2).ravel()

        # Round to several decimals because of our high sampling rate (counter still doesn't exceed ~20 on average)
        if round(thr2, 6) == round(prev_thr, 6) or counter >= HESSELS_MAX_ITER:
            break

        mean_vel = np.nanmean(vel[valid_idxs])
        std_vel = np.nanstd(vel[valid_idxs])
        prev_thr = thr2
        counter += 1

    thr2 = mean_vel + (HESSELS_LAMBDA * std_vel)
    return thr2


def get_blink_indices(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Detect where switches of True <-> False occur and return those indices
    # zeroes = np.isnan(p)
    zeroes = p < 10  # < 10 is no pupil signal
    onsets, offsets = detect_switches(zeroes)

    return onsets, offsets


def insert_blinks(p: np.ndarray, ts: np.ndarray, imarks: np.ndarray, phase_types: List[str]) -> Tuple[np.ndarray,
                                                                                                      np.ndarray,
                                                                                                      np.ndarray]:
    onsets, offsets = get_blink_indices(p)

    # If no blinks are detected, return the original data
    if len(onsets) == 0:
        smarks = ts[imarks.astype(int)]
        return smarks, imarks, np.array(phase_types)

    # Create new list with length of p
    new_imarks, new_phase_types = np.zeros(len(p), dtype=int), np.array(['Empty'] * len(p))

    # Fill these lists with the corresponding imark and phase type as repeating values
    # (e.g. len(p) = 200k and imarks = [1, 320, ...]; fill new_imarks[1:320] with 1
    for i in range(len(imarks) - 1):
        idxs = np.arange(imarks[i], imarks[i + 1] - 1)
        new_imarks[idxs] = imarks[i]
        new_phase_types[idxs] = phase_types[i]

    # IF there are blinks, fill new lists between on- and offsets
    for on, off in zip(onsets, offsets):
        new_imarks[on:off] = on
        new_phase_types[on:off] = 'BLINK'

    # Detect starting indices of each phase type, and put them together in a sorted list
    fix_onsets, _ = detect_switches(new_phase_types == 'FIXA')
    sacc_onsets, _ = detect_switches(new_phase_types == 'SACC')
    blink_onsets, _ = detect_switches(new_phase_types == 'BLINK')
    imarks = np.array(sorted(np.concatenate([fix_onsets, sacc_onsets, blink_onsets]))).astype(int)

    # For each onset index, grab the appropriate phase type and timestamp
    phase_types = new_phase_types[imarks]
    smarks = ts[imarks]

    return smarks, imarks, phase_types


def classify_hessels2020(df: pd.DataFrame,
                         ID: str,
                         verbose: bool = True) -> pd.DataFrame:
    """
    :param df: pd.DataFrame with all gaze data for one participant, one condition
    :param ID:
    :param condition:
    :param verbose: Prints df.head() after detection is performed
    :return:
    """

    np.seterr(divide='ignore', invalid='ignore')

    results = {'ID': [],
               'label': [],
               'onset': [],
               'duration': [],
               'offset': [],
               'amp': [],
               'avg_vel': [],
               'med_vel': [],
               'peak_vel': [],
               'start_x': [],
               'start_y': [],
               'end_x': [],
               'end_y': [],
               'avg_x': [],
               'avg_y': []
               }

    try:
        x_before = np.array(df['x'])
        y_before = np.array(df['y'])
        ts = np.array(df['timestamp'])
        p = np.array(df['pupilsize'])

        x = savgol_filter(x_before, HESSELS_SAVGOL_LEN, 2, mode='nearest')
        y = savgol_filter(y_before, HESSELS_SAVGOL_LEN, 2, mode='nearest')

        print('ID', ID, ':', len(x), 'rows of samples')

        # vel = detect_velocity_python(x, y, ts)
        vx = detect_velocity(x, ts)
        vy = detect_velocity(y, ts)
        vel = np.sqrt(vx ** 2 + vy ** 2)

        window_size = int(HESSELS_WINDOW_SIZE)
        last_window_start_idx = len(ts) - window_size

        thr, ninwin = np.zeros(len(ts)), np.zeros(len(ts))

        start_indices = list(np.arange(0, last_window_start_idx, step=STEP_SIZE))
        for i in start_indices:
            idxs = np.arange(i, i + window_size + 1)

            window_thr = threshold(vel[idxs])
            thr[idxs] += window_thr
            ninwin[idxs] += 1

            if verbose and i % int(last_window_start_idx / 20) == 0:
                print(f'Processed {i} of {len(start_indices)} threshold windows '
                      f'({round((i / len(start_indices)) * 100)}%)', end='\r')

        thr /= ninwin

        # Get slow events
        emarks = fmark(vel, ts, thr)

        # Get indices of timestamps if they are in emarks
        imarks = [i for i, t in enumerate(ts) if
                  t in emarks]  # Get index of timestamp if that ts is found in emarks
        assert len(emarks) == len(imarks), 'Not all output samples have a corresponding input time!'

        starts, ends = np.array(imarks[::2]), np.array(imarks[1::2])
        imarks = np.array(sorted(np.concatenate([starts, ends + 1])))

        imarks = merge_fix_candidates(imarks, x, y)

        # Alternate slow and fast phases
        phase_types = []
        for i, _ in enumerate(imarks):
            if i % 2 == 0:
                phase_types.append('FIXA')
            else:
                phase_types.append('SACC')

        smarks, imarks, phase_types = insert_blinks(p, ts, imarks, phase_types)

        # Check for too much missing data
        for i in range(len(imarks) - 1):
            idxs = np.arange(imarks[i] + 1, imarks[i + 1] - 2)
            nans = np.sum(np.isnan(x[idxs]))
            if nans >= (len(idxs) * 0.5) and phase_types[i] != 'BLINK':
                phase_types[i] = 'LOSS'

        # Retrieve some extra information from the data
        start_x, start_y, end_x, end_y, avg_x, avg_y, = _get_starts_ends(x, y, imarks)

        for p, pt in enumerate(phase_types):
            results['ID'].append(ID)
            results['label'].append(phase_types[p])
            results['onset'].append(smarks[p])
            results['duration'].append(_get_durations(smarks)[p])
            results['offset'].append(smarks[p] + _get_durations(smarks)[p])
            results['amp'].append(_get_amplitudes(start_x, start_y, end_x, end_y)[p])
            results['avg_vel'].append(_get_velocities(vel, imarks, 'mean')[p])
            results['med_vel'].append(_get_velocities(vel, imarks, 'median')[p])
            results['peak_vel'].append(_get_velocities(vel, imarks, 'peak')[p])
            results['start_x'].append(start_x[p])
            results['start_y'].append(start_y[p])
            results['end_x'].append(end_x[p])
            results['end_y'].append(end_y[p])
            results['avg_x'].append(avg_x[p])
            results['avg_y'].append(avg_y[p])

    except Exception as e:
        print(ID, e)

    results = pd.DataFrame(results)
    # results = results.dropna()

    return results
