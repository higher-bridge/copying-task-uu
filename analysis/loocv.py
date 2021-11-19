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

from pathlib import Path
import pickle
import time
from typing import Dict, List

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut

from cimulate_batch import simulate_batch
import constants_analysis as constants
from find_saccade_slopes import fit_movement_regression, get_saccades
import helperfunctions as hf


def get_linear_models(IDs: List[str], files: List[Path], pixels_or_fitts: str = 'pixels') -> pd.DataFrame:
    try:
        movements_ = Parallel(n_jobs=4, backend='loky', verbose=True)(
            delayed(get_saccades)(ID, f) for ID, f in zip(IDs, files))
    except:
        print(f'Failed multiprocessing for linear models. Attempting single core...')
        movements_ = [get_saccades(ID, file) for ID, file in zip(IDs, files)]

    movements = pd.concat(movements_)

    conditions, intercepts, coefs, r2s, ps = [], [], [], [], []
    for condition in sorted(list(movements['Condition'].unique())):
        intercept, coef, r_squared, p = fit_movement_regression(movements,
                                                                condition,
                                                                X=f'Distance ({pixels_or_fitts})',
                                                                x_limit=1800,
                                                                plot=False)

        conditions.append(condition)
        intercepts.append(intercept)
        coefs.append(coef)
        r2s.append(r_squared)
        ps.append(p)

    lm_results = pd.DataFrame()
    lm_results['Condition'] = conditions
    lm_results['Intercept'] = intercepts
    lm_results['Coefficient'] = coefs
    lm_results['R-squared'] = r2s
    lm_results['p'] = ps

    return lm_results


def save_simulation_results(results: List[Dict], IDs: List[int], cv_run: int) -> None:
    try:
        pickle.dump(results, open(f'../results/simulations/simulation_cv_{cv_run}.p', 'wb'))
    except Exception as e:
        print(f'Could not save pickle for CV run {cv_run}: {e}')

    for ID, tracking_dict in zip(IDs, results):
        try:
            df = pd.DataFrame(tracking_dict)
            df.to_csv(f'../results/simulations/simulation_cv_{cv_run}_ID_{ID}.csv')
        except Exception as e:
            print(f'Could not save dataframe for CV run {cv_run}, ID {ID}: {e}')


def simulate_one_cv(lm_eye: pd.DataFrame, lm_mouse: pd.DataFrame, cv_run: int) -> None:
    start = time.time()
    IDs = list(np.arange(1, constants.NUM_JOBS_SIM + 1))

    lin_models_eye = [lm_eye] * len(IDs)
    lin_models_mouse = [lm_mouse] * len(IDs)
    save_results = [False] * len(IDs)

    results = Parallel(n_jobs=4, backend='loky', verbose=True)(delayed(simulate_batch)
                                                               (ID, lm_s, lm_m, r) for
                                                               ID, lm_s, lm_m, r in
                                                               zip(IDs, lin_models_eye, lin_models_mouse, save_results))

    # results = []
    # for ID in IDs:
    #     results.append(simulate_batch(ID, lm_eye, lm_mouse, False))
    #     print(f'Simulated {ID} of {len(IDs)} in CV run {cv_run}')

    save_simulation_results(results, IDs, cv_run)

    print(f'CV run {cv_run} took {round((time.time() - start) / 60, 1)} minutes')


def main() -> None:
    start = time.time()

    # Create outer loop which takes a list of all participants and creates a LOOCV grid
    pp_info = pd.read_excel(Path(constants.base_location) / 'participant_info.xlsx')
    ID_list = sorted(list(pp_info['ID'].unique()))
    print(f'Running LOOCV simulations with the following {len(ID_list)} IDs: \n{ID_list}\n')

    loo = LeaveOneOut().split(ID_list)

    cv_run = 1
    for train_indices, test_indices in loo:
        train_IDs = [ID_list[x] for x in train_indices]
        test_IDs = [ID_list[x] for x in test_indices]
        # print(f'Training with {train_IDs}, testing on {test_IDs}')

        # Fit the linear slopes (saccades/mouse) to train set
        try:
            fixation_files = [Path(constants.base_location) / f'{ID}/{ID}-allFixations.csv' for ID in train_IDs]
            mouse_files = [Path(constants.base_location) / f'{ID}/{ID}-mouseEvents.csv' for ID in train_IDs]
            # TODO: add error rates
        except Exception as e:
            raise e

        print(f'Fitting linear models on train set')
        lm_eye = get_linear_models(train_IDs, fixation_files, 'pixels')
        lm_mouse = get_linear_models(train_IDs, mouse_files, 'fitts')

        # Run the simulation
        print(f'Simulating CV run {cv_run} out of {len(list(loo))}...')
        simulate_one_cv(lm_eye, lm_mouse, cv_run)

        # Get and log test statistic of simulation fit to test participant

        cv_run += 1

    print(f'Total duration: {round((time.time() - start) / 60, 1)} minutes')


main()
