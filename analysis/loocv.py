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
from cimulation_analysis_helper import parse_results
import constants_analysis as constants
from find_saccade_slopes import fit_movement_regression, get_saccades
import helperfunctions as hf


def get_linear_models(IDs: List[str], files: List[Path],
                      pixels_or_fitts: str = 'pixels', n_jobs: int = 4) -> pd.DataFrame:
    try:
        movements_ = Parallel(n_jobs=n_jobs, backend='loky', verbose=True)(delayed(get_saccades)
                                                                           (ID, f) for
                                                                           ID, f in
                                                                           zip(IDs, files))
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


def save_simulation_results(results: List[Dict], sim_IDs: List[int], cv_run: int) -> None:
    # Save entire Python object in one pickle
    try:
        pickle.dump(results, open(Path(constants.base_location) / 'simulations' / f'simulation_cv_{cv_run}.p', 'wb'))
    except Exception as e:
        print(f'Could not save pickle for CV run {cv_run}: {e}')

    # Save data for each simulation separately
    for sim_ID, tracking_dict in zip(sim_IDs, results):
        try:
            df = pd.DataFrame(tracking_dict)
            df.to_csv(Path(constants.base_location) / 'simulations'/ f'simulation_cv_{cv_run}_sim_ID_{sim_ID}.csv')
        except Exception as e:
            print(f'Could not save dataframe for CV run {cv_run}, ID {sim_ID}: {e}')


def simulate_one_cv(lm_eye: pd.DataFrame, lm_mouse: pd.DataFrame, cv_run: int, n_jobs: int = 4) -> List[Dict]:
    start = time.time()
    sim_IDs = list(np.arange(1, constants.NUM_SIMS + 1))

    lin_models_eye = [lm_eye] * len(sim_IDs)
    lin_models_mouse = [lm_mouse] * len(sim_IDs)
    save_results = [False] * len(sim_IDs)

    try:
        results = Parallel(n_jobs=n_jobs, backend='loky', verbose=True)(delayed(simulate_batch)
                                                                        (sim_ID, lm_s, lm_m, r) for
                                                                        sim_ID, lm_s, lm_m, r in
                                                                        zip(sim_IDs,
                                                                            lin_models_eye,
                                                                            lin_models_mouse,
                                                                            save_results))
    except:
        print(f'Failed multiprocessing for single CV simulation. Attempting single core...')
        results = []
        for sim_ID in sim_IDs:
            results.append(simulate_batch(sim_ID, lm_eye, lm_mouse, False))
            print(f'Simulated {sim_ID} of {len(sim_IDs)} in CV run {cv_run}')

    save_simulation_results(results, sim_IDs, cv_run)
    print(f'CV run {cv_run} took {round((time.time() - start) / 60, 1)} minutes')

    return results


def save_analysis_results(r: List[Dict], cv_run: int) -> None:
    result_dfs = [pd.DataFrame(d) for d in r]
    results = pd.concat(result_dfs)

    results.to_csv(Path(constants.base_location) / 'simulations' / f'simulation_analysis_CV_{cv_run}.csv')

    results = results.sort_values(by=['Mean Scaled RMSE'], ignore_index=True, kind='stable', ascending=True)
    print(results.head(5))


def get_simulation_fit(test_IDs: List[str], sim_data: List[Dict], observed_data: pd.DataFrame, cv_run: int,
                       n_jobs: int = 4) -> List[Dict]:
    start = time.time()

    # Filter observed data for only the test ID
    observed_data_filtered = pd.DataFrame()
    for ID in test_IDs:
        df_id = observed_data.loc[observed_data['ID'] == ID]
        observed_data_filtered = observed_data_filtered.append(df_id, ignore_index=True)

    # Pre-filter for the different encoding schemes here so we can analyze them in parallel
    sim_data_merged = pd.concat([pd.DataFrame(x) for x in sim_data], ignore_index=True)
    all_encoding_schemes = sorted(list(sim_data_merged['Encoding scheme'].unique()))

    sim_data_schemes = [sim_data_merged.loc[sim_data_merged['Encoding scheme'] == scheme] for scheme in all_encoding_schemes]
    features_repeated = [constants.FIT_FEATURES] * len(all_encoding_schemes)
    observed_data_repeated = [observed_data_filtered] * len(all_encoding_schemes)

    try:
        results = Parallel(n_jobs=n_jobs, backend='loky', verbose=True)(delayed(parse_results)
                                                                        (sim, odr, s, f) for
                                                                        sim, odr, s, f in
                                                                        zip(sim_data_schemes,
                                                                            observed_data_repeated,
                                                                            all_encoding_schemes,
                                                                            features_repeated))
    except:
        print(f'Failed multiprocessing for single CV analysis. Attempting single core...')
        results = []
        for scheme in all_encoding_schemes:
            sim_data_scheme = sim_data_merged.loc[sim_data_merged['Encoding scheme'] == scheme]
            results.append(parse_results(sim_data_scheme, observed_data_filtered, scheme, constants.FIT_FEATURES))

    save_analysis_results(results, cv_run)
    print(f'Analysis of CV run {cv_run} took {round((time.time() - start) / 60, 1)} minutes')

    return results


def combine_cv_results(results: [List[List[Dict]]]) -> None:
    # First, merge all dicts into one big dataframe. 'results' contains a dict for each simulation, within each CV run
    big_df = []
    for cv in results:
        cv_df = pd.concat([pd.DataFrame(x) for x in cv], ignore_index=True)
        big_df.append(cv_df)

    big_df = pd.concat(big_df, ignore_index=True)

    # Compute average outcome measure for each parameter combination
    grouped_df = big_df.groupby(by=['Encoding scheme', 'Repetitions', 'Parameters']).agg(np.mean).reset_index()
    grouped_df = grouped_df.sort_values(by=['Mean Scaled RMSE'], ignore_index=True, kind='stable', ascending=True)
    grouped_df.to_csv(Path(constants.base_location) / 'simulation_analysis_grouped.csv')

    print(grouped_df.head(5))


def main() -> None:
    start = time.time()

    # Load observational data
    observed_data = pd.read_csv(Path(constants.base_location) / 'results-grouped-ID-condition.csv')
    observed_data['ID'] = observed_data['ID'].astype(str)

    # Create outer loop which takes a list of all participants and creates a LOOCV grid
    pp_info = pd.read_excel(Path(constants.base_location) / 'participant_info.xlsx')
    ID_list = sorted(list(pp_info['ID'].astype(str).unique()))

    ###
    # ID_list.remove('1003')
    ###

    print(f'Running Leave-One-Out Cross-Validation simulations with the following {len(ID_list)} IDs: \n{ID_list}\n')

    loo = list(LeaveOneOut().split(ID_list))
    results = []

    for cv_run, (train_indices, test_indices) in enumerate(loo):
        train_IDs = [ID_list[x] for x in train_indices]
        test_IDs = [ID_list[x] for x in test_indices]
        print(f'\nTraining with {train_IDs}, testing on {test_IDs}')

        # Fit the linear slopes (saccades/mouse) to train set
        try:
            fixation_files = [Path(constants.base_location) / f'{ID}/{ID}-allFixations.csv' for ID in train_IDs]
            mouse_files = [Path(constants.base_location) / f'{ID}/{ID}-mouseEvents.csv' for ID in train_IDs]
            # TODO: add error rates
            # TODO: change memory parameter selection
        except Exception as e:
            raise e

        print(f'Fitting linear models on train set (CV {cv_run + 1})...')
        lm_eye = get_linear_models(train_IDs, fixation_files, 'pixels', n_jobs=constants.NUM_JOBS_SIM)
        lm_mouse = get_linear_models(train_IDs, mouse_files, 'fitts', n_jobs=constants.NUM_JOBS_SIM)

        # Run the simulation
        print(f'Simulating CV run {cv_run + 1} out of {len(loo)}...')
        cv_results = simulate_one_cv(lm_eye, lm_mouse, cv_run + 1, n_jobs=-5)

        # Get and log test statistic of simulation fit to test participant
        print(f'Analyzing CV run {cv_run + 1} out of {len(loo)}...')
        results.append(get_simulation_fit(test_IDs, cv_results, observed_data, cv_run + 1,
                                          n_jobs=constants.NUM_JOBS_SIM))

    # Compute the overall CV scores
    combine_cv_results(results)

    print(f'Total duration: {round((time.time() - start) / 60, 1)} minutes')


main()
