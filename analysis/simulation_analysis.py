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
import pickle
import time

import constants_analysis as constants
import helperfunctions as hf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib import rcParams
from sklearn.preprocessing import MinMaxScaler

PARSE_RESULTS = False

def compute_se(x, y):
    squared_error = (np.mean(x) - np.mean(y)) ** 2
    
    mms = MinMaxScaler()
    x = mms.fit_transform(np.array(x).reshape(-1, 1))
    y = mms.transform(np.array(y).reshape(-1, 1))
    
    norm_squared_error = (np.mean(x) - np.mean(y)) ** 2
    
    return squared_error, norm_squared_error

def parse_results(sim_data_s, scheme, features):    
    results_dict = {key: [] for key in ['Encoding scheme', 'Repetitions', 'Parameters', 
                                        'Mean Scaled RMSE',
                                        'Crossings', 'Time', 'Fixations']}
    
    # Loop through params and repetition strategies
    all_repetitions = list(sim_data_s['Repetitions'].unique())
    all_params = list(sim_data_s['Parameters'].unique())
        
    for repetitions in all_repetitions:
        sim_data_r = sim_data_s.loc[sim_data_s['Repetitions'] == repetitions]
        
        for params in all_params:
            sim_data_p = sim_data_r.loc[sim_data_r['Parameters'] == params]

            scaled_squared_errors = {key: [] for key in features}
            
            for condition in constants.CONDITIONS:
                exp_data_c = exp_data.loc[exp_data['Condition'] == condition]
                sim_data_c = sim_data_p.loc[sim_data_p['Condition'] == condition]    
                
                sim_grouped = sim_data_c.groupby(['ID', 'Condition']).agg({f: ['mean'] for f in features}).reset_index()
                sim_grouped.columns = sim_grouped.columns.get_level_values(0)
                
                # For every feature, calculate scaled squared error
                for feat in features:
                    x = list(exp_data_c[feat])
                    y = list(sim_grouped[feat])
                    
                    se, nse = compute_se(x, y)
                    
                    scaled_squared_errors[feat].append(nse)
                 
    
            # After calculating statistics for each condition, calculate the RMSE for each feature
            all_scaled_rmse = [np.sqrt(np.mean(scaled_squared_errors[feat])) for feat in features]
                        
            results_dict['Encoding scheme'].append(scheme)
            results_dict['Repetitions'].append(repetitions)
            results_dict['Parameters'].append(params)
            results_dict['Mean Scaled RMSE'].append(np.mean(all_scaled_rmse))
            results_dict['Crossings'].append(all_scaled_rmse[0])
            results_dict['Time'].append(all_scaled_rmse[1])
            results_dict['Fixations'].append(all_scaled_rmse[2])
                
    return results_dict


if __name__ == '__main__':    
    features = [
                'Number of crossings',
                'Completion time (s)',
                'Fixations per second'
                ]
    
    start = time.time()
        
    # =============================================================================
    # IMPORT DATA
    # =============================================================================
    exp_data = pd.read_csv('../results/results-grouped-ID-condition.csv')
    exp_data = exp_data.drop(['Unnamed: 0'], axis=1)
    
    sim_location = '../results/simulations'
    
    files = []
    for f in sorted(os.listdir(sim_location)):
        if 'results.p' in f: 
            loc = f'{sim_location}/{f}'
            p = pickle.load(open(loc, 'rb'))
            df = pd.DataFrame(p)
            files.append(df)
    
    sim_data = pd.concat(files, ignore_index=True)
    del(files, p, df)
    
    print(f'Loaded files ({round(time.time() - start, 1)} s)')

    # =============================================================================
    # PARSE RESULTS
    # =============================================================================
    
    if PARSE_RESULTS:
        # Run the analysis for each encoding scheme separately. Parallel processing
        # speed it up a lot, but beware that it consumes more RAM
        all_schemes = sorted(list(sim_data['Encoding scheme'].unique()))
        scheme_dfs = [sim_data.loc[sim_data['Encoding scheme'] == scheme] for scheme in all_schemes]
        features_rep = [features] * len(all_schemes)
        
        r = Parallel(n_jobs=7, backend='loky', verbose=True)(delayed(parse_results)(df, scheme, f) \
                                                              for df, scheme, f \
                                                              in zip(scheme_dfs, all_schemes, features_rep))

        # Convert dicts to dataframes, and concatenate them into one df
        result_dfs = [pd.DataFrame(d) for d in r]
        results = pd.concat(result_dfs)
        
        del(scheme_dfs, r, result_dfs)
        
        results = results.sort_values(by=['Mean Scaled RMSE'], ignore_index=True, kind='mergesort', ascending=True)
        results.to_csv('../results/simulation_analysis.csv')
    
    else:
        results = pd.read_csv('../results/simulation_analysis.csv')
        results = results.drop(['Unnamed: 0'], axis=1)


    print(results.head(3))
    print(results.tail(3))    
    
    
    # =============================================================================
    # PROCESS RESULTS
    # =============================================================================
    best_scheme = results.iloc[0]['Encoding scheme']
    best_reps = results.iloc[0]['Repetitions']
    best_params = results.iloc[0]['Parameters']
    
    best_results = sim_data.loc[sim_data['Encoding scheme'] == best_scheme]
    best_results = best_results.loc[best_results['Repetitions'] == best_reps]
    best_results = best_results.loc[best_results['Parameters'] == best_params]
    
    # Group by ID and Condition, use median
    # results_grouped = best_results.groupby(['Condition']).agg({f: ['mean'] for f in features}).reset_index()
    # results_grouped.columns = results_grouped.columns.get_level_values(0)
    
    results_grouped = best_results
    
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times']
    rcParams['font.size'] = 11
    
            
    # =============================================================================
    # COMBINE BARPLOTS
    # =============================================================================
    results_grouped['Source'] = ['Model'] * len(results_grouped)
    exp_data['Source'] = ['Observed'] * len(exp_data)
    
    exp_data = exp_data.drop([c for c in list(exp_data.columns) if c not in list(results_grouped.columns)], axis=1)
    
    combined_data = pd.concat([exp_data, results_grouped], ignore_index=True)
    combined_data['Condition'] = combined_data['Condition'].apply(hf.condition_number_to_name)
    
    sp = [f'13{x}' for x in range(1, len(features) + 1)]
    
    f = plt.figure(figsize=(7.5, 5))
    axes = [f.add_subplot(s) for s in sp]
    # pal = sns.color_palette('Blues')
    
    for i, feat in enumerate(features):
        sns.barplot(x='Condition', y=feat, data=combined_data, hue='Source', 
                    capsize=.1, errwidth=1.5, 
                    palette='Blues',
                    ax=axes[i])
    
        axes[i].set_xlabel('')
        axes[i].set_ylabel(feat, fontsize=12)
    
        if i == 1:
            axes[i].set_xlabel('Reliability of access', fontsize=12)
        
        if i != 2:
            axes[i].get_legend().remove()
            
    f.tight_layout() #(pad=1, w_pad=0.2)
    f.savefig('../results/plots/model-barplots.png', dpi=500)
    plt.show()
    
    
    # =============================================================================
    # Compare models 
    # =============================================================================
    best_rmse = results.iloc[0]['Mean Scaled RMSE']
    other_results = results.loc[results['Parameters'] == best_params]
    other_rmse_len = len(other_results['Mean Scaled RMSE'])
    other_rmse = np.mean(other_results['Mean Scaled RMSE'])
    other_rmse_sd = np.std(other_results['Mean Scaled RMSE'])
    
    print(f'Best: {best_rmse}, other with same params: {other_rmse} (SD={other_rmse_sd}), N={other_rmse_len}')
    
    other_results = other_results.loc[other_results['Repetitions'] == best_reps]
    other_rmse_len = len(other_results['Mean Scaled RMSE'])
    other_rmse = np.mean(other_results['Mean Scaled RMSE'])
    other_rmse_sd = np.std(other_results['Mean Scaled RMSE'])

    print(f'Best: {best_rmse}, other with same params, same reps: {other_rmse} (SD={other_rmse_sd}), N={other_rmse_len}') 
    
    other_results = results.loc[results['Parameters'] == best_params]
    other_results = other_results.loc[other_results['Encoding scheme'] == best_scheme]    
    other_rmse_len = len(other_results['Mean Scaled RMSE'])
    other_rmse = np.mean(other_results['Mean Scaled RMSE'])
    other_rmse_sd = np.std(other_results['Mean Scaled RMSE'])
    
    print(f'Best: {best_rmse}, other with same params, same scheme: {other_rmse} (SD={other_rmse_sd}), N={other_rmse_len}')    
    
    
    best_results = sim_data.loc[sim_data['Parameters'] == best_params]
    
    # Group by ID and Condition, use median
    # results_grouped = best_results.groupby(['ID', 'Condition']).agg({f: ['median'] for f in features}).reset_index()
    # results_grouped.columns = results_grouped.columns.get_level_values(0)

    results_grouped = best_results
                 
    results_grouped['Source'] = ['Model'] * len(results_grouped)
    exp_data['Source'] = ['Observed'] * len(exp_data)
    
    exp_data = exp_data.drop([c for c in list(exp_data.columns) if c not in list(results_grouped.columns)], axis=1)
    
    combined_data = pd.concat([exp_data, results_grouped], ignore_index=True)
    combined_data['Condition'] = combined_data['Condition'].apply(hf.condition_number_to_name)
    
    sp = [f'13{x}' for x in range(1, len(features) + 1)]
    
    f = plt.figure(figsize=(7.5, 5))
    axes = [f.add_subplot(s) for s in sp]
    
    for i, feat in enumerate(features):
        sns.barplot(x='Condition', y=feat, data=combined_data, hue='Source', 
                    capsize=.1, errwidth=1.5, 
                    # palette='Blues',
                    ax=axes[i])
    
        axes[i].set_xlabel('')   
    
        if i == 1:
            axes[i].set_xlabel('Reliability of access')
        
        if i != 0:
            axes[i].get_legend().remove()
            
    
    plt.suptitle(f'Mean of all models with params {best_params}')
    f.tight_layout() #(pad=1, w_pad=0.2)
    f.savefig('../results/plots/model-barplots-overall.png', dpi=500)
    plt.show()

    # =============================================================================
    # PLOT RMSE BY RANK                
    # ============================================================================
    plt.figure(figsize=(3.75, 2.5))
    plt.plot(results['Mean Scaled RMSE'])
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Mean scaled RMSE', fontsize=12)
    plt.tight_layout()
    plt.savefig('../results/plots/model-score-by-rank.png', dpi=500)
    plt.show() 
    
    del(sim_data)
