#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 12:02:14 2020

@author: alexos
"""

import numpy as np
import pandas as pd
import random
import time
import os
import pickle

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from joblib import Parallel, delayed

import simulation_helper as sh
import helperfunctions as hf
import constants


exp_data = pd.read_csv('../results/results-grouped-ID-condition.csv')
exp_data = exp_data.drop(['Unnamed: 0'], axis=1)

features = [
            'Number of crossings',
            'Completion time (s)',
            'Fixations per second'
            ]

start = time.time()


# Import data
sim_location = '../results/simulations'

files = []
for f in os.listdir(sim_location):
    if 'results.p' in f:
        loc = f'{sim_location}/{f}'
        p = pickle.load(open(loc, 'rb'))
        df = pd.DataFrame(p)
        files.append(df)
        del(p, df)

sim_data = pd.concat(files, ignore_index=True)
del(files)

print(f'Loaded files ({round(time.time() - start, 1)} s)')


PARSE_RESULTS = True

if PARSE_RESULTS:
    results_dict = {key: [] for key in ['Encoding scheme', 'Repetitions', 'Parameters', 
                                        'Mean RMSE', 'Mean MSE', 'Scaled Mean RMSE',
                                        'Crossings', 'Time', 'Fixations']}
    
    # Loop through scheme and repetition strategies
    all_schemes = list(sim_data['Encoding scheme'].unique())
    
    for i, scheme in enumerate(all_schemes):
        scheme_start = time.time()
        sim_data_s = sim_data.loc[sim_data['Encoding scheme'] == scheme]
        
        for repetitions in list(sim_data_s['Repetitions'].unique()):
            sim_data_r = sim_data_s.loc[sim_data_s['Repetitions'] == repetitions]
            
            for params in list(sim_data_r['Parameters'].unique()):
                sim_data_p = sim_data_r.loc[sim_data_r['Parameters'] == params]
    
                squared_errors = {key: [] for key in features}
                norm_squared_errors = {key: [] for key in features}
                for condition in sorted(list(exp_data['Condition'].unique())):
                    exp_data_c = exp_data.loc[exp_data['Condition'] == condition]
                    sim_data_c = sim_data_p.loc[sim_data_p['Condition'] == condition]    
                    
                    sim_grouped = sim_data_c.groupby(['ID', 'Condition']).agg({f: ['median'] for f in features}).reset_index()
                    sim_grouped.columns = sim_grouped.columns.get_level_values(0)
                    
                    # For every feature, perform t-test for data in this condition,
                    # and calculate squared difference between the means in this condition
                    t_vals, p_vals, se_vals = [], [], []
                    for feat in features:
                        x = list(exp_data_c[feat])
                        y = list(sim_grouped[feat])
                        
                        se, nse = hf.compute_se(x, y)
                        
                        squared_errors[feat].append(se)
                        norm_squared_errors[feat].append(nse)
                        
        
                # After calculating statistics for each condition, calculate the RMSE for each feature
                all_rmse = [np.sqrt(np.mean(squared_errors[feat])) for feat in features]
                all_mse = [np.mean(squared_errors[feat]) for feat in features]
                all_norm_rmse = [np.sqrt(np.mean(norm_squared_errors[feat])) for feat in features]
                            
                results_dict['Encoding scheme'].append(scheme)
                results_dict['Repetitions'].append(repetitions)
                results_dict['Parameters'].append(params)
                results_dict['Mean RMSE'].append(np.mean(all_rmse))
                results_dict['Mean MSE'].append(np.mean(all_mse))
                results_dict['Scaled Mean RMSE'].append(np.mean(all_norm_rmse))
                results_dict['Crossings'].append(all_norm_rmse[0])
                results_dict['Time'].append(all_norm_rmse[1])
                results_dict['Fixations'].append(all_norm_rmse[2])
                
                
        print(f'Processed {i + 1} of {len(all_schemes)} schemes ({round(time.time() - scheme_start, 1)} s)')
    
    results = pd.DataFrame(results_dict)
    results = results.sort_values(by=['Scaled Mean RMSE'], ignore_index=True, kind='mergesort', ascending=True)
    results.to_csv('../results/simulation_analysis.csv')
    
    print(f'Analysis took {round((time.time() - start) / 60, 1)} minutes')


else:
    results = pd.read_csv('../results/simulation_analysis.csv')
    results = results.drop(['Unnamed: 0'], axis=1)


print(results.head(3))
print(results.tail(3))

best_scheme = results.iloc[0]['Encoding scheme']
best_reps = results.iloc[0]['Repetitions']
best_params = results.iloc[0]['Parameters']

best_results = sim_data.loc[sim_data['Encoding scheme'] == best_scheme]
best_results = best_results.loc[best_results['Repetitions'] == best_reps]
best_results = best_results.loc[best_results['Parameters'] == best_params]

# Group by ID and Condition, use median
results_grouped = best_results.groupby(['ID', 'Condition']).agg({f: ['median'] for f in features}).reset_index()
results_grouped.columns = results_grouped.columns.get_level_values(0)

# =============================================================================
# COMBINED BOXPLOTS                
# =============================================================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times']
rcParams['font.size'] = 12

# y_lims = [(1.5, 7.5), (4, 15), (2.5, 5)]

# sp = [f'13{x}' for x in range(1, len(features) + 1)]

# f = plt.figure(figsize=(7.5, 5))
# axes = [f.add_subplot(s) for s in sp]

# for i, feat in enumerate(features):
#     sns.boxplot(x='Condition', y=feat, data=results_grouped, #capsize=.1, errwidth=1.5,
#                 palette='Blues', ax=axes[i])
#     axes[i].set_xlabel('')   

#     if i == 1:
#         axes[i].set_xlabel('Condition')
    
#     try:
#         axes[i].set_ylim(y_lims[i])
#     except:
#         pass

# plt.suptitle(f'Results for model with scheme {best_scheme}, repetitions {best_reps}, params {best_params}')
# f.tight_layout() #(pad=1, w_pad=0.2)
# f.savefig('../results/plots/model-boxplots.png', dpi=500)
# plt.show()
        
        
# =============================================================================
# COMBINE BARPLOTS
# =============================================================================
results_grouped['Source'] = ['Model'] * len(results_grouped)
exp_data['Source'] = ['Observed'] * len(exp_data)

exp_data = exp_data.drop([c for c in list(exp_data.columns) if c not in list(results_grouped.columns)], axis=1)

combined_data = pd.concat([exp_data, results_grouped], ignore_index=True)

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

    if i == 1:
        axes[i].set_xlabel('Condition')
    
    if i != 2:
        axes[i].get_legend().remove()
        

# plt.suptitle(f'Results for model with scheme {best_scheme}, repetitions {best_reps}, params {best_params}')
f.tight_layout() #(pad=1, w_pad=0.2)
f.savefig('../results/plots/model-barplots.png', dpi=500)
plt.show()




# =============================================================================
# Compare models 
# =============================================================================
best_rmse = results.iloc[0]['Scaled Mean RMSE']
other_results = results.loc[results['Parameters'] == best_params]
other_rmse = np.mean(other_results['Scaled Mean RMSE'])
other_rmse_sd = np.std(other_results['Scaled Mean RMSE'])

print(f'Best: {best_rmse}, other: {other_rmse} (SD={other_rmse_sd})')


best_results = sim_data.loc[sim_data['Parameters'] == best_params]

# Group by ID and Condition, use median
results_grouped = best_results.groupby(['ID', 'Condition']).agg({f: ['median'] for f in features}).reset_index()
results_grouped.columns = results_grouped.columns.get_level_values(0)
             
results_grouped['Source'] = ['Model'] * len(results_grouped)
exp_data['Source'] = ['Observed'] * len(exp_data)

exp_data = exp_data.drop([c for c in list(exp_data.columns) if c not in list(results_grouped.columns)], axis=1)

combined_data = pd.concat([exp_data, results_grouped], ignore_index=True)

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
        axes[i].set_xlabel('Condition')
    
    if i != 0:
        axes[i].get_legend().remove()
        

plt.suptitle(f'Mean of all models with params {best_params}')
f.tight_layout() #(pad=1, w_pad=0.2)
# f.savefig('../results/plots/model-barplots.png', dpi=500)
plt.show()