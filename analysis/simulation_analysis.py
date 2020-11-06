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

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from joblib import Parallel, delayed

import simulation_helper as sh
import helperfunctions as hf
import constants

# Import data
exp_data = pd.read_csv('../results/results-grouped-ID-condition.csv')
sim_data = pd.read_csv('../results/simulation_results.csv')

exp_data = exp_data.drop(['Unnamed: 0'], axis=1)
sim_data = sim_data.drop(['Unnamed: 0'], axis=1)

features = [
            'Number of crossings',
            'Completion time (s)',
            'Fixations per second'
            ]

results_dict = {key: [] for key in ['Encoding scheme', 'Repetitions', 
                                    'Mean RMSE', 'Mean MSE']} #, 't', 'p']}

# Loop through scheme and repetition strategies
for scheme in list(sim_data['Encoding scheme'].unique()):
    sim_data_s = sim_data.loc[sim_data['Encoding scheme'] == scheme]
    
    for repetitions in list(sim_data_s['Repetitions'].unique()):
        sim_data_r = sim_data_s.loc[sim_data_s['Repetitions'] == repetitions]

        squared_errors = {key: [] for key in features}
        for condition in sorted(list(exp_data['Condition'].unique())):
            exp_data_c = exp_data.loc[exp_data['Condition'] == condition]
            sim_data_c = sim_data_r.loc[sim_data_r['Condition'] == condition]    
            
            sim_grouped = sim_data_c.groupby(['ID', 'Condition']).agg({f: ['median'] for f in features}).reset_index()
            sim_grouped.columns = sim_grouped.columns.get_level_values(0)
            
            # For every feature, perform t-test for data in this condition,
            # and calculate squared difference between the means in this condition
            t_vals, p_vals, se_vals = [], [], []
            for feat in features:
                x = list(exp_data_c[feat])
                y = list(sim_grouped[feat])
                
                t, p, se = hf.test_ttest(x, y)
    
                t_vals.append(t)
                p_vals.append(p)
                
                squared_errors[feat].append(se)
                            
            # mean_t = round(np.mean(t_vals), 4)
            # mean_p = round(np.mean(p_vals), 4)

        # After calculating statistics for each condition, calculate the RMSE for each feature
        all_rmse = [np.sqrt(np.mean(squared_errors[feat])) for feat in features]
        all_mse = [np.mean(squared_errors[feat]) for feat in features]
                    
        results_dict['Encoding scheme'].append(scheme)
        results_dict['Repetitions'].append(repetitions)
        results_dict['Mean RMSE'].append(np.mean(all_rmse))
        results_dict['Mean MSE'].append(np.mean(all_mse))
        # results_dict['t'].append(mean_t)
        # results_dict['p'].append(mean_p)
            

results = pd.DataFrame(results_dict)
results = results.sort_values(by=['Mean MSE'], ignore_index=True, kind='mergesort', ascending=True)
print(results.head())

best_scheme = results.iloc[0]['Encoding scheme']
best_reps = results.iloc[0]['Repetitions']
best_results = sim_data.loc[sim_data['Encoding scheme'] == best_scheme]
best_results = best_results.loc[best_results['Repetitions'] == best_reps]

# Group by ID and Condition, use median
results_grouped = best_results.groupby(['ID', 'Condition']).agg({f: ['median'] for f in features}).reset_index()
results_grouped.columns = results_grouped.columns.get_level_values(0)

# =============================================================================
# COMBINED BOXPLOTS                
# =============================================================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times']
rcParams['font.size'] = 12

y_lims = [(1.5, 7.5), (4, 15), (2.5, 5)]

sp = [f'13{x}' for x in range(1, len(features) + 1)]

f = plt.figure(figsize=(7.5, 5))
axes = [f.add_subplot(s) for s in sp]

for i, feat in enumerate(features):
    sns.boxplot(x='Condition', y=feat, data=results_grouped, #capsize=.1, errwidth=1.5,
                palette='Blues', ax=axes[i])
    axes[i].set_xlabel('')   

    if i == 1:
        axes[i].set_xlabel('Condition')
    
    try:
        axes[i].set_ylim(y_lims[i])
    except:
        pass

plt.suptitle(f'Results for model with scheme {best_scheme}, repetitions {best_reps}')
f.tight_layout() #(pad=1, w_pad=0.2)
f.savefig('../results/plots/model-boxplots.png', dpi=500)
plt.show()
        
        
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
        

plt.suptitle(f'Results for model with scheme {best_scheme}, repetitions {best_reps}')
f.tight_layout() #(pad=1, w_pad=0.2)
f.savefig('../results/plots/model-barplots.png', dpi=500)
plt.show()


# =============================================================================
# 
# =============================================================================
# =============================================================================
# WORST MODELS
# =============================================================================
best_scheme = results.iloc[-1]['Encoding scheme']
best_reps = results.iloc[-1]['Repetitions']
best_results = sim_data.loc[sim_data['Encoding scheme'] == best_scheme]
best_results = best_results.loc[best_results['Repetitions'] == best_reps]

# Group by ID and Condition, use median
results_grouped = best_results.groupby(['ID', 'Condition']).agg({f: ['median'] for f in features}).reset_index()
results_grouped.columns = results_grouped.columns.get_level_values(0)
             
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
        

plt.suptitle(f'WORST MODEL with scheme {best_scheme}, repetitions {best_reps}')
f.tight_layout() #(pad=1, w_pad=0.2)
# f.savefig('../results/plots/model-barplots.png', dpi=500)
plt.show()