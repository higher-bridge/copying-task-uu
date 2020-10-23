#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 13:26:03 2020

@author: alexos
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from joblib import Parallel, delayed
from pingouin import linear_regression

import helperfunctions as hf
from constants import EXCLUDE_TRIALS, base_location
from simulation_helper import euclidean_distance

def plot_saccades(results, condition):
    df = results.loc[results['Condition'] == condition]
    df = df.loc[df['Duration'] < 500]
    df = df.loc[df['Distance'] < 1800]
    
    intercept, coef, r_squared, p = get_linear_regression(df)
    print(f'Condition {condition}\
          \nIntercept: {intercept}\
          \ncoef: {coef}\
          \nr2: {r_squared} (p={p})\n')
    
    x = np.arange(0, max(df['Distance']))
    y = [xx * coef + intercept for xx in x]
    
    plt.figure()
    sns.scatterplot('Distance', 'Duration', data=df)
    plt.plot(x, y, 'r')
    plt.title(f'Condition {condition}, r2={r_squared}')
    plt.show()
    
    return intercept, coef, r_squared, p

def get_linear_regression(df):    
    lm = linear_regression(list(df['Distance']), list(df['Duration']))
    lm = lm.round(4)
    
    intercept = lm.loc[lm['names'] == 'Intercept']['coef'].values[0]
    coef = lm.loc[lm['names'] == 'x1']['coef'].values[0]
    r_squared = list(lm['r2'])[0]
    p = list(lm['pval'])[0]
    
    return intercept, coef, r_squared, p
    
                
def get_saccades(ID, f):
    features = [\
            'Saccade',
            'Distance',
            'Duration']
    cols = ['ID', 'Condition', 'Trial']
    [cols.append(f) for f in features]
    results = pd.DataFrame(columns=cols)
        
    fix_df = pd.read_csv(f)
        
    for condition in list(fix_df['Condition'].unique()):
        fix_df_c = fix_df.loc[fix_df['Condition'] == condition]
        
        for trial in list(fix_df_c['Trial'].unique()): 
            if trial not in EXCLUDE_TRIALS:                
                fix_df_t = fix_df_c.loc[fix_df_c['Trial'] == trial]

                saccades = fix_df_t.loc[fix_df_t['type'] == 'saccade']
                                

                for i in range(len(saccades)):
                    df = saccades.iloc[i]
                    d = euclidean_distance((df['gstx'], df['gsty']), (df['genx'], df['geny']))
                
                    r = pd.DataFrame({'ID': ID,
                                      'Condition': int(condition),
                                      'Trial': int(trial),
                                      'Saccade': i,
                                      'Distance': d,
                                      'Duration': df['end'] - df['start']},
                                     index=[0])
                    results = results.append(r, ignore_index=True)
                    
    return results

if __name__ == '__main__':
    pp_info = pd.read_excel('../results/participant_info.xlsx')
    pp_info['ID'] = [str(x).zfill(3) for x in list(pp_info['ID'])]
    
    pp_info = hf.remove_from_pp_info(pp_info, [f'Trials condition {i}' for i in range(4)])
    IDs = sorted(list(pp_info['ID'].unique()))
    
    fixations_files = sorted([f for f in hf.getListOfFiles(base_location) if '-allFixations.csv' in f])
    files = [f for f in fixations_files if not '/008/' in f]
    
    dfs = Parallel(n_jobs=11, backend='loky', verbose=True)(delayed(get_saccades)(ID, f) for ID, f in zip(IDs, files))
    results = pd.concat(dfs, ignore_index=True)
    
    conditions, intercepts, coefs, r2s, ps = [], [], [], [], []
    for condition in sorted(list(results['Condition'].unique())):
        intercept, coef, r_squared, p = plot_saccades(results, condition)

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
    lm_results.to_excel('../results/lm_results.xlsx')





