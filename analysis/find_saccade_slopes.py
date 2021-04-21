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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from joblib import Parallel, delayed
from pingouin import linear_regression
from math import atan2

import helperfunctions as hf
import constants_analysis as constants
from simulation_helper import euclidean_distance, fitts_id

def get_angle(x=None, y=None):
    if x != None and y == None:
        return np.rad2deg(atan2(x * constants.PIXEL_WIDTH, constants.DISTANCE))
    elif x == None and y != None:
        return np.rad2deg(atan2(y * constants.PIXEL_HEIGHT, constants.DISTANCE))
    elif x != None and y != None:
        x_angle = np.rad2deg(atan2((x * constants.PIXEL_WIDTH), constants.DISTANCE))
        y_angle = np.rad2deg(atan2((y * constants.PIXEL_HEIGHT), constants.DISTANCE))
        return (x_angle, y_angle)
    else:
        raise ValueError('Give either x or y parameters, or both.')

def compute_angle_change(loc1:tuple, loc2:tuple):
    deg_from_center_start = get_angle(x=loc1[0] - constants.SCREEN_CENTER[0],
                                      y=loc1[1] - constants.SCREEN_CENTER[1])
    deg_from_center_end = get_angle(x=loc2[0] - constants.SCREEN_CENTER[0],
                                    y=loc2[1] - constants.SCREEN_CENTER[1])
        
    angle_change = euclidean_distance(deg_from_center_start, deg_from_center_end)
    
    return angle_change 

# def get_poly_linear_regression(df, condition, X, Y):
#     x = np.array(df[X]).reshape(-1, 1)
#     y = np.array(df[Y]).reshape(-1, 1)
    
#     polynomial_features = PolynomialFeatures(degree=5)
#     x_poly = polynomial_features.fit_transform(x)

#     model = LinearRegression()
#     model.fit(x_poly, y)
#     y_poly_pred = model.predict(x_poly)
    
#     rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
#     r2 = r2_score(y,y_poly_pred).round(4)
#     print(f'\nCondition {condition}, RMSE={rmse}, R2={r2}')
    
#     plt.scatter(x, y, s=10)
#     # sort the values of x before line plot
#     sort_axis = operator.itemgetter(0)
#     sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
#     x, y_poly_pred = zip(*sorted_zip)
#     plt.plot(x, y_poly_pred, color='m')
#     plt.title(f'Condition {condition}, r2={r2}')
#     plt.xlabel(X)
#     plt.ylabel(Y)
#     plt.show()


def get_linear_regression(df, X, Y):    
    lm = linear_regression(list(df[X]), list(df[Y]))
    lm = lm.round(4)
    
    intercept = lm.loc[lm['names'] == 'Intercept']['coef'].values[0]
    coef = lm.loc[lm['names'] == 'x1']['coef'].values[0]
    
    r_squared = list(lm['r2'])[0]
    p = list(lm['pval'])[0]
    
    return intercept, coef, r_squared, p

def plot_saccades(results, condition, X='Distance (pixels)', Y='Duration (ms)',
                  x_limit=2560, y_limit=500, x_min=5):
    
    df = results.loc[results['Condition'] == condition]

    df = df.loc[df[X] < x_limit]
    df = df.loc[df[X] > x_min]

    df = df.loc[df[Y] < y_limit]
    df = df.loc[df[Y] > 3]
    
    intercept, coef, r_squared, p = get_linear_regression(df, X, Y)
    print(f'Condition {condition}\
          \nIntercept: {intercept}\
          \ncoef: {coef}\
          \nr2: {r_squared} (p={p})\n')
              
    x = np.arange(0, max(df[X]))
    y = [xx * coef + intercept for xx in x]
    
    plt.figure()
    sns.scatterplot(x=X, y=Y, data=df) #, hue='Dragging')
    plt.plot(x, y, 'r')
    plt.title(f'Condition {condition}, r2={r_squared}')
    plt.show()
    
    return intercept, coef, r_squared, p

def plot_slopes(model_results, title:str, x_str:str='Distance (pixels)', x_limit:int=constants.RESOLUTION[0]):    
    d = {key: [] for key in [x_str, 'Duration (ms)', 'Condition']}
    
    for i, condition in enumerate(list(model_results['Condition'].unique())):
        mr = model_results.loc[model_results['Condition'] == condition]
        intercept = list(mr['Intercept'])[0]
        coef = list(mr['Coefficient'])[0]
        # r2 = list(mr['r2'])[0]
        
        x_range = np.arange(0, x_limit)
        
        for x in x_range:
            y = x * coef + intercept
            
            d[x_str].append(x)
            d['Duration (ms)'].append(y)
            d['Condition'].append(condition)


    df = pd.DataFrame(d)

    plt.figure(figsize=(3.75, 2.5))        
    sns.lineplot(x=x_str, y='Duration (ms)', hue='Condition', style='Condition', data=df)
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'../results/{title}.png', dpi=500)
    plt.show()
        
                
def get_saccades(ID, f):
    # Set list of features. Split by dependent and independent features, in essence
    features = [\
            'Saccade',
            'Distance (pixels)',
            'Distance (degrees)',
            'Distance (fitts)',
            'Duration (ms)',
            'Dragging']
    cols = ['ID', 'Condition', 'Trial', 'Start time', 'End time']
    [cols.append(f) for f in features]
    results = pd.DataFrame(columns=cols)
        
    fix_df = pd.read_csv(f)
    
    if 'Fixations' in f:
        features = ['gstx', 'gsty', 'genx', 'geny']
    elif 'mouse' in f:
        features = ['start_x', 'start_y', 'end_x', 'end_y']
    else:
        raise KeyError('Could not determine whether to use gaze or mouse')

    # Run trhough each condition and trial and retrieve the saccades made        
    for condition in list(fix_df['Condition'].unique()):
        fix_df_c = fix_df.loc[fix_df['Condition'] == condition]
        
        for trial in list(fix_df_c['Trial'].unique()): 
            if trial not in constants.EXCLUDE_TRIALS:                
                fix_df_t = fix_df_c.loc[fix_df_c['Trial'] == trial]

                saccades = fix_df_t.loc[fix_df_t['type'] == 'saccade']

                # Compute features for each saccade in trial and write to df
                for i in range(len(saccades)):
                    df = saccades.iloc[i]
                    d = euclidean_distance((df[features[0]], df[features[1]]), (df[features[2]], df[features[3]]))
                    angle_change = compute_angle_change((df[features[0]], df[features[1]]), (df[features[2]], df[features[3]]))
                    fitts = fitts_id((df[features[0]], df[features[1]]), (df[features[2]], df[features[3]]))
                    
                    dragging = df['Dragging']
                
                    r = pd.DataFrame({'ID': ID,
                                      'Condition': int(condition),
                                      'Trial': int(trial),
                                      'Start time': df['start'],
                                      'End time': df['end'],
                                      'Saccade': i,
                                      'Distance (pixels)': d,
                                      'Distance (degrees)': angle_change,
                                      'Distance (fitts)': fitts,
                                      'Duration (ms)': df['end'] - df['start'],
                                      'Dragging': dragging},
                                     index=[0])
                    results = results.append(r, ignore_index=True)
                    
    return results

if __name__ == '__main__':
    pp_info = pd.read_excel('../results/participant_info.xlsx')
    pp_info['ID'] = [str(x).zfill(3) for x in list(pp_info['ID'])]
    
    pp_info = hf.remove_from_pp_info(pp_info, [f'Trials condition {i}' for i in range(4)])
    IDs = sorted(list(pp_info['ID'].unique()))

    # =============================================================================
    # EYE ANALYSIS    
    # =============================================================================    
    fixations_files = sorted([f for f in hf.getListOfFiles(constants.base_location) if '-allFixations.csv' in f])
    files = [f for f in fixations_files if not '/008/' in f]
    
    dfs = Parallel(n_jobs=-3, backend='loky', verbose=True)(delayed(get_saccades)(ID, f) for ID, f in zip(IDs, files))
    results = pd.concat(dfs, ignore_index=True)
    results.to_csv('../results/all-saccades.csv')
    
    conditions, intercepts, coefs, r2s, ps = [], [], [], [], []
    for condition in sorted(list(results['Condition'].unique())):
        # plot_saccades(results, condition)
        intercept, coef, r_squared, p = plot_saccades(results, condition, X='Distance (pixels)', x_limit=1800)

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
    
    plot_slopes(lm_results, 'Regressions of saccade duration', x_limit=1000)
    
    # =============================================================================
    # MOUSE ANALYSIS    
    # =============================================================================
    fixations_files = sorted([f for f in hf.getListOfFiles(constants.base_location) if '-mouseEvents.csv' in f])
    files = [f for f in fixations_files if not '/008/' in f]
    
    dfs = Parallel(n_jobs=-3, backend='loky', verbose=True)(delayed(get_saccades)(ID, f) for ID, f in zip(IDs, files))
    results = pd.concat(dfs, ignore_index=True)
    results.to_csv('../results/all-saccades-mouse.csv')
    
    conditions, intercepts, coefs, r2s, ps = [], [], [], [], []
    for condition in sorted(list(results['Condition'].unique())):
        # plot_saccades(results, condition)
        intercept, coef, r_squared, p = plot_saccades(results, condition, X='Distance (fitts)', 
                                                      y_limit=2000, x_limit=1000, x_min=0)

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
    lm_results.to_excel('../results/lm_results_mouse.xlsx')    
    
    plot_slopes(lm_results, 'Regressions of mouse movement duration', x_limit=6, x_str='Distance (fitts)')
    





