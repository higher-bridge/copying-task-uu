#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 13:40:00 2020

@author: alexos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


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

def find_nearest_index(array, timestamp):
    array = np.asarray(array)
    idx = (np.abs(array - timestamp)).argmin()
    return idx

def write_IDs_to_dict(all_IDs:list):
    ''' Takes a list of IDs and writes to dict with sub-lists for 
    barreled IDs (001-2, etc.) '''
    ID_dict = dict()
    
    for ID in all_IDs:
        temp_ID = ID[0:3]
        
        if temp_ID not in ID_dict.keys():
            ID_dict[temp_ID] = []        
        else:
            ID_dict[temp_ID].append(ID[-1])

    return ID_dict

def find_files(ID:str, sessions:list, location:str, subfix:str):
    ''' Returns a list of all files which match {ID}-{session} for all sessions '''
    all_files = []
    all_folders = [f'{location}/{ID}']
    
    if len(sessions) > 0:
        [all_folders.append(f'{location}/{ID}-{s}') for s in sessions]
    
    for folder in all_folders:
        file_list = sorted([f'{folder}/{f}' for f in os.listdir(folder)])
        for file in file_list:
            if subfix in file:
                all_files.append(file)
    
    return all_files

def concat_event_files(eventfiles:list):
    ''' Reads and concatenates multiple dataframes '''
    all_sessions = []

    for ef in eventfiles:        
        # Load events
        sess = pd.read_csv(ef)
        all_sessions.append(sess)

    events = pd.concat(all_sessions, ignore_index=True)
    return events

def order_by_condition(df, condition_order:list):
    new_order = []
    
    for c in condition_order:
        df_c = df.loc[df['Condition'] == c]
        new_order.append(df_c)
        
    new_df = pd.concat(new_order, ignore_index=True)

    return new_df

def get_condition_order(df, ID:str, conditions:list=[1, 2, 3, 4]):
    df_id = df.loc[df['ID'] == ID]
    
    condition_order = []
    
    for condition in conditions:
        colname = f'Block {condition}'
        c = list(df_id[colname])[0]
        condition_order.append(c)
        
    return condition_order

def get_num_trials(df, conditions:list=[0, 1, 2, 3]):
    num_trials = []
    
    # conditions = sorted(list(df['Condition'].unique()))
    for condition in conditions:
        df_c = df.loc[df['Condition'] == condition]
        trials = list(df_c['Trial'].unique())
        
        if len(trials) > 0:
            num_trials.append(len(trials))
        else:
            num_trials.append(0)
        
    return num_trials

def get_midline_crossings(xpos:list, midline=1280):
    num_crossings = 0
    prev_x = 2560
    
    for x in xpos:
        if prev_x > midline and x < midline:
            num_crossings += 1
        
        prev_x = x
    
    return num_crossings

def get_left_side_fixations(xpos:list, midline=1280):
    return len([x for x in xpos if x < midline])
    
def get_left_ratio_fixations(xpos:list, midline=1280):
    return len([x for x in xpos if x < midline]) / len(xpos)


def scatterplot_fixations(data, x, y, title:str, savestr:str, save=True):
    # Plot fixations
    plt.figure()
    sns.scatterplot(x, y, data=data)
    plt.xlim((0, 2560))
    plt.ylim((1440, 0)) # Note that the y-axis needs to be flipped
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.title(title)
    plt.savefig(savestr, dpi=500)
    plt.show()