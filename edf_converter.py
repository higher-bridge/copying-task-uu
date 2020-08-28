#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:36:05 2020

@author: alexos
"""

from pyedfread import edf
import pandas as pd

import os
import sys

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

path = f'{os.getcwd()}/results'

allfiles = [f for f in getListOfFiles(path)]
files = [f for f in allfiles if '.edf' in f]


for i, file in enumerate(files):
    new_filename = file.replace('.edf', '-samples.csv')
    
    if new_filename not in allfiles:
        samples, events, messages = edf.pread(file)
        # samples.to_csv(new_filename)
        events.to_csv(file.replace('.edf', '-events.csv'))
    
    print(f'Parsed {i + 1} of {len(files)} files')
    sys.stdout.write("\033[F")
    
print(f'Succesfully converted {len(files)} files!')