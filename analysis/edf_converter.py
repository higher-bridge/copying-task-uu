#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:36:05 2020

@author: alexos
"""

from pyedfread import edf
import pandas as pd
from joblib import Parallel, delayed

import os
import sys

import helperfunctions as hf

    
def convert_edf(file, allfiles):
    try:
        new_filename = file.replace('.edf', '-samples.csv')
        
        if new_filename not in allfiles:
            samples, events, messages = edf.pread(file)
            samples.to_csv(file.replace('.edf', '-samples.csv'))
            events.to_csv(file.replace('.edf', '-events.csv'))
        
        # print(f'Parsed {i + 1} of {len(files)} files')
        # sys.stdout.write("\033[F")
        
        return True
    
    except Exception as e:
        print(file, e)
        return False


if __name__ == '__main__':
    path = '../results'

    allfiles = [f for f in hf.getListOfFiles(path)]
    files = [f for f in allfiles if '.edf' in f]
    
    allfiles_repeated = [allfiles] * len(files)
    
    results = Parallel(n_jobs=10, backend='loky', verbose=True)(delayed(convert_edf)(f, af) for f, af in zip(files, allfiles_repeated))

    # print(f'Succesfully converted {len(files)} files!')