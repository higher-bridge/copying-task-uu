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

import helperfunctions as hf

path = '../results'

allfiles = [f for f in hf.getListOfFiles(path)]
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