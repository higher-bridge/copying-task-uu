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

path = f'{os.getcwd()}/results'

allfiles = [f'{path}/{f}' for f in os.listdir(path)]
files = [f for f in allfiles if '.edf' in f]


for i, file in enumerate(files):
    samples, events, messages = edf.pread(file)
    
    new_filename = file.replace('.edf', '-samples.csv')
    
    if new_filename not in allfiles:
        samples.to_csv(new_filename)
        events.to_csv(file.replace('.edf', '-events.csv'))
    
    print(f'Parsed {i + 1} of {len(files)} files')
    sys.stdout.write("\033[F")
    
print(f'Succesfully converted {len(files)} files!')