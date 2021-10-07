#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 15:29:53 2021

@author: alexos
"""

from Cython.Build import cythonize
from setuptools import setup

setup(
    name='cimulation',
    ext_modules=cythonize(['cimulation_helper.pyx', 
                           'cimulate_trial.pyx',
                           'cimulate_batch.pyx',
                           'cimulation_analysis_helper.pyx']),
    zip_safe=False,
)

# python setup.py build_ext --inplace
