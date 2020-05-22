#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 19:01:21 2020
Author: Alex Hoogerbrugge (@higher-bridge)
"""

class Copygrid():
    # Represents the grid to which participants copy
    def __init__(self, nrow:int, ncol:int):
        self.nrow = nrow
        self.ncol = ncol
        