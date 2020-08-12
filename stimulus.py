"""
Created on Wed Feb 26 19:06:29 2020
Author: Alex Hoogerbrugge (@higher-bridge)
"""

import os
from random import sample

from pathlib import Path
from PyQt5.QtGui import QImage


class Stimulus():
    """Represents a stimulus"""
    
    def __init__(self, image:QImage, path:str):
        self.qimage = image
        self.name = path


def load_stimulus(path:str, image_width:int):
    """"Loads a file into a QImage and returns it as a Stimulus object"""
    
    with open(path, 'rb') as f:
        im = f.read()
    
    image = QImage()
    image.loadFromData(im)
    
    # print(image.isNull())
    
    if image.width() <= 0:
        raise ValueError(f'Image "{path}" has a width of 0.')
        
    image = image.scaledToWidth(image_width)
    # path = path.replace('/Users/mba/copying-task-uu/stimuli/', '')
    
    return Stimulus(image, path)


def load_stimuli(path:str, image_width:int=100, extension:str='.png'):
    """Loads all images with the specified extension from the specified path.
    Returns a list of n randomly picked Stimulus objects

    Arguments:
        path {str} -- [description]

    Keyword Arguments:
        image_width {int} -- [description] (default: {100})
        extension {str} -- [description] (default: {'.png'})

    Returns:
        list -- n stimuli
    """
    
    # [print(path, file) for file in os.listdir(path) if extension in file]
    
    # paths = sorted([os.path.join(path, file) for file in os.listdir(path) \
                    # if extension in file])
    
    paths = sorted([path / Path(file) for file in os.listdir(path) if extension in file])
    # print(paths)
    
    stimuli = []
    for p in paths:
        stimuli.append(load_stimulus(str(p), image_width))
    
    # n = len(stimuli) if n is None else n    
    return stimuli

def pick_stimuli(stimuli:list, n:int=6):
    return sample(stimuli, k=n)
