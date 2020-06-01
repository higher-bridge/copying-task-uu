"""
Created on Wed Feb 26 19:06:29 2020
Author: Alex Hoogerbrugge (@higher-bridge)
"""

from PyQt5.QtGui import QImage
from random import sample
import os

class Stimulus():
    """Represents a stimulus"""
    
    def __init__(self, image:QImage, path:str):
        self.qimage = image
        self.name = path


def load_stimulus(path:str, image_width:int):
    """"Loads a file into a QImage and returns it as a Stimulus object"""
    
    image = QImage(path)
    
    if image.width() <= 0:
        raise ValueError(f'Image "{path}" has a width of 0.')
        
    image = image.scaledToWidth(image_width)
    
    return Stimulus(image, path)


def pick_stimuli(path:str, n:int=6, image_width:int=100, extension:str='.png'):
    """Loads all images with the specified extension from the specified path.
    Returns a list of n randomly picked Stimulus objects

    Arguments:
        path {str} -- [description]

    Keyword Arguments:
        n {int} -- [description] (default: {6})
        image_width {int} -- [description] (default: {100})
        extension {str} -- [description] (default: {'.png'})

    Returns:
        list -- n randomly picked stimuli
    """
    
    paths = sorted([os.path.join(path, file) for file in os.listdir(path) \
                    if extension in file])
    
    stimuli = []
    for p in paths:
        stimuli.append(load_stimulus(p, image_width))
    
    n = len(stimuli) if n is None else n    
    return sample(stimuli, k=n)










