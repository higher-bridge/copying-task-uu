"""
Created on Thu Jul  9 14:49:03 2020

@author: Alex
"""


import os
from random import sample

from pathlib import Path
from PyQt5.QtGui import QImage

fullpath = Path('C:/Users/Alex/Documents/copying-task-uu/stimuli/PNG_rnd_shp_1_1.png')
path = str(fullpath)

# image = QImage()
# with open(path, 'rb') as f:
#     im = f.read()
#     image.loadFromData(im)

image = QImage(path)

print(image.isNull())

if image.width() <= 0:
    raise ValueError(f'Image "{path}" has a width of 0.')
    
    

# from PIL import Image

# for i in range(1, 6):
#     for j in range(1, 5):
#             im1 = Image.open(r'C:/Users/Alex/Documents/copying-task-uu/stimuli/rnd_shp_{}_{}.jpg'.format(i, j))
#             im1.save(r'C:/Users/Alex/Documents/copying-task-uu/stimuli/PNG_rnd_shp_{}_{}.png'.format(i, j))
