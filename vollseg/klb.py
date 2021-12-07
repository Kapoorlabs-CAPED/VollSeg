
from pyklb import readfull, writefull, readheader
from .spatial_image import SpatialImage
import numpy as np

def read_klb(filename, SP_im = True):
    tmp = readfull(filename)
    if len(tmp.shape) == 2:
        tmp = tmp.reshape((1, ) + tmp.shape)
    if SP_im:
        im = SpatialImage(tmp.transpose(2, 1, 0), copy = False)
        im.voxelsize = readheader(filename).get('pixelspacing_tczyx', [1., ]*5)[:1:-1]
    else:
        im = tmp.transpose(2, 1, 0)
    return im

def write_klb(filename, im):
    if np.isfortran(im):
        writefull(im.transpose(2, 1, 0), filename, pixelspacing_tczyx = (1, 1) + tuple(im.resolution[::-1]))
    else:    
        writefull(im, filename)