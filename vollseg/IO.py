from os.path import exists, splitext, isdir, split as psplit, expanduser as expusr
import os, fnmatch
import warnings
from struct import pack,unpack,calcsize
from pickle import dumps,loads
import numpy as np

from .spatial_image import SpatialImage

from .inrimage import read_inrimage, write_inrimage
try:
    from .tif import read_tif, write_tif
except Exception as e:
    warnings.warn('pylibtiff library is not installed')

try:
    from .h5 import read_h5
except Exception as e:
    warnings.warn('h5py library is not installed')

try:
    from .klb import read_klb, write_klb
except Exception as e:
    warnings.warn('KLB library is not installed')

from .folder import read_folder


def imread (filename, parallel = True, SP_im = True) :
    """Reads an image file completely into memory.

    It uses the file extension to determine how to read the file. It first tries
    some specific readers for volume images (Inrimages, TIFFs, LSMs, NPY) or falls
    back on PIL readers for common formats if installed.

    In all cases the returned image is 3D (2D is upgraded to single slice 3D).
    If it has colour or is a vector field it is even 4D.

    :Parameters:
     - `filename` (str)

    :Returns Type:
        |SpatialImage|
    """
    filename = expusr(filename)
    if not exists(filename) :
        raise IOError("The requested file do not exist: %s" % filename)

    root, ext = splitext(filename)
    ext = ext.lower()
    if ext == ".gz":
        root, ext = splitext(root)
        ext = ext.lower()
    if ext == ".inr":
        return read_inrimage(filename)
    elif ext in [".tif", ".tiff"]:
        return read_tif(filename)
    elif ext in [".npz", ".npy"]:
        return load(filename)
    elif ext in [".h5", ".hdf5"]:
        return read_h5(filename)
    elif ext == '.klb':
        return read_klb(filename, SP_im)
    elif isdir(filename):
        return read_folder(filename, parallel)

def imsave(filename, img):
    """Save a |SpatialImage| to filename.

    .. note: `img` **must** be a |SpatialImage|.

    The filewriter is choosen according to the file extension. However all file extensions
    will not match the data held by img, in dimensionnality or encoding, and might raise `IOError`s.

    For real volume data, Inrimage and NPY are currently supported.
    For |SpatialImage|s that are actually 2D, PNG, BMP, JPG among others are supported if PIL is installed.

    :Parameters:
     - `filename` (str)
     - `img` (|SpatialImage|)
    """

    filename = expusr(filename)
    root, ext = splitext(filename)

    # assert isinstance(img, SpatialImage) or ext == '.klb'
    # -- images are always at least 3D! If the size of dimension 3 (indexed 2) is 1, then it is actually
    # a 2D image. If it is 4D it has vectorial or RGB[A] data. --
    head, tail = psplit(filename)
    head = head or "."
    if not exists(head):
        raise IOError("The directory do not exist: %s" % head)

    # is2D = img.shape[2] == 1
    ext = ext.lower()
    if ext == ".gz":
        root, ext = splitext(root)
        ext = ext.lower()
    if ext == ".inr":
        write_inrimage(filename, img)
    elif ext in [".npz", ".npy"]:
        save(filename, img)
    elif ext in [".tiff", ".tif"]:
        write_tif(filename, img)
    elif ext == '.klb':
        write_klb(filename, img)
    elif ext in ['.h5', '.hdf5']:
        write_h5(filename, img)