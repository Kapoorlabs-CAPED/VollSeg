from h5py import File
from .spatial_image import SpatialImage
import numpy as np

def read_h5(filename):
    """Read an hdf5 file

    :Parameters:
     - `filename` (str) - name of the file to read
    """
    data = File(filename.replace('\\', ''), 'r')['Data']
    im_out = np.zeros(data.shape, dtype=data.dtype)
    data.read_direct(im_out)
    return SpatialImage(im_out.transpose(2, 1, 0))

def write_h5(filename, im):
    """Write an image to an hdf5

    :Parameters:
     - `filename` (str) - name of the file to write
     - `im` SpatialImage - image to write
    """
    f = File(filename, mode = 'w')
    dset = f.create_dataset("Data", data=im.transpose(2, 1, 0), chunks=True)
    f.close()