from os import path
import os
import numpy as np
from multiprocessing import Pool

def populate_im(parameters):
    from .IO import imread, SpatialImage
    path, pos = parameters
    return imread(path)[:, :, 0]


def read_folder(folder, parallel):
    from .IO import imread, SpatialImage
    data_files = sorted(os.listdir(folder))
    nb_images = len(data_files)
    first_image = imread(path.join(folder, data_files[0]))
    to_read = []
    for i, p in enumerate(data_files):
        to_read.append([path.join(folder, p), i])#(i, path.join(folder, p)))

    if parallel:
        pool = Pool()
        out = pool.map(populate_im, to_read)
        pool.close()
        pool.terminate()
    else:
        out = []
        for param in to_read:
            out.append(populate_im(param))
    return SpatialImage(np.array(out).transpose(1, 2, 0), voxelsize = first_image.voxelsize)