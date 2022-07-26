import os
import glob
import sys
import numpy as np
from tqdm import tqdm
from tifffile import imread, imwrite
from vollseg.utils import expand_labels
from pathlib import Path


image_dir = '/mask_images/'

save_dir = image_dir + '/real_mask/'

Raw_path = os.path.join(image_dir, '*.tif')
filesRaw = glob.glob(Raw_path)
filesRaw.sort

for fname in filesRaw:
     
     image = imread(fname)
     Name = os.path.basename(os.path.splitext(fname)[0])

     for i in range(0, image.shape[0]):

            newimage = expand_labels(image[i,:,:], distance = 5 )

            imwrite(save_dir + Name + str(i) + '.tif', newimage.astype('uint16'))

raw_image_dir = '/raw_image/'

save_dir = raw_image_dir + '/Raw/'

Raw_path = os.path.join(raw_image_dir, '*.tif')
filesRaw = glob.glob(Raw_path)
filesRaw.sort

for fname in filesRaw:
     
     image = imread(fname)
     Name = os.path.basename(os.path.splitext(fname)[0])

     for i in range(0, image.shape[0]):

            newimage = image[i,:,:]

            imwrite(save_dir + Name + str(i) + '.tif', newimage.astype('float32'))