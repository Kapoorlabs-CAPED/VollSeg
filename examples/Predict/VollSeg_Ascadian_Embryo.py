#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import glob
import sys
import numpy as np
from tqdm import tqdm
from tifffile import imread, imwrite
from vollseg import StarDist3D, UNET, VollSeg, MASKUNET
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path


# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


image_dir = 'data/tiffiles/'
model_dir = 'models/'
save_dir = image_dir + 'Results/'


unet_model_name = 'unet_model'


unet_model = UNET(config = None, name = unet_model_name, basedir = model_dir)


# In[ ]:


Raw_path = os.path.join(image_dir, '*.tif')
filesRaw = glob.glob(Raw_path)
filesRaw.sort
#Minimum size in pixels for the cells to be segmented
min_size = 1
#Minimum size in pixels for the mask region, regions below this threshold would be removed
min_size_mask=10
#maximum size of the region, set this to veto regions above a certain size
max_size = 1000000
#Adjust the number of tiles depending on how good your GPU is, tiling ensures that your image tiles fit into the runtime
#memory 
n_tiles = (2,2,2)

dounet = True
#Wether unet create labelling in 3D or slice by slice can be set by this parameter, if true it will merge neighbouring slices
slice_merge = False
iou_threshold = 0.5
#Set up the axes keyword depending on the type of image you have, if it is a time lapse movie of XYZ images 
#your axes would be TZYX, if it is a timelapse of 2D images the axes would be TYX, for a directory of XYZ images
#the axes is ZYX and for a directory of XY images the axes is YX
axes = 'ZYX'
for fname in filesRaw:
     
     image = imread(fname)
     Name = os.path.basename(os.path.splitext(fname)[0])
     VollSeg( image, 
             unet_model = unet_model, 
             axes = axes, 
             min_size = min_size,  
             min_size_mask = min_size_mask,
             max_size = max_size,
             n_tiles = n_tiles, 
             slice_merge = slice_merge, 
             save_dir = save_dir, 
             Name = Name, 
             dounet = dounet,
             iou_threshold = iou_threshold)


# In[ ]:




