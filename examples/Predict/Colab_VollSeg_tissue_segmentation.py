#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive', force_remount = True)
get_ipython().run_line_magic('tensorflow_version', '2.x')


# In[ ]:


get_ipython().system('pip uninstall keras -y')
get_ipython().system('pip uninstall keras-nightly -y')
get_ipython().system('pip uninstall keras-Preprocessing -y')
get_ipython().system('pip uninstall keras-vis -y')
get_ipython().system('pip uninstall tensorflow -y')
get_ipython().system('pip install napari[all]')
get_ipython().system('pip install tensorflow==2.2.0')
get_ipython().system('pip install keras==2.3.0')
get_ipython().system('pip install vollseg ')
get_ipython().system('pip install napari[all]')


# In[ ]:


import os
import glob
import sys
import numpy as np
from tqdm import tqdm
from tifffile import imread, imwrite
from vollseg import  UNET, VollSeg
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path


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

#Minimum size in pixels for the mask region, regions below this threshold would be removed
min_size_mask=10
#maximum size of the region, set this to veto regions above a certain size
max_size = 1000000
#Adjust the number of tiles depending on how good your GPU is, tiling ensures that your image tiles fit into the runtime
#memory 
n_tiles = (2,2)

#Set up the axes keyword depending on the type of image you have, if it is a time lapse movie of XYZ images 
#your axes would be TZYX, if it is a timelapse of 2D images the axes would be TYX, for a directory of XYZ images
#the axes is ZYX and for a directory of XY images the axes is YX
axes = 'YX'
for fname in filesRaw:
     
     image = imread(fname)
     Name = os.path.basename(os.path.splitext(fname)[0])
     VollSeg( image, 
             unet_model = unet_model, 
             axes = axes, 
             min_size_mask = min_size_mask,
             max_size = max_size,
             n_tiles = n_tiles, 
             save_dir = save_dir, 
             Name = Name)

