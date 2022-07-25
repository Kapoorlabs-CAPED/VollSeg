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
from vollseg import UNET, VollSeg
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path


# In[ ]:


image_dir = 'data/tiffiles/'
model_dir = 'models/'
save_dir = image_dir + 'Results/'


unet_model_name = 'kymo_unet_model'


unet_model = UNET(config = None, name = unet_model_name, basedir = model_dir)


# In[ ]:


Raw_path = os.path.join(image_dir, '*.tif')
filesRaw = glob.glob(Raw_path)
filesRaw.sort

#Adjust the number of tiles depending on how good your GPU is, tiling ensures that your image tiles fit into the runtime
#memory 
n_tiles = (2,2)
axes = 'YX'
for fname in filesRaw:
     
     image = imread(fname)
     Name = os.path.basename(os.path.splitext(fname)[0])
     VollSeg( image, 
             unet_model = unet_model, 
             axes = axes, 
             n_tiles = n_tiles, 
             save_dir = save_dir, 
             Name = Name)

