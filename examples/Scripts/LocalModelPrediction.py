#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import glob
import sys
import numpy as np
from tqdm import tqdm
from tifffile import imread, imwrite
from stardist.models import StarDist3D
from csbdeep.models import Config, CARE
from vollseg import SmartSeedPrediction3D
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path
from n2v.models import N2V


# In[ ]:


ImageDir = 'data/tiffiles/'
DenoiseImageDir = 'data/denoisedtiffiles/'
Model_Dir = 'data/'

SaveDir = ImageDir + 'Results/'


NoiseModelName = 'ScipyDenoising'
UNETModelName = 'UNETVolumeSeg'
StarModelName = 'VolumeSeg'

UnetModel = CARE(config = None, name = UNETModelName, basedir = Model_Dir)
StarModel = StarDist3D(config = None, name = StarModelName, basedir = Model_Dir)
NoiseModel = N2V(config=None, name=NoiseModelName, basedir=Model_Dir)


# In[ ]:


Raw_path = os.path.join(ImageDir, '*.tif')
filesRaw = glob.glob(Raw_path)
filesRaw.sort
min_size = 5
n_tiles = (1,1,1)

       
        
for fname in filesRaw:
     
     SmartSeedPrediction3D(SaveDir, fname, UnetModel, StarModel, NoiseModel, min_size = min_size,  n_tiles = n_tiles, UseProbability = True)


# In[ ]:




