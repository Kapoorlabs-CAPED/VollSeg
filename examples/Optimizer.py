#!/usr/bin/env python
# coding: utf-8

# # This notebook is used to obtain 2D dual segmentations.
# One of the segmetation is the mask region obtianed by applying the trained UNET model, second segmentation is the stardist trained model to obtain instance segmentation of objects inside that mask, for example 2D cells inside a 2D mask region.

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import glob
import sys
sys.path.append('../')
import cv2
import numpy as np
from tqdm import tqdm
from stardist.models import StarDist3D
from csbdeep.models import Config, CARE
from tifffile import imread
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from vollseg.OptimizeThreshold import OptimizeThreshold
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path


# In[2]:


BaseDir = '/data/u934/service_imagerie/v_kapoor/CurieTrainingDatasets/MouseClaudia/AugmentedGreenCell3D/'

Model_Dir = '/data/u934/service_imagerie/v_kapoor/CurieDeepLearningModels/MouseClaudia/'
SaveDir = '/data/u934/service_imagerie/v_kapoor/CurieTrainingDatasets/MouseClaudia/'


StardistModelName = 'ScipyDeepGreenCells'
UNETModelName = 'UNETScipyDeepGreenCells'


NoiseModel = None
Starmodel = StarDist3D(config = None, name = StardistModelName, basedir = Model_Dir)
UnetModel = CARE(config = None, name = UNETModelName, basedir = Model_Dir)


# In[3]:


#Number of tiles to break the image into for applying the prediction to fit in the computer memory
n_tiles = (1,2,2)


#Use Probability map = True or distance map = False as the image to perform watershed on
UseProbability = False


# In[ ]:


Raw = sorted(glob.glob(BaseDir + '/Raw/' + '*.tif'))
RealMask = sorted(glob.glob(BaseDir + '/RealMask/' + '*.tif'))
X = list(map(imread,Raw))
Y = list(map(imread,RealMask))
OptimizeThreshold(Starmodel,UnetModel,X,Y,BaseDir, UseProbability = UseProbability, n_tiles=n_tiles)


# In[ ]:




