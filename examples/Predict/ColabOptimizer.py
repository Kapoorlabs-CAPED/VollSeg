#!/usr/bin/env python
# coding: utf-8

# # Optimize prob and nms thresholds

# In[ ]:


from google.colab import drive
drive.mount('/content/drive', force_remount = True)
get_ipython().run_line_magic('tensorflow_version', '2.x')


# In[ ]:


get_ipython().system('pip install tiffile')

get_ipython().system('pip install vollseg')


# In[ ]:



import glob
import numpy as np
from tqdm import tqdm
from csbdeep.models import Config, CARE
from tifffile import imread
from vollseg import StarDist3D, UNET, VollSeg3D
from vollseg.OptimizeThreshold import OptimizeThreshold

from pathlib import Path


# In[ ]:


BaseDir = '/content/drive/My Drive/training/'

Model_Dir = '/content/drive/My Drive/TrainedModels/'
SaveDir = BaseDir + '/Raw/AugmentedResults/'

UNETModelName = 'UNETAugmented'
StardistModelName = 'Augmented'

NoiseModel = None
Starmodel = StarDist3D(config = None, name = StardistModelName, basedir = Model_Dir)
UnetModel = UNET(config = None, name = UNETModelName, basedir = Model_Dir)


# In[ ]:


#Number of tiles to break the image into for applying the prediction to fit in the computer memory
n_tiles = (1,2,2)


#Use Probability map = True or distance map = False as the image to perform watershed on
UseProbability = True


# In[ ]:


Raw = sorted(glob.glob(BaseDir + '/Raw/' + '*.tif'))
RealMask = sorted(glob.glob(BaseDir + '/RealMask/' + '*.tif'))
X = list(map(imread,Raw))
Y = list(map(imread,RealMask))
OptimizeThreshold(Starmodel,UnetModel,X,Y,BaseDir, UseProbability = UseProbability, n_tiles=n_tiles)


# In[ ]:




