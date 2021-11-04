# -*- coding: utf-8 -*-





import os
import glob
import sys
import numpy as np
from tqdm import tqdm
from tifffile import imread, imwrite
from stardist.models import StarDist3D
from csbdeep.models import Config, CARE
from vollseg import SmartSeedPrediction3D
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path
from n2v.models import  N2V

ImageDir = '/home/sancere/VKepler/WildTypeTest/wt12/'
Model_Dir = '/home/sancere/VKepler/CurieDeepLearningModels/'

SaveDir = ImageDir + 'Results/'


NoiseModelName = None #'ScipyDenoising'
UNETModelName = 'UNETdrosophila_fake_3D'
StarModelName = 'drosophila_fake_3D'
UnetModel = CARE(config = None, name = UNETModelName, basedir = Model_Dir)
StarModel = StarDist3D(config = None, name = StarModelName, basedir = Model_Dir)
NoiseModel = None #N2V(config=None, name=NoiseModelName, basedir=Model_Dir)

Raw_path = os.path.join(ImageDir, '*.tif')
filesRaw = glob.glob(Raw_path)
filesRaw.sort
min_size = 10
n_tiles = (64,32,32)



for fname in filesRaw:
     
     SmartSeedPrediction3D(SaveDir, fname, UnetModel, StarModel,NoiseModel, min_size = min_size,  n_tiles = n_tiles, UseProbability = True)

