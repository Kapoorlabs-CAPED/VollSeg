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

ImageDir = '/data/u934/service_imagerie/v_kapoor/Twitter/Test/'
Model_Dir = '/data/u934/service_imagerie/v_kapoor/Twitter/'

SaveDir = ImageDir + 'Results/'


NoiseModelName = 'ArabidopsisDenoising'
UNETModelName = 'UNETArabidopsisVollSeg'
StarModelName = 'ArabidopsisVollSeg'
UnetModel = CARE(config = None, name = UNETModelName, basedir = Model_Dir)
StarModel = StarDist3D(config = None, name = StarModelName, basedir = Model_Dir)
NoiseModel = N2V(config=None, name=NoiseModelName, basedir=Model_Dir)

Raw_path = os.path.join(ImageDir, '*.tif')
filesRaw = glob.glob(Raw_path)
filesRaw.sort
min_size = 10
n_tiles = (2,2,2)


fname = filesRaw[0]
test_image_dimensions = imread(fname)
if len(test_image_dimensions) > 3:
      #Split the image
      SplitImageDir = ImageDir + 'SplitTimelapse/'
      Path(SplitImageDir).mkdir(exist_ok=True)
      for fname in filesRaw:
         image = imread(fname)
         Name = os.path.basename(os.path.splitext(fname)[0])
         for i in range(image.shape[0]):

              split_image = image[i,:]
              imwrite(SplitImageDir + Name + str(i) + '.tif' , split_image.astype('float16'))
else:

       SplitImageDir = ImageDir  


for fname in SplitImageDir:
     
     SmartSeedPrediction3D(SaveDir, fname, UnetModel, StarModel,NoiseModel, min_size = min_size,  n_tiles = n_tiles, UseProbability = False)

