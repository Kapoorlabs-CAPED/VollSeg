#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
from vollseg import SmartSeeds2D
from tifffile import imread, imwrite
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# # In the cell below specify the following:
# 
# 

# In[ ]:


base_dir = '/gpfsstore/rech/jsy/uzj81mi/Segmentation_Datasets/Drosophila_Segmentation/TrainingData/Incremental/'
npz_filename = 'DrosophilaSeg'
model_dir = '/gpfsstore/rech/jsy/uzj81mi/Segmentation_Models/'
model_name = 'DrosophilaSeg'

raw_dir = 'Raw/'
real_mask_dir = 'real_mask/' 
binary_mask_dir = 'binary_mask/'
binary_erode_mask_dir = 'binary_erode_mask/'

# # In this cell choose the network training parameters for the Neural Network
# 
# 

# In[ ]:


#Network training parameters
depth = 3
epochs = 200
learning_rate = 1.0E-4
batch_size = 10
patch_x = 384
patch_y = 384
kern_size = 3
n_patches_per_image = 96
n_rays = 64
startfilter = 32
use_gpu_opencl = True
generate_npz = True
load_data_sequence = False
validation_split = 0.01
n_channel_in = 1
train_unet = True
train_star = True


# # Generate the npz file first and then train the model

# In[ ]:



SmartSeeds2D(base_dir = base_dir, 
             npz_filename = npz_filename, 
             model_name = model_name, 
             model_dir = model_dir,
             raw_dir = raw_dir,
             real_mask_dir = real_mask_dir,
             binary_mask_dir = binary_mask_dir,
             binary_erode_mask_dir = binary_erode_mask_dir,
             n_channel_in = n_channel_in,
             load_data_sequence = load_data_sequence, 
             validation_split = validation_split, 
             n_patches_per_image = n_patches_per_image, 
             generate_npz = generate_npz, 
             train_unet = train_unet, 
             train_star = train_star, 
             train_seed_unet = True,
             patch_x= patch_x, 
             patch_y= patch_y, 
             use_gpu = use_gpu_opencl,  
             batch_size = batch_size, 
             depth = depth, 
             kern_size = kern_size, 
             startfilter = startfilter, 
             n_rays = n_rays, 
             epochs = epochs, 
             learning_rate = learning_rate)

