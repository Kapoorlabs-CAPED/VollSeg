#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import glob
import glob
from vollseg import SmartSeeds3D
from tifffile import imread, imwrite
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# # In the cell below specify the following:
# 
# 

# In[ ]:


base_dir = '/data/'
npz_filename = 'VolumeSeg'
model_dir = '/data/'
model_name = 'VolumeSeg'

raw_dir = '/Raw/'
real_mask_dir = '/real_mask/' 
binary_mask_dir = '/binary_mask/'
val_raw_dir = '/val_raw/'
val_real_mask_dir = '/val_real_mask/'


# # In this cell choose the network training parameters for the Neural Network
# 
# 

# In[ ]:


#Network training parameters
depth = 3
Epochs = 100
learning_rate = 1.0E-4
batch_size = 1
patch_x = 128
patch_y = 128
patch_z = 16
kern_size = 3
n_patches_per_image = 16
n_rays = 128
startfilter = 48
use_gpu_opencl = True
generate_npz = True
backbone = 'resnet'
load_data_sequence = True
validation_split = 0.01
n_channel_in = 1
train_unet = True
train_star = True


# # Generate the npz file first and then train the model

# In[ ]:



SmartSeeds3D(base_dir = base_dir, 
             npz_filename = npz_filename, 
             model_name = model_name, 
             model_dir = model_dir,
             raw_dir = raw_dir,
             real_mask_dir = real_mask_dir,
             binary_mask_dir = binary_mask_dir,
             val_raw_dir = val_raw_dir,
             val_real_mask_dir = val_real_mask_dir,
             n_channel_in = n_channel_in,
             backbone = backbone, 
             load_data_sequence = load_data_sequence, 
             validation_split = validation_split, 
             n_patches_per_image = n_patches_per_image, 
             generate_npz = generate_npz, 
             train_unet = train_unet, 
             train_star = train_star, 
             patch_x= patch_x, 
             patch_y= patch_y, 
             patch_z = patch_z,  
             use_gpu = use_gpu_opencl,  
             batch_size = batch_size, 
             depth = depth, 
             kern_size = kern_size, 
             startfilter = startfilter, 
             n_rays = n_rays, 
             epochs = epochs, 
             learning_rate = learning_rate)

