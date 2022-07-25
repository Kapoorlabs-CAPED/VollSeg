#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
from vollseg import SmartSeeds3D
from tifffile import imread, imwrite
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# # In the cell below specify the following:
# 
# 

# In[ ]:


base_dir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data/TrainingData/'
npz_filename = 'XenopusNucleiSeg'
model_dir = '/gpfsstore/rech/jsy/uzj81mi/Mari_Data/Models/Unet3D/Unet_Nuclei_Xenopus/'
model_name = 'Unet_Nuclei_Xenopus'

raw_dir = 'Raw/'
real_mask_dir = 'real_mask/' 
binary_mask_dir = 'binary_mask/'


# # In this cell choose the network training parameters for the Neural Network
# 
# 

# In[ ]:


#Network training parameters
depth = 3
epochs = 200
learning_rate = 1.0E-4
batch_size = 1
patch_x = 200
patch_y = 200
patch_z = 16
kern_size = 3
n_patches_per_image = 32
n_rays = 96
startfilter = 32
use_gpu_opencl = False
generate_npz = True
backbone = 'resnet'
load_data_sequence = False
validation_split = 0.01
n_channel_in = 1
train_unet = True
train_star = False


# # Generate the npz file first and then train the model

# In[ ]:



SmartSeeds3D(base_dir = base_dir, 
             npz_filename = npz_filename, 
             model_name = model_name, 
             model_dir = model_dir,
             raw_dir = raw_dir,
             real_mask_dir = real_mask_dir,
             binary_mask_dir = binary_mask_dir,
             n_channel_in = n_channel_in,
             erosion_iterations = 0,
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

