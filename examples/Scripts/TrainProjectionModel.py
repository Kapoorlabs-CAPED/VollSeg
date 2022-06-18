#!/usr/bin/env python
# coding: utf-8

# In[1]:

from vollseg import Projection3D
from tifffile import imread, imwrite


# # In the cell below specify the following:
# 
# 

# In[ ]:


base_dir = '/data/'
npz_filename = 'VolumeProjection'
model_dir = '/data/'
model_name = 'VolumeProjection'

raw_dir = 'Raw/'
projection_dir = 'real_mask/' 



# # In this cell choose the network training parameters for the Neural Network
# 
# 

# In[ ]:


#Network training parameters
depth = 3
epochs = 100
learning_rate = 1.0E-4
batch_size = 1
patch_x = 128
patch_y = 128
kern_size = 3
n_patches_per_image = 16
startfilter = 48
generate_npz = True
validation_split = 0.01
n_channel_in = 1



# # Generate the npz file first and then train the model

# In[ ]:



Projection3D(base_dir = base_dir, 
             npz_filename = npz_filename, 
             model_name = model_name, 
             model_dir = model_dir,
             raw_dir = raw_dir,
             projection_dir = projection_dir,
             n_channel_in = n_channel_in,
             validation_split = validation_split, 
             n_patches_per_image = n_patches_per_image, 
             generate_npz = generate_npz, 
             patch_x= patch_x, 
             patch_y= patch_y, 
             batch_size = batch_size, 
             depth = depth, 
             kern_size = kern_size, 
             startfilter = startfilter, 
             epochs = epochs, 
             learning_rate = learning_rate)

