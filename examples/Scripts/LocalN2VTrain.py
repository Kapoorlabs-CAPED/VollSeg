#!/usr/bin/env python
# coding: utf-8

# In[ ]:



# We import all our dependencies.
from n2v.models import N2VConfig, N2V
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from csbdeep.utils import plot_history,plot_some
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
import urllib
import os
import zipfile


# In[ ]:


BaseDir = '/data/u934/service_imagerie/v_kapoor/Twitter/'

Model_Dir = '/data/u934/service_imagerie/v_kapoor/Twitter/'
Model_Name = 'ArabDenoising'
datagen = N2V_DataGenerator()

imgs = datagen.load_imgs_from_directory(directory = BaseDir + "Raw/", dims='ZYX')




# In[ ]:


patch_shape = (64, 256, 256)
Epochs = 200
NetworkDepth = 3
batch_size = 1
patches = datagen.generate_patches_from_list(imgs[:1], shape=patch_shape)
print(patches.shape)
X = patches[:14]
X_val = patches[14:]




# In[ ]:


config = N2VConfig(X, unet_kern_size=3, unet_n_depth = NetworkDepth,
                   train_steps_per_epoch=400,train_epochs= Epochs, train_loss='mse', batch_norm=True, 
                   train_batch_size=batch_size, n2v_perc_pix=0.198,n2v_patch_shape= patch_shape, 
                   n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=4)

# Let's look at the parameters stored in the config-object.
vars(config)
model = N2V(config=config, name=Model_Name, basedir=BaseDir)


# In[ ]:



history = model.train(X, X_val)











