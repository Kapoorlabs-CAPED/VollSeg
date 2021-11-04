#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import glob
import glob
from vollseg import SmartSeeds3D
from tifffile import imread, imwrite
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# # In the cell below specify the following:
# 
# 1) Directory where the training data is, inside this directory there should be the two subfolders called Raw and Mask. Inside the Raw folder are the raw images and inside the Mask folder are the labelled images.
# 
# 2) The training data for doing UNET training is stored in NPZ format so please specify the NPZ filename which is suitable for your data.
# 
# 3) Model directory is where the trained Neural network models are stored, please chooose a location if you want to change the default location which is where the training data is.
# 
# 4) Copy Model name is optional, in case you have a previouis trained model and want to re-train it on new data but store it with a new name.
# 
# 5) Model name is the unique name of the trained models.

# In[ ]:


Data_dir = '/data/u934/service_imagerie/v_kapoor/CurieTrainingDatasets/Drosophila_fake_3D/'
NPZ_filename = 'drosophila_fake_3D'
Model_dir = '/data/u934/service_imagerie/v_kapoor/CurieDeepLearningModels/'
Model_Name = 'drosophila_fake_3D'


# # In this cell choose the network training parameters for the Neural Network
# 
# 1) NetworkDepth = Depth of the network, with each increasing depth the image is downsampled by 2 hence the XYZ dimension of the data / 2^depth has to be greater than 1.
# 
# 2) Epochs, training for longer epochs ensures a well converged network and requires longer GPU runtimes.
# 
# 3) Learning rate is the parameter which controls the step size used in the optimization process and it should not be greater than 0.001 at the start of the training.
# 
# 4) batch size controls the number of images used for doing stochastic gradient descent and is a parameter that is limited by the GPU RAM available, if you do not have a lot of ran batch size < 10 should be optimal. 
# 
# 5) PatchX,Y,Z is the patch size used for making patches out of the iamge data. The original image is broken down into patches for training. Patch size is chosen based on having enough context for the network to learn but at the same time not being too big to obscure memory usage.
# 
# 6) Kernel is the receptive field of the neural network, usual choices are 3,5 or 7 but not larger than that. This is the size of the convolutional kernel used in the network
# 
# 7) n_patches_per_image is the number of patches sampled for each image to create the npz file, choose an optimal value so that the file is not too big for the computer memory. 
# 
# 8) Rays is the number of rays used the learn the distance map, low rays decreases the spatial resoultion and high rays are able to resolve the shape better.
# 
# 
# 9) OpenCL is a boolean parameter that is set true if you want to do some opencl computations on the GPU, this requires GPU tools but if you do not have them set this to false.
# 
# Some optimal values have been chosen by default and should work well for any NVDIA enabled GPU computer

# In[ ]:


#Network training parameters
NetworkDepth = 3
Epochs = 200
LearningRate = 1.0E-4
batch_size = 1
PatchX = 128
PatchY = 128
PatchZ = 64
Kernel = 3
n_patches_per_image = 400
Rays = 192
startfilter = 32
use_gpu_opencl = False
GenerateNPZ = True
TrainUNET = True
TrainSTAR = True


# # Generate the npz file first and then train the model

# In[ ]:



SmartSeeds3D(BaseDir = Data_dir, backbone = 'unet', NPZfilename = NPZ_filename, model_name = Model_Name, model_dir = Model_dir, n_patches_per_image = n_patches_per_image,GenerateNPZ = GenerateNPZ, CroppedLoad = False, TrainUNET = TrainUNET, TrainSTAR = TrainSTAR, PatchX= PatchX, PatchY= PatchY, PatchZ = PatchZ,  use_gpu = use_gpu_opencl,  batch_size = batch_size, depth = NetworkDepth, kern_size = Kernel, startfilter = startfilter, n_rays = Rays, epochs = Epochs, learning_rate = LearningRate)


# In[ ]:




# In[ ]:




