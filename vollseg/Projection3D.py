#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:40:47 2019

@author: aimachine
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:38:04 2019

@author: aimachine
"""

import numpy as np
import os
#from IPython.display import clear_output
from stardist.models import Config3D, StarDist3D
from stardist import  Rays_GoldenSpiral,calculate_extents
from scipy.ndimage import binary_fill_holes
from scipy.ndimage.measurements import find_objects
from scipy.ndimage import  binary_dilation
from csbdeep.utils import normalize
from csbdeep.io import load_training_data
from csbdeep.utils import axes_dict
from csbdeep.models import ProjectionConfig, ProjectionCARE
from tifffile import imread
from tensorflow.keras.utils import Sequence
from csbdeep.data import RawData, create_patches_reduced_target
from skimage.measure import label, regionprops
from scipy import ndimage
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from tifffile import imread, imwrite
from csbdeep.utils import  plot_history
def _raise(e):
    raise e

def _fill_label_holes(lbl_img, **kwargs):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in (set(np.unique(lbl_img)) - set([0])):
        mask = lbl_img==l
        mask_filled = binary_fill_holes(mask,**kwargs)
        lbl_img_filled[mask_filled] = l
    return lbl_img_filled
def fill_label_holes(lbl_img, **kwargs):
    """Fill small holes in label image."""
    # TODO: refactor 'fill_label_holes' and 'edt_prob' to share code
    def grow(sl,interior):
        return tuple(slice(s.start-int(w[0]),s.stop+int(w[1])) for s,w in zip(sl,interior))
    def shrink(interior):
        return tuple(slice(int(w[0]),(-1 if w[1] else None)) for w in interior)
    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)
    for i,sl in enumerate(objects,1):
        if sl is None: continue
        interior = [(s.start>0,s.stop<sz) for s,sz in zip(sl,lbl_img.shape)]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl,interior)]==i
        mask_filled = binary_fill_holes(grown_mask,**kwargs)[shrink_slice]
        lbl_img_filled[sl][mask_filled] = i
    return lbl_img_filled


def dilate_label_holes(lbl_img, iterations):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in (range(np.min(lbl_img), np.max(lbl_img) + 1)):
        mask = lbl_img==l
        mask_filled = binary_dilation(mask,iterations = iterations)
        lbl_img_filled[mask_filled] = l
    return lbl_img_filled     
    
def erode_labels(segmentation, erosion_iterations= 2):
    # create empty list where the eroded masks can be saved to
    list_of_eroded_masks = list()
    regions = regionprops(segmentation)
    erode = np.zeros(segmentation.shape)
    def erode_mask(segmentation_labels, label_id, erosion_iterations):
        
        only_current_label_id = np.where(segmentation_labels == label_id, 1, 0)
        eroded = ndimage.binary_erosion(only_current_label_id, iterations = erosion_iterations)
        relabeled_eroded = np.where(eroded == 1, label_id, 0)
        return(relabeled_eroded)

    for i in range(len(regions)):
        label_id = regions[i].label
        erode = erode + erode_mask(segmentation, label_id, erosion_iterations)

    # convert list of numpy arrays to stacked numpy array
    return erode

class Projection3D(object):





     def __init__(self, base_dir, npz_filename, model_name, model_dir, n_patches_per_image, raw_dir = 'Raw/',  projection_dir = 'projection/',
      val_raw_dir = 'val_raw/', val_projection_dir = 'val_projection/', n_channel_in = 1,  downsample_factor = 1,   generate_npz = True,
      validation_split = 0.01,  patch_x=256, patch_y=256, grid_x = 1, grid_y = 1,   use_gpu = True,  batch_size = 4, depth = 3, kern_size = 3, startfilter = 48, epochs = 400, learning_rate = 0.0001):

         
         
         
         
         self.npz_filename = npz_filename
         self.base_dir = base_dir
         self.downsample_factor = downsample_factor
         self.model_dir = model_dir
         self.raw_dir = raw_dir
         self.projection_dir = projection_dir
         self.val_raw_dir = val_raw_dir
         self.val_projection_dir = val_projection_dir
         self.projection_dir = projection_dir
         self.generate_npz = generate_npz
         self.model_name = model_name
         self.epochs = epochs
         self.learning_rate = learning_rate
         self.depth = depth
         self.n_channel_in = n_channel_in
         self.kern_size = kern_size
         self.patch_x = patch_x
         self.patch_y = patch_y
         self.grid_x = grid_x
         self.grid_y = grid_y
         self.validation_split = validation_split
         self.batch_size = batch_size
         self.use_gpu = use_gpu
         self.startfilter = startfilter
         self.n_patches_per_image =  n_patches_per_image
       
         self.Train()
         
    
        
        
         
     def Train(self):
         
         
         

                   
                    if self.generate_npz:
                        
                      raw_data = RawData.from_folder (
                      basepath    = self.base_dir,
                      source_dirs = [self.raw_dir],
                      target_dir  = self.projection_dir,
                      axes        = 'ZYX',
                       )
                    
                      X, Y, XY_axes = create_patches_reduced_target (
                      raw_data            = raw_data,
                      patch_size          = (None,self.patch_y,self.patch_x),
                      n_patches_per_image = self.n_patches_per_image,
                      target_axes         = 'YX',
                      reduction_axes      = 'Z',
                      save_file           = self.base_dir + self.npz_filename + '.npz',
                      )        
                            
                    
                    print('Training UNET model')
                    load_path = self.base_dir + self.npz_filename + '.npz'

                    (X,Y), (X_val,Y_val), axes = load_training_data(load_path, validation_split=self.validation_split, verbose=True)
                    c = axes_dict(axes)['C']
                    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]
                    
                    config = ProjectionConfig(axes, n_channel_in, n_channel_out, unet_n_depth= self.depth,train_epochs= self.epochs, train_batch_size = self.batch_size, unet_n_first = self.startfilter, train_loss = 'mse', unet_kern_size = self.kern_size, train_learning_rate = self.learning_rate, train_reduce_lr={'patience': 5, 'factor': 0.5})
                    print(config)
                    vars(config)
                    
                    model = ProjectionCARE(config , name = 'projection_' + self.model_name, basedir = self.model_dir)
                            
                    
                    
                    if os.path.exists(self.model_dir + 'projection_' + self.model_name + '/' + 'weights_now.h5'):
                        print('Loading checkpoint model')
                        model.load_weights(self.model_dir + 'projection_' + self.model_name + '/' + 'weights_now.h5')
                    
                    if os.path.exists(self.model_dir + 'projection_' + self.model_name + '/' + 'weights_last.h5'):
                        print('Loading checkpoint model')
                        model.load_weights(self.model_dir + 'projection_' + self.model_name + '/' + 'weights_last.h5')
                        
                    if os.path.exists(self.model_dir + 'projection_' + self.model_name + '/' + 'weights_best.h5'):
                        print('Loading checkpoint model')
                        model.load_weights(self.model_dir + 'projection_' + self.model_name + '/' + 'weights_best.h5')    
                    
                    history = model.train(X,Y, validation_data=(X_val,Y_val))
                    
                    print(sorted(list(history.history.keys())))
                   
            
         
def read_float(fname):

    return imread(fname).astype('float32')         
         



         
def DownsampleData(image, downsample_factor):
                    


                    scale_percent = int(100/downsample_factor) # percent of original size
                    width = int(image.shape[2] * scale_percent / 100)
                    height = int(image.shape[1] * scale_percent / 100)
                    dim = (width, height)
                    smallimage = np.zeros([image.shape[0],  height,width])
                    for i in range(0, image.shape[0]):
                          # resize image
                          smallimage[i,:] = zoom(image[i,:].astype('float32'), dim)         
         
                    return smallimage
         
         
