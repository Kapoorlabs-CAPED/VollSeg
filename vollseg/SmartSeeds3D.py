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
import glob
from csbdeep.io import load_training_data
from csbdeep.utils import axes_dict
from csbdeep.models import Config, CARE
from tensorflow.keras.utils import Sequence
from csbdeep.data import RawData, create_patches
from skimage.measure import label, regionprops
from scipy import ndimage
from tqdm import tqdm
from .utils import plot_train_history
import matplotlib.pyplot as plt
from pathlib import Path
from tifffile import imread, imwrite
from csbdeep.utils import  plot_history
from scipy.ndimage import zoom

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

class SmartSeeds3D(object):





     def __init__(self, base_dir,  model_name, model_dir,npz_filename = None, n_patches_per_image = 1, train_loss = 'mae', raw_dir = 'Raw/', real_mask_dir = 'real_mask/', binary_mask_dir = 'binary_mask/',
      val_raw_dir = 'val_raw/', val_real_mask_dir = 'val_real_mask/', n_channel_in = 1, pattern = '.tif' ,  downsample_factor = 1, backbone = 'resnet', load_data_sequence = True, train_unet = True, train_star = True, generate_npz = True,
      validation_split = 0.01, erosion_iterations = 2, patch_x=256, patch_y=256, patch_z = 16, grid_x = 1, grid_y = 1, annisotropy = (1,1,1),  use_gpu = True,  batch_size = 4, depth = 3, kern_size = 3, startfilter = 48, n_rays = 16, epochs = 400, learning_rate = 0.0001):

         
         
         
         
         self.npz_filename = npz_filename
         self.base_dir = base_dir
         self.downsample_factor = downsample_factor
         self.model_dir = model_dir
         self.backbone = backbone
         self.raw_dir = raw_dir
         self.real_mask_dir = real_mask_dir
         self.val_raw_dir = val_raw_dir
         self.val_real_mask_dir = val_real_mask_dir
         self.binary_mask_dir = binary_mask_dir
         self.generate_npz = generate_npz
         self.annisotropy = annisotropy
         self.train_unet = train_unet
         self.train_star = train_star
         self.model_name = model_name
         self.epochs = epochs
         self.learning_rate = learning_rate
         self.depth = depth
         self.n_channel_in = n_channel_in
         self.n_rays = n_rays
         self.pattern = pattern
         self.train_loss = train_loss
         self.erosion_iterations = erosion_iterations
         self.kern_size = kern_size
         self.patch_x = patch_x
         self.patch_y = patch_y
         self.patch_z = patch_z
         self.grid_x = grid_x
         self.grid_y = grid_y
         self.validation_split = validation_split
         self.batch_size = batch_size
         self.use_gpu = use_gpu
         self.search_pattern = '*' + self.pattern
         self.startfilter = startfilter
         self.n_patches_per_image =  n_patches_per_image
         self.load_data_sequence = load_data_sequence
         self.Train()
         
     class DataSequencer(Sequence):
         
         
            def __init__(self, files, axis_norm, Normalize = True, labelMe = False):
                super().__init__() 
                
                self.files = files
               
                self.axis_norm = axis_norm
                self.labelMe = labelMe
                self.Normalize = Normalize
                
            def __len__(self):
                return len(self.files)
            
            
            def __getitem__(self, i):
                
                        #Read Raw images
                         if self.Normalize == True:
                                 x = read_float(self.files[i]) 
                                 x = normalize(x,1,99.8,axis= self.axis_norm)
                                 x = x
                         if self.labelMe == True:
                                 #Read Label images
                                 x = read_int(self.files[i])
                                 x = x
                         return x
        
        
         
     def Train(self):
         
         
         

                    Raw_path = Path(self.base_dir + self.raw_dir)
                    Raw = list(Raw_path.glob(self.search_pattern))

                    Val_Raw_path = Path(self.base_dir + self.val_raw_dir)
                    ValRaw = list(Val_Raw_path.glob(self.search_pattern))
                    
                    Mask_path = Path(self.base_dir + self.binary_mask_dir)
                    Mask_path.mkdir(exist_ok=True)
                    Mask = list(Mask_path.glob(self.search_pattern))

                    Real_Mask_path = Path(self.base_dir + self.real_mask_dir)
                    Real_Mask_path.mkdir(exist_ok=True)
                    RealMask = list(Real_Mask_path.glob(self.search_pattern))
    
                    Val_Real_Mask_path = Path(self.base_dir + self.val_real_mask_dir)
                    Val_Real_Mask_path.mkdir(exist_ok=True)
                    ValRealMask = list(Val_Real_Mask_path.glob(self.search_pattern))

                 
                    print('Instance segmentation masks:', len(RealMask))
                    print('Semantic segmentation masks:', len(Mask))
                    if self.train_star and  len(Mask) > 0 and len(RealMask) < len(Mask):
                        
                        print('Making labels')
                        Mask = sorted(glob.glob(self.base_dir + self.binary_mask_dir + '*' +  self.pattern))
                        
                        for fname in Mask:
                    
                           image = imread(fname)
                    
                           Name = os.path.basename(os.path.splitext(fname)[0])
                           if np.max(image) == 1:
                               image = image * 255
                           Binaryimage = label(image) 
                    
                           imwrite((self.base_dir + self.real_mask_dir + Name + self.pattern), Binaryimage.astype('uint16'))
                           
                
                    
                    
                    if len(RealMask) > 0  and len(Mask) < len(RealMask):
                        print('Generating Binary images')
               
                               
                        RealfilesMask = sorted(glob.glob(self.base_dir + self.real_mask_dir +'*' +  self.pattern))  
                
                
                        for fname in RealfilesMask:
                    
                            image = imread(fname)
                            if self.erosion_iterations > 0:
                               image = erode_labels(image.astype('uint16'), self.erosion_iterations)
                            Name = os.path.basename(os.path.splitext(fname)[0])
                    
                            Binaryimage = image > 0
                    
                            imwrite((self.base_dir + self.binary_mask_dir + Name + self.pattern), Binaryimage.astype('uint16'))
                            
                    if self.generate_npz:
                        
                      raw_data = RawData.from_folder (
                      basepath    = self.base_dir,
                      source_dirs = [self.raw_dir],
                      target_dir  = self.binary_mask_dir,
                      pattern = self.search_pattern,
                      axes        = 'ZYX',
                       )
                    
                      X, Y, XY_axes = create_patches (
                      raw_data            = raw_data,
                      patch_size          = (self.patch_z,self.patch_y,self.patch_x),
                      n_patches_per_image = self.n_patches_per_image,
                      save_file           = self.base_dir + self.npz_filename + '.npz',
                      )        
                            
                    
                    # Training UNET model
                    if self.train_unet:
                            print('Training UNET model')
                            load_path = self.base_dir + self.npz_filename + '.npz'
        
                            (X,Y), (X_val,Y_val), axes = load_training_data(load_path, validation_split=self.validation_split, verbose=True)
                            c = axes_dict(axes)['C']
                            n_channel_in, n_channel_out = X.shape[c], Y.shape[c]
                            
                            config = Config(axes, n_channel_in, n_channel_out, unet_n_depth= self.depth,train_epochs= self.epochs, train_batch_size = self.batch_size, unet_n_first = self.startfilter, train_loss = self.train_loss, unet_kern_size = self.kern_size, train_learning_rate = self.learning_rate, train_reduce_lr={'patience': 5, 'factor': 0.5})
                            print(config)
                            vars(config)
                            
                            model = CARE(config , name = 'unet_' + self.model_name, basedir = self.model_dir)
                            
                            if os.path.exists(self.model_dir + 'unet_' + self.model_name + '/' + 'weights_now.h5'):
                                print('Loading checkpoint model')
                                model.load_weights(self.model_dir + 'unet_' + self.model_name + '/' + 'weights_now.h5')
                            
                            if os.path.exists(self.model_dir + 'unet_' + self.model_name + '/' + 'weights_last.h5'):
                                print('Loading checkpoint model')
                                model.load_weights(self.model_dir + 'unet_' + self.model_name + '/' + 'weights_last.h5')
                                
                            if os.path.exists(self.model_dir + 'unet_' + self.model_name + '/' + 'weights_best.h5'):
                                print('Loading checkpoint model')
                                model.load_weights(self.model_dir + 'unet_' + self.model_name + '/' + 'weights_best.h5')    
                            
                            history = model.train(X,Y, validation_data=(X_val,Y_val))
                            
                            print(sorted(list(history.history.keys())))
                            plt.figure(figsize=(16,5))
                            plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae'])

                    if self.train_star:
                            print('Training StarDistModel model with' , self.backbone , 'backbone')
                            self.axis_norm = (0,1,2)
                            if self.load_data_sequence == False:
                                     assert len(Raw) > 1, "not enough training data"
                                     print(len(Raw))
                                     rng = np.random.RandomState(42)
                                     ind = rng.permutation(len(Raw))

                                     X_train = list(map(read_float,Raw))
                                     Y_train = list(map(read_int,RealMask))
                                     self.Y = [label(DownsampleData(y, self.downsample_factor)) for y in tqdm(Y_train)]
                                     self.X = [normalize(DownsampleData(x, self.downsample_factor),1,99.8,axis=self.axis_norm) for x in tqdm(X_train)]
                                     n_val = max(1, int(round(self.validation_split * len(ind))))
                                     ind_train, ind_val = ind[:-n_val], ind[-n_val:]

                                     self.X_val, self.Y_val = [self.X[i] for i in ind_val]  , [self.Y[i] for i in ind_val]
                                     self.X_trn, self.Y_trn = [self.X[i] for i in ind_train], [self.Y[i] for i in ind_train]


                                     print('number of images: %3d' % len(self.X))
                                     print('- training:       %3d' % len(self.X_trn))
                                     print('- validation:     %3d' % len(self.X_val))
 

                            
                            if self.load_data_sequence:
                                        self.X_trn = self.DataSequencer(Raw, self.axis_norm, Normalize = True, labelMe = False)
                                        self.Y_trn = self.DataSequencer(RealMask, self.axis_norm, Normalize = False, labelMe = True)
                                    
                                        self.X_val = self.DataSequencer(ValRaw, self.axis_norm, Normalize = True, labelMe = False)
                                        self.Y_val = self.DataSequencer(ValRealMask, self.axis_norm, Normalize = False, labelMe = True)
                                        self.train_sample_cache = False
                           
                            

                          
                            print(Config3D.__doc__)
                           
                            extents = calculate_extents(self.Y_trn)
                            self.annisotropy = tuple(np.max(extents) / extents)
                            rays = Rays_GoldenSpiral(self.n_rays, anisotropy=self.annisotropy)
                                    
                                    
                            if self.backbone == 'resnet':
                                 
                                
                                conf = Config3D (
                                  rays       = rays,
                                  anisotropy = self.annisotropy,
                                  backbone = self.backbone,
                                  train_epochs = self.epochs,
                                  train_learning_rate = self.learning_rate,
                                  resnet_n_blocks = self.depth,
                                  train_checkpoint = self.model_dir + self.model_name +'.h5',
                                  resnet_kernel_size = (self.kern_size, self.kern_size, self.kern_size),
                                  train_patch_size = (self.patch_z, self.patch_x, self.patch_y ),
                                  train_batch_size = self.batch_size,
                                  resnet_n_filter_base = self.startfilter,
                                  train_dist_loss = 'mse',
                                  grid         = (1,self.grid_y,self.grid_x),
                                  use_gpu      = self.use_gpu,
                                  n_channel_in = self.n_channel_in
                                  )
                                
                            if self.backbone == 'unet':
                                
                                conf = Config3D (
                                  rays       = rays,
                                  anisotropy = self.annisotropy,
                                  backbone = self.backbone,
                                  train_epochs = self.epochs,
                                  train_learning_rate = self.learning_rate,
                                  unet_n_depth = self.depth,
                                  train_checkpoint = self.model_dir + self.model_name +'.h5',
                                  unet_kernel_size = (self.kern_size, self.kern_size, self.kern_size),
                                  train_patch_size = (self.patch_z, self.patch_x, self.patch_y ),
                                  train_batch_size = self.batch_size,
                                  unet_n_filter_base = self.startfilter,
                                  train_dist_loss = 'mse',
                                  grid         = (1,self.grid_y,self.grid_x),
                                  use_gpu      = self.use_gpu,
                                  n_channel_in = self.n_channel_in,
                                  train_sample_cache = False
                                  )
                                
                            
                            

                            
                            print(conf)
                            vars(conf)
                                 
                                
                            Starmodel = StarDist3D(conf, name=self.model_name, basedir=self.model_dir)
                            print(Starmodel._axes_tile_overlap('ZYX'), os.path.exists(self.model_dir + self.model_name + '/' + 'weights_now.h5'))                            
                                 
                                 
                            
                            if os.path.exists(self.model_dir + self.model_name + '/' + 'weights_now.h5'):
                                print('Loading checkpoint model')
                                Starmodel.load_weights(self.model_dir + self.model_name + '/' + 'weights_now.h5')
                                
                            if os.path.exists(self.model_dir + self.model_name + '/' + 'weights_last.h5'):
                                print('Loading checkpoint model')
                                Starmodel.load_weights(self.model_dir + self.model_name + '/' + 'weights_last.h5')   
                                
                            if os.path.exists(self.model_dir + self.model_name + '/' + 'weights_best.h5'):
                                print('Loading checkpoint model')
                                Starmodel.load_weights(self.model_dir + self.model_name + '/' + 'weights_best.h5')     
                                 
                            historyStar = Starmodel.train(self.X_trn, self.Y_trn, validation_data=(self.X_val,self.Y_val), epochs = self.epochs)
                            print(sorted(list(historyStar.history.keys())))
                            plt.figure(figsize=(16,5))
                            plot_history(historyStar,['loss','val_loss'],['dist_relevant_mae','val_dist_relevant_mae','dist_relevant_mse','val_dist_relevant_mse'])
        
        
                 
         
def read_float(fname):

    return imread(fname).astype('float32')         
         

def read_int(fname):

    return imread(fname).astype('uint16')         



         
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
         
         
