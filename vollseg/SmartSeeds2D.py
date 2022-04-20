#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:38:04 2019

@author: aimachine
"""

from csbdeep.utils.utils import normalize_mi_ma
import numpy as np
import os
import glob
from tifffile import imread, imwrite
from csbdeep.utils import axes_dict
from skimage.morphology import binary_dilation, binary_erosion
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.measurements import find_objects
from csbdeep.data import RawData, create_patches, create_patches_reduced_target 
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
from skimage.measure import label
from csbdeep.utils import Path, normalize
from vollseg.helpers import normalizeZero255
#from IPython.display import clear_output
from stardist.models import Config2D, StarDist2D
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
from skimage.measure import label, regionprops
from scipy import ndimage
from pathlib import Path
import cv2
    
    
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

def erode_label_holes(lbl_img, iterations):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in (range(np.min(lbl_img), np.max(lbl_img) + 1)):
        mask = lbl_img==l
        mask_filled = binary_erosion(mask,iterations = iterations)
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
    
class SmartSeeds2D(object):






     def __init__(self, base_dir, npz_filename, model_name, model_dir, n_patches_per_image, raw_dir = '/Raw/', real_mask_dir = '/real_mask/', binary_mask_dir = '/binary_mask/',
     binary_erode_mask_dir = '/binary_erode_mask/',  val_raw_dir = '/val_raw/', val_real_mask_dir = '/val_real_mask/',
     downsample_factor = 1, startfilter = 48, RGB = False, validation_split = 0.01, n_channel_in = 1,erosion_iterations = 2,
     train_seed_unet = True, train_unet = True, train_star = True, load_data_sequence = False, grid = (1,1),  generate_npz = True, patch_x=256, patch_y=256,  use_gpu = True,unet_n_first = 64,  batch_size = 1, depth = 3, kern_size = 7, n_rays = 16, epochs = 400, learning_rate = 0.0001):
         
         
         
         
         self.npz_filename = npz_filename
         self.base_dir = base_dir
         self.downsample_factor = downsample_factor
         self.train_unet = train_unet
         self.train_seed_unet = train_seed_unet
         self.train_star = train_star
         self.load_data_sequence = load_data_sequence
         self.model_dir = model_dir
         self.raw_dir = raw_dir
         self.real_mask_dir = real_mask_dir
         self.binary_mask_dir = binary_mask_dir
         self.binary_erode_mask_dir = binary_erode_mask_dir
         self.val_raw_dir = val_raw_dir
         self.val_real_mask_dir = val_real_mask_dir
         self.model_name = model_name
         self.generate_npz = generate_npz
         self.epochs = epochs
         self.learning_rate = learning_rate
         self.depth = depth
         self.n_channel_in = n_channel_in
         self.erosion_iterations = erosion_iterations
         self.n_rays = n_rays
         self.kern_size = kern_size
         self.patch_x = patch_x
         self.patch_y = patch_y
         self.RGB = RGB
         self.validation_split = validation_split
         self.startfilter = startfilter
         self.batch_size = batch_size
         self.use_gpu = use_gpu
         self.grid = grid
         self.unet_n_first = unet_n_first 
         self.n_patches_per_image =  n_patches_per_image
         
         
         #Load training and validation data
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
                                 x = ReadFloat(self.files[i]) 
                                 x = normalize(x,1,99.8,axis= self.axis_norm)
                                 x = x
                         if self.labelMe == True:
                                 #Read Label images
                                 x = read_int(self.files[i])
                                 x = x
                         return x



     def Train(self):
         
                   
                    Raw = sorted(glob.glob(self.base_dir + self.raw_dir + '*.tif'))
                    Path(self.base_dir + self.binary_mask_dir).mkdir(exist_ok=True)
                    Path(self.base_dir + self.binary_erode_mask_dir).mkdir(exist_ok=True)
                    Path(self.base_dir + self.real_mask_dir).mkdir(exist_ok=True)
                    RealMask = sorted(glob.glob(self.base_dir + self.real_mask_dir + '*.tif'))
                    ValRaw = sorted(glob.glob(self.base_dir + self.val_raw_dir + '*.tif'))        
                    ValRealMask = sorted(glob.glob(self.base_dir + self.val_real_mask_dir + '*.tif'))
                    Mask = sorted(glob.glob(self.base_dir + self.binary_mask_dir + '*.tif'))
                    ErodeMask = sorted(glob.glob(self.base_dir + self.binary_erode_mask_dir + '*.tif'))
                    
                    


                    
                    print('Instance segmentation masks:', len(RealMask))
                    print('Semantic segmentation masks:', len(Mask))
                    if self.train_star and  len(Mask) > 0 and len(RealMask) < len(Mask):
                        
                        print('Making labels')
                        Mask = sorted(glob.glob(self.base_dir + self.binary_mask_dir + '*.tif'))
                        
                        for fname in Mask:
                    
                           image = imread(fname)
                    
                           Name = os.path.basename(os.path.splitext(fname)[0])
                           if np.max(image) == 1:
                               image = image * 255
                           Binaryimage = label(image) 
                    
                           imwrite((self.base_dir + self.real_mask_dir + Name + '.tif'), Binaryimage)
                    
                    
                    if self.train_seed_unet and len(RealMask) > 0  and len(ErodeMask) < len(RealMask):
                        print('Generating Eroded Binary images')
                               
                        RealfilesMask = sorted(glob.glob(self.base_dir + self.real_mask_dir + '*tif'))  
                
                        for fname in RealfilesMask:
                    
                            image = imread(fname)
                            if self.erosion_iterations > 0:
                               image = erode_labels(image.astype('uint16'), self.erosion_iterations)
                            Name = os.path.basename(os.path.splitext(fname)[0])
                            Binaryimage = image > 0
                            imwrite((self.base_dir + self.binary_erode_mask_dir + Name + '.tif'), Binaryimage.astype('uint16'))

                    if self.train_unet and len(RealMask) > 0  and len(Mask) < len(RealMask):
                        print('Generating Binary images')
                               
                        RealfilesMask = sorted(glob.glob(self.base_dir + self.real_mask_dir + '*tif'))  
                
                        for fname in RealfilesMask:
                    
                            image = imread(fname)
                    
                            Name = os.path.basename(os.path.splitext(fname)[0])
                    
                            Binaryimage = image > 0
                    
                            imwrite((self.base_dir + self.binary_mask_dir + Name + '.tif'), Binaryimage.astype('uint16'))
                    
                    if self.generate_npz:
                      if self.RGB:
                          
                             raw_data = RawData.from_folder (
                             basepath    = self.base_dir,
                             source_dirs = [self.raw_dir],
                             target_dir  = self.binary_mask_dir,
                             axes        = 'YXC',
                              )
                           
                             X, Y, XY_axes = create_patches_reduced_target (
                             raw_data            = raw_data,
                             patch_size          = (self.patch_y,self.patch_x, None),
                             n_patches_per_image = self.n_patches_per_image,
                             patch_filter = None,
                             target_axes         = 'YX',
                             reduction_axes      = 'C',
                             save_file           = self.base_dir + self.npz_filename + '.npz',
                             
                             )
                             if self.train_seed_unet:
                                    raw_data = RawData.from_folder (
                                    basepath    = self.base_dir,
                                    source_dirs = [self.raw_dir],
                                    target_dir  = self.binary_erode_mask_dir,
                                    axes        = 'YXC',
                                    )
                                
                                    X, Y, XY_axes = create_patches_reduced_target (
                                    raw_data            = raw_data,
                                    patch_size          = (self.patch_y,self.patch_x, None),
                                    n_patches_per_image = self.n_patches_per_image,
                                    patch_filter = None,
                                    target_axes         = 'YX',
                                    reduction_axes      = 'C',
                                    save_file           = self.base_dir + self.npz_filename + "Erode" + '.npz',
                                    )
                          
                          
                          
                      else:
                              raw_data = RawData.from_folder (
                              basepath    = self.base_dir,
                              source_dirs = [self.raw_dir],
                              target_dir  = self.binary_mask_dir,
                              axes        = 'YX',
                               )
                            
                              X, Y, XY_axes = create_patches (
                              raw_data            = raw_data,
                              patch_size          = (self.patch_y,self.patch_x),
                              n_patches_per_image = self.n_patches_per_image,
                              patch_filter = None,
                              save_file           = self.base_dir + self.npz_filename + '.npz',
                              )
                              if self.train_seed_unet:
                                    raw_data = RawData.from_folder (
                                    basepath    = self.base_dir,
                                    source_dirs = [self.raw_dir],
                                    target_dir  = self.binary_erode_mask_dir,
                                    axes        = 'YX',
                                    )
                                    
                                    X, Y, XY_axes = create_patches (
                                    raw_data            = raw_data,
                                    patch_size          = (self.patch_y,self.patch_x),
                                    n_patches_per_image = self.n_patches_per_image,
                                    patch_filter = None,
                                    save_file           = self.base_dir + self.npz_filename + "Erode" + '.npz',
                                    )
                    
                  


                    if self.train_star: 
                                print('Training StarDistModel model')
                     
                                self.axis_norm = (0,1)   # normalize channels independently
                                    
                                if self.load_data_sequence == False:
                                     assert len(Raw) > 1, "not enough training data"
                                     print(len(Raw))
                                     rng = np.random.RandomState(42)
                                     ind = rng.permutation(len(Raw))

                                     X_train = list(map(ReadFloat,Raw))
                                     Y_train = list(map(read_int,RealMask))
                                     self.Y = [label(DownsampleData(y, self.downsample_factor)) for y in tqdm(Y_train)]
                                     self.X = [normalize(DownsampleData(x, self.downsample_factor),1,99.8,axis=self.axis_norm) for x in tqdm(X_train)]
                                     n_val = max(1, int(round(0.15 * len(ind))))
                                     ind_train, ind_val = ind[:-n_val], ind[-n_val:]

                                     self.X_val, self.Y_val = [self.X[i] for i in ind_val]  , [self.Y[i] for i in ind_val]
                                     self.X_trn, self.Y_trn = [self.X[i] for i in ind_train], [self.Y[i] for i in ind_train]


                                     print('number of images: %3d' % len(self.X))
                                     print('- training:       %3d' % len(self.X_trn))
                                     print('- validation:     %3d' % len(self.X_val))
                                     self.train_sample_cache = True
                                
                                if self.load_data_sequence:
                                        self.X_trn = self.DataSequencer(Raw, self.axis_norm, Normalize = True, labelMe = False)
                                        self.Y_trn = self.DataSequencer(RealMask, self.axis_norm, Normalize = False, labelMe = True)
                                    
                                        self.X_val = self.DataSequencer(ValRaw, self.axis_norm, Normalize = True, labelMe = False)
                                        self.Y_val = self.DataSequencer(ValRealMask, self.axis_norm, Normalize = False, labelMe = True)
                                        self.train_sample_cache = False
                                
                                  
                  
                                print(Config2D.__doc__)
                                
                                conf = Config2D (
                                  n_rays       = self.n_rays,
                                  train_epochs = self.epochs,
                                  train_learning_rate = self.learning_rate,
                                  unet_n_depth = self.depth ,
                                  train_patch_size = (self.patch_y,self.patch_x),
                                  n_channel_in = self.n_channel_in,
                                  unet_n_filter_base = self.unet_n_first,
                                  train_checkpoint= self.model_dir + self.model_name +'.h5',
                                  grid         = self.grid,
                                  train_loss_weights=(1, 0.05),
                                  use_gpu      = self.use_gpu,
                                  train_batch_size = self.batch_size, 
                                  train_sample_cache = self.train_sample_cache

                                  
                                  )
                                print(conf)
                                vars(conf)
                             
                            
                                Starmodel = StarDist2D(conf, name=self.model_name, basedir=self.model_dir)
                                
                                if os.path.exists(self.model_dir + self.model_name + '/' + 'weights_now.h5'):
                                    print('Loading checkpoint model')
                                    Starmodel.load_weights(self.model_dir + self.model_name + '/' + 'weights_now.h5')
                                    
                                if os.path.exists(self.model_dir + self.model_name + '/' + 'weights_last.h5'):
                                    print('Loading checkpoint model')
                                    Starmodel.load_weights(self.model_dir + self.model_name + '/' + 'weights_last.h5')
                                    
                                if os.path.exists(self.model_dir + self.model_name + '/' + 'weights_best.h5'):
                                    print('Loading checkpoint model')
                                    Starmodel.load_weights(self.model_dir + self.model_name + '/' + 'weights_best.h5')    
                             
                                Starmodel.train(self.X_trn, (self.Y_trn), validation_data=(self.X_val,(self.Y_val)), epochs = self.epochs)
                                Starmodel.optimize_thresholds(self.X_val, self.Y_val)
                   # Training UNET model
                    if self.train_unet:
                                    print('Training UNET model')
                                    load_path = self.base_dir + self.npz_filename  + '.npz'
                
                                    (X,Y), (X_val,Y_val), axes = load_training_data(load_path, validation_split= self.validation_split, verbose=True)
                                    c = axes_dict(axes)['C']
                                    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]
                                    
                                    config = Config(axes, n_channel_in, n_channel_out, unet_n_depth= self.depth,train_epochs= self.epochs, train_batch_size = self.batch_size, unet_kern_size = self.kern_size,unet_n_first = self.unet_n_first, train_learning_rate = self.learning_rate, train_reduce_lr={'patience': 5, 'factor': 0.5})
                                    print(config)
                                    vars(config)
                                    
                                    model = CARE(config , name = 'UNET' + self.model_name, basedir = self.model_dir)
                                    
                                    if self.copy_model_dir is not None:   
                                      if os.path.exists(self.copy_model_dir + 'UNET' + self.copy_model_name + '/' + 'weights_now.h5') and os.path.exists(self.model_dir + 'UNET' + self.model_name + '/' + 'weights_now.h5') == False:
                                         print('Loading copy model')
                                         model.load_weights(self.copy_model_dir + 'UNET' + self.copy_model_name + '/' + 'weights_now.h5')   
                                    
                                    if os.path.exists(self.model_dir + 'UNET' + self.model_name + '/' + 'weights_now.h5'):
                                        print('Loading checkpoint model')
                                        model.load_weights(self.model_dir + 'UNET' + self.model_name + '/' + 'weights_now.h5')
                                        
                                    if os.path.exists(self.model_dir + 'UNET' + self.model_name + '/' + 'weights_last.h5'):
                                        print('Loading checkpoint model')
                                        model.load_weights(self.model_dir + 'UNET' + self.model_name + '/' + 'weights_last.h5')
                                        
                                    if os.path.exists(self.model_dir + 'UNET' + self.model_name + '/' + 'weights_best.h5'):
                                        print('Loading checkpoint model')
                                        model.load_weights(self.model_dir + 'UNET' + self.model_name + '/' + 'weights_best.h5')    
                               
                                    
                                        
                                    history = model.train(X,Y, validation_data=(X_val,Y_val))
                                    
                                    
                     # Training UNET model
                    if self.train_seed_unet:
                                    print('Training Seed UNET model')
                                    load_path = self.base_dir + self.npz_filename + "Erode" + '.npz'
                
                                    (X,Y), (X_val,Y_val), axes = load_training_data(load_path, validation_split= self.validation_split, verbose=True)
                                    c = axes_dict(axes)['C']
                                    n_channel_in, n_channel_out = X.shape[c], Y.shape[c]
                                    
                                    config = Config(axes, n_channel_in, n_channel_out, unet_n_depth= self.depth,train_epochs= self.epochs, train_batch_size = self.batch_size, unet_kern_size = self.kern_size,unet_n_first = self.unet_n_first, train_learning_rate = self.learning_rate, train_reduce_lr={'patience': 5, 'factor': 0.5})
                                    print(config)
                                    vars(config)
                                    
                                    model = CARE(config , name = 'SeedUNET' + self.model_name, basedir = self.model_dir)
                                    
                                    
                                    
                                    if os.path.exists(self.model_dir + 'SeedUNET' + self.model_name + '/' + 'weights_now.h5'):
                                        print('Loading checkpoint model')
                                        model.load_weights(self.model_dir + 'SeedUNET' + self.model_name + '/' + 'weights_now.h5')
                                        
                                    if os.path.exists(self.model_dir + 'SeedUNET' + self.model_name + '/' + 'weights_last.h5'):
                                        print('Loading checkpoint model')
                                        model.load_weights(self.model_dir + 'SeedUNET' + self.model_name + '/' + 'weights_last.h5')
                                        
                                    if os.path.exists(self.model_dir + 'SeedUNET' + self.model_name + '/' + 'weights_best.h5'):
                                        print('Loading checkpoint model')
                                        model.load_weights(self.model_dir + 'SeedUNET' + self.model_name + '/' + 'weights_best.h5')    
                               
                                    
                                        
                                    history = model.train(X,Y, validation_data=(X_val,Y_val))                
                 
                 
def ReadFloat(fname):

    return imread(fname).astype('float32')         
         

def read_int(fname):

    return imread(fname).astype('uint16')           
         
         
         
         
def DownsampleData(image, downsample_factor):
                    


                    scale_percent = int(100/downsample_factor) # percent of original size
                    width = int(image.shape[1] * scale_percent / 100)
                    height = int(image.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    image = cv2.resize(image.astype('float32'), dim)         
         
                    return image
                  