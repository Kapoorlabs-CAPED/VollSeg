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
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.morphology import  binary_dilation
from csbdeep.utils import normalize
import glob
import cv2
from csbdeep.io import load_training_data
from csbdeep.utils import axes_dict
from csbdeep.models import Config, CARE
from tifffile import imread
from six.moves import range, zip
from tensorflow.keras.utils import Sequence
from csbdeep.data import RawData, create_patches
from skimage.measure import label
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
    
class SmartSeeds3D(object):





     def __init__(self, BaseDir, NPZfilename, model_name, model_dir, n_patches_per_image, DownsampleFactor = 1, backbone = 'resnet', CroppedLoad = False, TrainUNET = True, TrainSTAR = True, GenerateNPZ = True,  copy_model_dir = None, PatchX=256, PatchY=256, PatchZ = 16, gridX = 1, gridY = 1, annisotropy = (1,1,1),  use_gpu = True,  batch_size = 4, depth = 3, kern_size = 3, startfilter = 48, n_rays = 16, epochs = 400, learning_rate = 0.0001):

         
         
         
         
         self.NPZfilename = NPZfilename
         self.BaseDir = BaseDir
         self.DownsampleFactor = DownsampleFactor
         self.model_dir = model_dir
         self.backbone = backbone
         self.GenerateNPZ = GenerateNPZ
         self.annisotropy = annisotropy
         self.TrainUNET = TrainUNET
         self.TrainSTAR = TrainSTAR
         self.copy_model_dir = copy_model_dir
         self.model_name = model_name
         self.epochs = epochs
         self.learning_rate = learning_rate
         self.depth = depth
         self.n_rays = n_rays
         self.kern_size = kern_size
         self.PatchX = PatchX
         self.PatchY = PatchY
         self.PatchZ = PatchZ
         self.gridX = gridX
         self.gridY = gridY
         self.batch_size = batch_size
         self.use_gpu = use_gpu
         self.startfilter = startfilter
         self.n_patches_per_image =  n_patches_per_image
         self.CroppedLoad = CroppedLoad
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
                                 x = ReadInt(self.files[i])
                                 x = x
                         return x
        
        
         
     def Train(self):
         
         
         

                    BinaryName = 'BinaryMask/' 
                    RealName = 'RealMask/'
                    Raw = sorted(glob.glob(self.BaseDir + '/Raw/' + '*.tif'))
                    Path(self.BaseDir + '/' + BinaryName).mkdir(exist_ok=True)
                    Path(self.BaseDir + '/' + RealName).mkdir(exist_ok=True)
                    RealMask = sorted(glob.glob(self.BaseDir + '/' + RealName + '*.tif'))
                    ValRaw = sorted(glob.glob(self.BaseDir + '/ValRaw/' + '*.tif'))        
                    ValRealMask = sorted(glob.glob(self.BaseDir + '/ValRealMask/' + '*.tif'))

                    
                      
                 
                    print('Instance segmentation masks:', len(RealMask))
                    if len(RealMask)== 0:
                        
                        print('Making labels')
                        Mask = sorted(glob.glob(self.BaseDir + '/' + BinaryName + '*.tif'))
                        
                        for fname in Mask:
                    
                           image = imread(fname)
                    
                           Name = os.path.basename(os.path.splitext(fname)[0])
                    
                           Binaryimage = label(image) 
                    
                           imwrite((self.BaseDir + '/' + RealName + Name + '.tif'), Binaryimage.astype('uint16'))
                           
                
                    Mask = sorted(glob.glob(self.BaseDir + '/' + BinaryName + '*.tif'))
                    print('Semantic segmentation masks:', len(Mask))
                    if len(Mask) == 0:
                        print('Generating Binary images')
               
                               
                        RealfilesMask = sorted(glob.glob(self.BaseDir + '/' + RealName + '*tif'))  
                
                
                        for fname in RealfilesMask:
                    
                            image = imread(fname)
                    
                            Name = os.path.basename(os.path.splitext(fname)[0])
                    
                            Binaryimage = image > 0
                    
                            imwrite((self.BaseDir + '/' + BinaryName + Name + '.tif'), Binaryimage.astype('uint16'))
                            
                    if self.GenerateNPZ:
                        
                      raw_data = RawData.from_folder (
                      basepath    = self.BaseDir,
                      source_dirs = ['Raw/'],
                      target_dir  = 'BinaryMask/',
                      axes        = 'ZYX',
                       )
                    
                      X, Y, XY_axes = create_patches (
                      raw_data            = raw_data,
                      patch_size          = (self.PatchZ,self.PatchY,self.PatchX),
                      n_patches_per_image = self.n_patches_per_image,
                      save_file           = self.BaseDir + self.NPZfilename + '.npz',
                      )        
                            
                    
                    # Training UNET model
                    if self.TrainUNET:
                            print('Training UNET model')
                            load_path = self.BaseDir + self.NPZfilename + '.npz'
        
                            (X,Y), (X_val,Y_val), axes = load_training_data(load_path, validation_split=0.1, verbose=True)
                            c = axes_dict(axes)['C']
                            n_channel_in, n_channel_out = X.shape[c], Y.shape[c]
                            
                            config = Config(axes, n_channel_in, n_channel_out, unet_n_depth= self.depth,train_epochs= self.epochs, train_batch_size = self.batch_size, unet_n_first = self.startfilter, train_loss = 'mse', unet_kern_size = self.kern_size, train_learning_rate = self.learning_rate, train_reduce_lr={'patience': 5, 'factor': 0.5})
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
                            
                            print(sorted(list(history.history.keys())))
                            plt.figure(figsize=(16,5))
                            plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae'])

                    if self.TrainSTAR:
                            print('Training StarDistModel model with' , self.backbone , 'backbone')
                            self.axis_norm = (0,1,2)
                            if self.CroppedLoad == False:
                                     assert len(Raw) > 1, "not enough training data"
                                     print(len(Raw))
                                     rng = np.random.RandomState(42)
                                     ind = rng.permutation(len(Raw))

                                     X_train = list(map(ReadFloat,Raw))
                                     Y_train = list(map(ReadInt,RealMask))
                                     self.Y = [label(DownsampleData(y, self.DownsampleFactor)) for y in tqdm(Y_train)]
                                     self.X = [normalize(DownsampleData(x, self.DownsampleFactor),1,99.8,axis=self.axis_norm) for x in tqdm(X_train)]
                                     n_val = max(1, int(round(0.15 * len(ind))))
                                     ind_train, ind_val = ind[:-n_val], ind[-n_val:]

                                     self.X_val, self.Y_val = [self.X[i] for i in ind_val]  , [self.Y[i] for i in ind_val]
                                     self.X_trn, self.Y_trn = [self.X[i] for i in ind_train], [self.Y[i] for i in ind_train]


                                     print('number of images: %3d' % len(self.X))
                                     print('- training:       %3d' % len(self.X_trn))
                                     print('- validation:     %3d' % len(self.X_val))
 

                            
                            if self.CroppedLoad:
                                        self.X_trn = self.DataSequencer(Raw, self.axis_norm, Normalize = True, labelMe = False)
                                        self.Y_trn = self.DataSequencer(RealMask, self.axis_norm, Normalize = False, labelMe = True)
                                    
                                        self.X_val = self.DataSequencer(ValRaw, self.axis_norm, Normalize = True, labelMe = False)
                                        self.Y_val = self.DataSequencer(ValRealMask, self.axis_norm, Normalize = False, labelMe = True)
                                        self.train_sample_cache = False
                           
                            

                          
                            print(Config3D.__doc__)
                           
                            
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
                                  train_patch_size = (self.PatchZ, self.PatchX, self.PatchY ),
                                  train_batch_size = self.batch_size,
                                  resnet_n_filter_base = self.startfilter,
                                  train_dist_loss = 'mse',
                                  grid         = (1,self.gridY,self.gridX),
                                  use_gpu      = self.use_gpu,
                                  n_channel_in = 1
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
                                  train_patch_size = (self.PatchZ, self.PatchX, self.PatchY ),
                                  train_batch_size = self.batch_size,
                                  unet_n_filter_base = self.startfilter,
                                  train_dist_loss = 'mse',
                                  grid         = (1,self.gridY,self.gridX),
                                  use_gpu      = self.use_gpu,
                                  n_channel_in = 1,
                                  train_sample_cache = False
                                  )
                                
                            
                            

                            
                            print(conf)
                            vars(conf)
                                 
                                
                            Starmodel = StarDist3D(conf, name=self.model_name, basedir=self.model_dir)
                            print(Starmodel._axes_tile_overlap('ZYX'), os.path.exists(self.model_dir + self.model_name + '/' + 'weights_now.h5'))                            
                                 
                                 
                            if self.copy_model_dir is not None:   
                              if os.path.exists(self.copy_model_dir + self.copy_model_name + '/' + 'weights_now.h5') and os.path.exists(self.model_dir + self.model_name + '/' + 'weights_now.h5') == False:
                                 print('Loading copy model')
                                 Starmodel.load_weights(self.copy_model_dir + self.copy_model_name + '/' + 'weights_now.h5')  
                              if os.path.exists(self.copy_model_dir + self.copy_model_name + '/' + 'weights_last.h5') and os.path.exists(self.model_dir + self.model_name + '/' + 'weights_last.h5') == False:
                                 print('Loading copy model')
                                 Starmodel.load_weights(self.copy_model_dir + self.copy_model_name + '/' + 'weights_last.h5')

                              if os.path.exists(self.copy_model_dir + self.copy_model_name + '/' + 'weights_best.h5') and os.path.exists(self.model_dir + self.model_name + '/' + 'weights_best.h5') == False:
                                 print('Loading copy model')
                                 Starmodel.load_weights(self.copy_model_dir + self.copy_model_name + '/' + 'weights_best.h5')

 
                            
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
        
        
                 
         
def ReadFloat(fname):

    return imread(fname).astype('float32')         
         

def ReadInt(fname):

    return imread(fname).astype('uint16')         



         
def DownsampleData(image, DownsampleFactor):
                    


                    scale_percent = int(100/DownsampleFactor) # percent of original size
                    width = int(image.shape[2] * scale_percent / 100)
                    height = int(image.shape[1] * scale_percent / 100)
                    dim = (width, height)
                    smallimage = np.zeros([image.shape[0],  height,width])
                    for i in range(0, image.shape[0]):
                          # resize image
                          smallimage[i,:] = cv2.resize(image[i,:].astype('float32'), dim)         
         
                    return smallimage
         
         
