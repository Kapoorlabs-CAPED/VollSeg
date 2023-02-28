import numpy as np
from torch.utils.data import Dataset
import os
import h5py
import csv
import itertools
import numpy as np

from scipy.ndimage import filters, distance_transform_edt

class PredictTiled(Dataset):
    
    """
    Dataset of fluorescently labeled cell membranes
    """
    
    def __init__(self, data_group, patch_size=(64,128,128), overlap=(10,10,10), crop=(10,10,10), data_norm='percentile',\
                 image_groups=('data/image',), 
                 dist_handling='bool', dist_scaling=(100,100), seed_handling='float', boundary_handling='bool', instance_handling='bool',\
                  reduce_dim=False, **kwargs):
           
        # Sanity checks
        assert len(patch_size)==3, 'Patch size must be 3-dimensional.'
        
        if reduce_dim:
            assert np.any([p==1 for p in patch_size]), 'Reduce is only possible, if there is a singleton patch dimension.'
        
        # Save parameters
        self.data_group = data_group
        self.patch_size = patch_size
        self.overlap = overlap
        self.crop = crop
        self.norm_method = data_norm
        self.dist_handling = dist_handling
        self.dist_scaling = dist_scaling
        self.seed_handling = seed_handling
        self.boundary_handling = boundary_handling
        self.instance_handling = instance_handling
      
        self.reduce_dim = reduce_dim
        self.image_groups = image_groups
        self.set_data()
        
    def get_fading_map(self):
               
        fading_map = np.ones(self.patch_size)
        
        if all([c==0 for c in self.crop]):
            self.crop = [1,1,1]
        
        # Exclude crop region
        crop_masking = np.zeros_like(fading_map)
        crop_masking[self.crop[0]:self.patch_size[0]-self.crop[0],\
                     self.crop[1]:self.patch_size[1]-self.crop[1],\
                     self.crop[2]:self.patch_size[2]-self.crop[2]] = 1
        fading_map = fading_map * crop_masking
            
        fading_map = distance_transform_edt(fading_map).astype(np.float32)
        
        # Normalize
        fading_map = fading_map / fading_map.max()
        
        return fading_map
    
    
    def get_whole_image(self):
        
        image = self.data_group[self.image_groups] [:]
        return image    
    
    
   
    
    def set_data(self):
        
        
        image = self.data_group[self.image_groups[0]]
        self.data_shape = image.shape[:3]
        # Calculate the position of each tile
        locations = []
        for i,p,o,c in zip(self.data_shape, self.patch_size, self.overlap, self.crop):
            # get starting coords
            coords = np.arange(np.ceil((i+o+c)/np.maximum(p-o-2*c,1)), dtype=np.int16)*np.maximum(p-o-2*c,1) -o-c
            locations.append(coords)
        self.locations = list(itertools.product(*locations))
        self.global_crop_before = np.abs(np.min(np.array(self.locations), axis=0))
        self.global_crop_after = np.array(self.data_shape) - np.max(np.array(self.locations), axis=0) - np.array(self.patch_size)
    
    
    def __len__(self):
        
        return len(self.locations)
    
    def __getitem__(self):
        
        self.patch_start = np.array(self.locations[0])
        self.patch_end = self.patch_start + np.array(self.patch_size) 
        
        pad_before = np.maximum(-self.patch_start, 0)
        pad_after = np.maximum(self.patch_end-np.array(self.data_shape), 0)
        pad_width = list(zip(pad_before, pad_after)) 
        
        slicing = tuple(map(slice, np.maximum(self.patch_start,0), self.patch_end))
        
        sample = {}
                
        # Load the mask patch
        # Load the image patch
        image = np.zeros((len(self.image_groups),)+self.patch_size, dtype=np.float32)
        for num_group, group_name in enumerate(self.image_groups):
                image_tmp = self.data_group[group_name]   
                image_tmp = image_tmp[slicing]
                
                # Pad if neccessary
                image_tmp = np.pad(image_tmp, pad_width, mode='reflect')
                
                # Store current image
                image[num_group,...] = image_tmp
        
        if self.reduce_dim:
            out_shape = [p for i,p in enumerate(image.shape) if p!=1 or i==0]
            image = np.reshape(image, out_shape)
        
        sample['image'] = image
        

                
        return sample
            
        
