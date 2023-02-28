import numpy as np
from torch.utils.data import Dataset
import os
import h5py
import csv
import itertools
import numpy as np

from scipy.ndimage import filters, distance_transform_edt

class MeristemH5Tiler(Dataset):
    
    """
    Dataset of fluorescently labeled cell membranes
    """
    
    def __init__(self, data_group, patch_size=(64,128,128), overlap=(10,10,10), crop=(10,10,10), data_norm='percentile',\
                 image_groups=('data/image',), mask_groups=('data/distance', 'data/seeds', 'data/boundary'), \
                 dist_handling='bool', dist_scaling=(100,100), seed_handling='float', boundary_handling='bool', instance_handling='bool',\
                 no_mask=False, no_img=False, reduce_dim=False, **kwargs):
           
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
        self.no_mask = no_mask
        self.no_img = no_img
        self.reduce_dim = reduce_dim
        
        # Read the filelist and construct full paths to each file
        self.image_groups = image_groups
        self.mask_groups = mask_groups
        
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
    
    
    def get_whole_mask(self, mask_groups=None): 
        
        if mask_groups is None:
            mask_groups = self.mask_groups
        if not isinstance(mask_groups, (list,tuple)):
            mask_groups = [mask_groups]
        
        mask = None
        for num_group, group_name in enumerate(mask_groups):
                mask_tmp = self.data_group[group_name]
                if mask is None:
                    mask = np.zeros((len(mask_groups),)+mask_tmp.shape, dtype=np.float32)                
                mask[num_group,...] = mask_tmp
        return mask        
    
    
    def set_data_idx(self, idx):
        
        # Restrict the idx to the amount of data available
        idx = idx%len(self.data_list)
        self.data_idx = idx
        
        # Get the current data size
        if not self.no_img:
                image = self.data_group[self.image_groups[0]]
                self.data_shape = image.shape[:3]
        elif not self.no_mask:
                mask = self.data_group[self.mask_groups[0]]
                self.data_shape = mask.shape[:3]
        else:
            raise ValueError('Can not determine data shape!')
            
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
    
    
    def _normalize(self, data, group_name):
        
        # Normalization
            
        if 'distance' in group_name:
            if self.dist_handling == 'float':
                data /= self.dist_scaling[0]
            elif self.dist_handling == 'bool':
                data = data<0
            elif self.dist_handling == 'bool_inv':
                data = data>=0
            elif self.dist_handling == 'exp':
                data = (data/self.dist_scaling[0])**3
            elif self.dist_handling == 'tanh':    
                foreground = np.float16(data>0)
                data = np.tanh(data/self.dist_scaling[0])*foreground + np.tanh(data/self.dist_scaling[1])*(1-foreground)
            
        elif 'seed' in group_name:                    
            if self.seed_handling == 'float':
                data = data.astype(np.float32)
                data = filters.gaussian_filter(data, 2)
                if np.max(data)>1e-4: data /= float(np.max(data))
            elif self.seed_handling == 'bool':
                data = data>0.1
            
        elif 'instance' in group_name or 'nuclei' in group_name:
            if self.instance_handling == 'bool':
                data = data>0
            
        elif 'boundary' in group_name:
            if self.boundary_handling == 'bool':
                data = data>0
                
        elif 'image' in group_name:
            data = data.astype(np.float32)
            data -= self.norm1
            data /= self.norm2
            if self.norm_method == 'minmax' or self.norm_method == 'percentile':
                data = np.clip(data, 1e-5, 1)
                
        return data
    
    
    def __getitem__(self, idx):
        
        self.patch_start = np.array(self.locations[idx])
        self.patch_end = self.patch_start + np.array(self.patch_size) 
        
        pad_before = np.maximum(-self.patch_start, 0)
        pad_after = np.maximum(self.patch_end-np.array(self.data_shape), 0)
        pad_width = list(zip(pad_before, pad_after)) 
        
        slicing = tuple(map(slice, np.maximum(self.patch_start,0), self.patch_end))
        
        sample = {}
                
        # Load the mask patch
        if not self.no_mask:            
            mask = np.zeros((len(self.mask_groups),)+self.patch_size, dtype=np.float32)
            for num_group, group_name in enumerate(self.mask_groups):
                    mask_tmp = self.data_group[group_name]
                    mask_tmp = mask_tmp[slicing]
                    
                    # Pad if neccessary
                    mask_tmp = np.pad(mask_tmp, pad_width, mode='reflect')
                    
                     # Normalization
                    mask_tmp = self._normalize(mask_tmp, group_name)
                    
                    # Store current mask
                    mask[num_group,...] = mask_tmp
                    
            mask = mask.astype(np.float32)
            
            if self.reduce_dim:
                out_shape = [p for i,p in enumerate(mask.shape) if p!=1 or i==0]
                mask = np.reshape(mask, out_shape)
            
            sample['mask'] = mask
            
            
        if not self.no_img:
            # Load the image patch
            image = np.zeros((len(self.image_groups),)+self.patch_size, dtype=np.float32)
            for num_group, group_name in enumerate(self.image_groups):
                    image_tmp = self.data_group[group_name]   
                    image_tmp = image_tmp[slicing]
                    
                    # Pad if neccessary
                    image_tmp = np.pad(image_tmp, pad_width, mode='reflect')
                    
                    # Normalization
                    image_tmp = self._normalize(image_tmp, group_name)
                    
                    # Store current image
                    image[num_group,...] = image_tmp
            
            if self.reduce_dim:
                out_shape = [p for i,p in enumerate(image.shape) if p!=1 or i==0]
                image = np.reshape(image, out_shape)
            
            sample['image'] = image
        

                
        return sample
            
        
