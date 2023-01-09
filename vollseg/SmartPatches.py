import os
import numpy as np 
from pathlib import Path
from tifffile import imread, imwrite
from skimage.measure import  regionprops

class SmartPatches(object):
    
    def __init__(self, base_dir,  raw_dir, real_mask_dir, raw_save_dir, real_mask_patch_dir, binary_mask_dir, patch_size, erosion_iterations = 2, pattern = '.tif', lower_ratio_fore_to_back = 0.3,
     upper_ratio_fore_to_back = 0.9):
        
        self.base_dir = base_dir
        self.raw_dir = os.path.join(base_dir,raw_dir) 
        self.real_mask_dir = os.path.join(base_dir,real_mask_dir)
        self.raw_save_dir = os.path.join(base_dir,raw_save_dir) 
        self.binary_mask_dir = os.path.join(base_dir,binary_mask_dir)  
        self.real_mask_patch_dir = os.path.join(base_dir, real_mask_patch_dir)
        self.patch_size = patch_size
        self.erosion_iterations = erosion_iterations
        self.pattern = pattern 
        self.lower_ratio_fore_to_back = lower_ratio_fore_to_back
        self.upper_ratio_fore_to_back = upper_ratio_fore_to_back
        
        
        
        self._create_smart_patches()  
        
        def _create_smart_patches(self):
            
            Path(self.raw_save_dir).mkdir(exist_ok = True)
            Path(self.binary_mask_dir).mkdir(exist_ok = True)
            Path(self.real_mask_patch_dir).mkdir(exist_ok = True)
            for fname in os.listdir(self.real_mask_dir):
                
                
                labelimage = imread(fname).astype(np.uint16)
                self.ndim = len(labelimage.shape)
                properties = regionprops(labelimage)
                for count, prop in enumerate(properties):
                          self._label_maker( fname, labelimage , count , prop )
                
        def _label_maker(self, fname, labelimage, count, prop):
            
                    name = os.path.splitext(os.path.basename(fname))[0]
                    
                    if self.ndim == 2:
                        
                        self.valid = False
                        centroid = prop.centroid
                        x = centroid[1]
                        y = centroid[0]

                        crop_Xminus = x  - int(self.patch_size[1]/2)
                        crop_Xplus = x   + int(self.patch_size[1]/2)
                        crop_Yminus = y  - int(self.patch_size[0]/2)
                        crop_Yplus = y   + int(self.patch_size[0]/2)
                        region =(slice(int(crop_Yminus), int(crop_Yplus)),
                                                                    slice(int(crop_Xminus), int(crop_Xplus)))
                    if self.ndim == 3:
                        
                            self.valid = False
                            centroid = prop.centroid
                            z = centroid[2]
                            x = centroid[1]
                            y = centroid[0]

                            crop_Xminus = x  - int(self.patch_size[1]/2)
                            crop_Xplus = x   + int(self.patch_size[1]/2)
                            crop_Yminus = y  - int(self.patch_size[0]/2)
                            crop_Yplus = y   + int(self.patch_size[0]/2)
                            crop_Zminus = z  - int(self.patch_size[2]/2)
                            crop_Zplus = z   + int(self.patch_size[2]/2)
                            
                            region =(slice(int(crop_Zminus), int(crop_Zplus)),slice(int(crop_Yminus), int(crop_Yplus)),
                                                                        slice(int(crop_Xminus), int(crop_Xplus)))
                            
                        
                    self.crop_labelimage = labelimage[region] 
                    self.crop_labelimage = remove_small_objects(
                                self.crop_labelimage.astype('uint16'), min_size=10)
                    if self.crop_labelimage.shape[0] == self.patch_size[0] and self.crop_labelimage.shape[1] == self.patch_size[1]:
                                self._region_selector()

                                if self.valid:

                                    imwrite(self.real_mask_patch_dir + '/' + name + str(count) + self.pattern, self.crop_labelimage.astype('uint16'))
                                    binary_image = self.crop_labelimage > 0   
                                    imwrite(self.binary_mask_dir + '/' + name + str(count) + self.pattern, binary_image.astype('uint16'))

                                    if self.erosion_iterations > 0:
                                        eroded_crop_labelimage = erode_labels(self.crop_labelimage.astype('uint16'), self.erosion_iterations)
                                    eroded_binary_image = eroded_crop_labelimage > 0   
                                    imwrite(self.binary_mask_dir + '/' + name + str(count) + self.pattern, eroded_binary_image.astype('uint16'))

                                    self.raw_image = imread(Path(self.raw_dir + name + self.pattern ))[region]
                                    
                                    imwrite(self.base_dir + self.raw_save_dir + '/' + name + str(count) + self.pattern, self.raw_image)
                        
                        
                        
        def _region_selector(self):
            
                non_zero_indices = list(zip(*np.where(self.crop_labelimage > 0)))
 
                total_indices = list(zip(*np.where(self.crop_labelimage >= 0)))
                if len(total_indices) > 0:
                  norm_foreground = len(non_zero_indices)/ len(total_indices)
                  index_ratio = float(norm_foreground) 
                  if index_ratio >= self.lower_ratio_fore_to_back  and index_ratio <= self.upper_ratio_fore_to_back:

                      self.valid = True                