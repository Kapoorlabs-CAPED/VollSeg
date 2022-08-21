#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed August 17 12:57:26 2022

@author: kapoorlab
"""

import numpy as np
from albumentations.augmentations.geometric.transforms import ElasticTransform
from albumentations import transforms    
from scipy import ndimage
import pandas as pd
from photutils.datasets import make_noise_image


class Augmentation2DC(object):



    """
    Augmentation creator for a TYX shape input image and labelimages
    Note:
        Only one type of augmentation can be applied for one creator.
    """
    def __init__(self,
                 rotate_angle=None, 
                 vertical_flip = None,
                 horizontal_flip = None,
                 alpha= 1,
                 alpha_affine=None,
                 mean = 0,
                 sigma = 5,
                 distribution = None,
                 brightness_limit=None,
                 contrast_limit=None,
                 brightness_by_max=True,
                 always_apply=True,
                 prob_bright_contrast=0.5,
                 multiplier= None,
                 ):
        """
        Arguments:
         
        
        rotate_angle: int or 'random'
                Angle by which image is rotated using the affine transformation matrix.
        alpha: (float) for elastic deformation
        alpha_affine (float): The range will be (-alpha_affine, alpha_affine)         
        mean: float
                Mean of the distribution used for adding noise to the image
        sigma  : float
                Standard Deviation of the distribution used for adding noise to the image and also filter for gauss kernel in elastic transform      
        distribution : "Gaussian", "Poisson", "Both"
                The tuple or a single string name for the distribution to add noise          
        brightness_limit ((float, float) or float): factor range for changing brightness.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        contrast_limit ((float, float) or float): factor range for changing contrast.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        brightness_by_max (Boolean): If True adjust contrast by image dtype maximum,
            else adjust contrast by image mean.
        prob_bright_contrast (float): probability of applying the transform. Default: 0.5.   
        multiplier (float): multiplier tuple for applying multiplicative image noise    
        """
       
        self.alpha = alpha 
        self.alpha_affine = alpha_affine
        self.rotate_angle = rotate_angle
        self.mean = mean 
        self.sigma = sigma
        self.vertical_flip = vertical_flip
        self.horizontal_flip = horizontal_flip
        self.distribution = distribution
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.brightness_by_max = brightness_by_max 
        self.always_apply = always_apply
        self.prob_bright_contrast = prob_bright_contrast
        self.multiplier = multiplier
    
    def build(self,
              image=None,
              labelimage=None
              ):
        """
        Arguments:
        build augmentor to augment input image according to initialization
        image : array
            Input image to be augmented.
            The shape of input image should have 3 dimension(y, x, c).
        labelimage : Integer labelimage images
            The shape of this labelimages should match the shape of image (y, x).
        labelcsv : oneat compatiable csv file of class and event lcoations.
            The oneat training datamaker writes T,Y,X of the selected event locations    
  
        Return:
            augmentor
        """
        if image.ndim != 3:
            raise ValueError('Input image should have 3 dimensions.')
       
        if image.ndim - 1 != labelimage.ndim:
                raise ValueError('Input image and labelimage size do not much.')

        self.image = image
        self.labelimage = labelimage
      
        self.image_dim = image.ndim
        self.image_shape = image.shape
       

        parse_dict = {}
        callback_geometric = None
        callback_intensity = None


        #Vertical flip
        if (self.vertical_flip is not None):
            callback_geometric = self._vertical_flip 
            parse_dict['vertical_flip'] = True


        #Horizontal flip
        if (self.horizontal_flip is not None):
             callback_geometric = self._horizontal_flip
             parse_dict['horizontal_flip'] = True
   

        # elastic deformation
        if (self.alpha_affine is not None):
             callback_geometric = self._elastic_deform_image
             parse_dict['alpha'] = self.alpha       
             parse_dict['sigma'] = self.sigma
             parse_dict['alpha_affine'] = self.alpha_affine

        # rotate
        if  (self.rotate_angle is not None):
            callback_geometric = self._rotate_image

            if self.rotate_angle == 'random':
                parse_dict['rotate_angle'] = np.radians(int(np.random.uniform(-180, 180)))
            elif type(self.rotate_angle) == int:
                parse_dict['rotate_angle'] = np.radians(self.rotate_angle)
            else:
                raise ValueError('Rotate angle should be int or random')

        # add additive noise
        if (self.distribution is not None):

            callback_intensity = self._noise_image  
            if self.distribution == 'Gaussian':
                parse_dict['distribution'] = 'Gaussian'
            if self.distribution == 'Poisson':
                parse_dict['distribution'] = 'Poisson'
            if self.distribution == 'Both':
                parse_dict['distribution'] = 'Both' 

            parse_dict['mean'] = self.mean
            parse_dict['sigma'] = self.sigma
                          

        # add multiplicative noise
        if (self.multiplier is not None):
                callback_intensity = self._multiplicative_noise
                parse_dict['multiplier'] = self.multiplier
                 
        # random brightness and contrast
        if (self.brightness_limit is not None) or (self.contrast_limit is not None):

            callback_intensity = self._random_bright_contrast

            parse_dict['brightness_limit'] = self.brightness_limit
            parse_dict['contrast_limit'] = self.contrast_limit
            parse_dict['brightness_by_max'] = self.brightness_by_max
            parse_dict['always_apply'] = self.always_apply
            parse_dict['prob_bright_contrast'] = self.prob_bright_contrast



        # build and return augmentor with specified callback function,  the calbacks are eitehr geometic affectging the co ordinates of the 
        # clicked locations or they are purely intensity based not affecting the csv clicked locations
        if callback_geometric is not None:
            return self._return_augmentor(callback_geometric, parse_dict)

        if callback_intensity is not None:
            return self._return_augmentor_intensity(callback_intensity, parse_dict)
        else:
            raise ValueError('No augmentor returned. Arguments are not set properly.')

    def _return_augmentor(self, callback, parse_dict):
        """return augmented image, label and csv"""

        target_image = self.image
        target_labelimage = self.labelimage
        
        # image and label augmentation by callback function
        ret_image = callback(target_image,  parse_dict) 
        ret_labelimage =  callback(target_labelimage, parse_dict) 

        return ret_image, ret_labelimage

    def _return_augmentor_intensity(self, callback, parse_dict):
        """return augmented image with same label and csv"""

        target_image = self.image
        target_labelimage = self.labelimage

        # image and label augmentation by callback function
        ret_image = callback(target_image,  parse_dict) 
        ret_labelimage =  callback(target_labelimage,  parse_dict) 

        return ret_image, ret_labelimage


    def _horizontal_flip(self, image, parse_dict):
        """ Flip the image horizontally"""
        flip_transform = transforms.HorizontalFlip()
        aug_image = transform_block(image, flip_transform) 
                            
        return aug_image

    def _vertical_flip(self, image, parse_dict):
        """ Flip the image vertically"""
        flip_transform = transforms.VerticalFlip()
        aug_image = transform_block(image, flip_transform) 
                            
        return aug_image
         
  

    def _elastic_deform_image(self, image, parse_dict):
        """ Elastically deform the image """
        alpha = parse_dict['alpha']
        alpha_affine = parse_dict['alpha_affine']
        sigma = parse_dict['sigma']
        elastic_transform = ElasticTransform(alpha = alpha, alpha_affine = alpha_affine, sigma = sigma)
        aug_image = transform_block(image, elastic_transform) 
                        
        return aug_image

    def _multiplicative_noise(self, image, parse_dict):

        """ Add multiplicative noise using the albumentations library function"""
        multiplier = parse_dict['multiplier']
        intensity_transform = transforms.MultiplicativeNoise(multiplier=multiplier)
        aug_image = transform_block(image, intensity_transform) 
                        
        return aug_image    
    

    def _random_bright_contrast(self, image, parse_dict):

        """ Add random brightness and contrast using the albumentations library function"""
         
        brightness_limit = parse_dict['brightness_limit']
        contrast_limit = parse_dict['contrast_limit']
        brightness_by_max = parse_dict['brightness_by_max']
        always_apply = parse_dict['always_apply']
        prob_bright_contrast = parse_dict['prob_bright_contrast']
        intensity_transform = transforms.RandomBrightnessContrast(brightness_limit= brightness_limit, 
            contrast_limit= contrast_limit, brightness_by_max= brightness_by_max, always_apply=always_apply, p= prob_bright_contrast) 
        aug_image = transform_block(image, intensity_transform)    


        return aug_image   
      

    def _noise_image(self, image, parse_dict):
          """ Add noise of the chosen distribution or a combination of distributions to all the timepoint of the input image"""
          mean = parse_dict['mean']
          sigma = parse_dict['sigma']
          distribution = parse_dict['distribution']   
          shape = (image.shape[0], image.shape[1])

          if distribution == 'Gaussian':
                
                addednoise = make_noise_image(shape, distribution='gaussian', mean=mean,
                          stddev=sigma)
              
          if distribution == 'Poisson':
  
                addednoise = make_noise_image(shape, distribution='poisson', mean=sigma)

          if distribution == 'Both':

                gaussiannoise = make_noise_image(shape, distribution='gaussian', mean=mean,
                          stddev=sigma)
                poissonnoise = make_noise_image(shape, distribution='poisson', mean=sigma)
            
                addednoise = gaussiannoise + poissonnoise

          else:

            raise ValueError('The distribution is not supported, has to be Gausssian, Poisson or Both (case sensitive names)')      
          
          
          aug_image = image
          if len(image.shape) == 3:
                channels = image.shape[-1]
                for i in range(channels):

                     aug_image[:,:,i] =  image[:,:,i] + addednoise  

          return aug_image

    def _rotate_image(self, image, parse_dict):
        """rotate array usiong affine transformation and also if the csv file of coordinates is supplied"""
        rotate_angle = parse_dict['rotate_angle']
        rotate_matrix =  np.array([[np.cos(rotate_angle), -np.sin(rotate_angle)], [np.sin(rotate_angle), np.cos(rotate_angle)]])
       
        aug_image = matrix_transform_block(image, rotate_matrix)
               
        return aug_image   


    
def matrix_transform_block(image, matrix):
        
        aug_image = image
        if len(image.shape) == 3:
                channels = image.shape[-1]
                for i in range(channels):
                    aug_image[:,:,i] =  ndimage.affine_transform(image[:,:,i],matrix)

        if len(image.shape) == 2:
                    aug_image =  ndimage.affine_transform(image,matrix)  

        return aug_image                 

def transform_block(image, intensity_transform):

        aug_image = image
        if len(image.shape) == 3:
                channels = image.shape[-1]
                for i in range(channels):
                     aug_image[:,:,i] =  intensity_transform.apply(image[:,:,i])  

        if len(image.shape) == 2:
         
               aug_image =  intensity_transform.apply(image)

        return aug_image       