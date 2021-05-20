#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:08:41 2019
@author: vkapoor
"""

from __future__ import print_function, unicode_literals, absolute_import, division
#import matplotlib.pyplot as plt
import numpy as np
import os
import collections
from tifffile import imread, imwrite
from skimage import morphology
from skimage.morphology import dilation, square
import cv2
from skimage.filters import gaussian
from six.moves import reduce
from matplotlib import cm
from skimage.filters import threshold_local, threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
from skimage.segmentation import watershed
from pathlib import Path
from skimage.segmentation import  relabel_sequential
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.morphology import  binary_dilation, binary_erosion
from skimage.util import invert as invertimage
from skimage import measure
from scipy.ndimage.filters import gaussian_filter
from skimage.measure import label
from csbdeep.utils import normalize
from skimage import filters
from skimage.util import random_noise


globalthreshold = 0.01
def SimplePrediction(x, UnetModel, StarModel, n_tiles = (2,2), UseProbability = True, min_size = 20):
    
               
    
             
                                           
                      Mask = UNETPrediction3D(x, UnetModel, min_size, n_tiles, 'YX')
                      
                      SmartSeeds, _, StarImage = STARPrediction3D(x, StarModel, min_size, n_tiles, MaskImage = Mask, smartcorrection = None, UseProbability = UseProbability)
                      
                      SmartSeeds = SmartSeeds.astype('uint16') 
                     
                
                
                      return SmartSeeds

def crappify_flou_G_P(x, y, mu, sigma, savedirx, savediry, name):
    x = x.astype('float32')
    gaussiannoise = np.random.normal(mu, sigma*0.05, x.shape)
    x = x + gaussiannoise 
        
    #add noise to original image
    imwrite(savedirx + '/' + name + 'pg' + str(mu) + str(sigma) + '.tif', x.astype('float32'))    
    #keep the label the same
    imwrite(savediry + '/' + name + 'pg' + str(mu) + str(sigma) + '.tif', y.astype('uint16'))     


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


def remove_big_objects(ar, max_size=6400, connectivity=1, in_place=False):
    
    out = ar.copy()
    ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")



    too_big = component_sizes > max_size
    too_big_mask = too_big[ccs]
    out[too_big_mask] = 0

    return out

def multiplotline(plotA, plotB, plotC, titleA, titleB, titleC, targetdir = None, File = None, plotTitle = None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()
    ax[0].plot(plotA)
    ax[0].set_title(titleA)
   
    ax[1].plot(plotB)
    ax[1].set_title(titleB)
    
    ax[2].plot(plotC)
    ax[2].set_title(titleC)
    
    plt.tight_layout()
    
    if plotTitle is not None:
      Title = plotTitle
    else :
      Title = 'MultiPlot'   
    if targetdir is not None and File is not None:
      plt.savefig(targetdir + Title + File + '.png')
    if targetdir is not None and File is None:
      plt.savefig(targetdir + Title + File + '.png')
    plt.show()




def BinaryDilation(Image, iterations = 1):

    DilatedImage = binary_dilation(Image, iterations = iterations) 
    
    return DilatedImage


def CCLabels(fname, max_size = 15000):
    
    
    BinaryImageOriginal = imread(fname)
    Orig = normalizeFloatZeroOne(BinaryImageOriginal)
    InvertedBinaryImage = invertimage(BinaryImageOriginal)
    BinaryImage = normalizeFloatZeroOne(InvertedBinaryImage)
    image = binary_dilation(BinaryImage)
    image = invertimage(image)
    IntegerImage = label(image)
    labelclean = remove_big_objects(IntegerImage, max_size = max_size) 
    AugmentedLabel = dilation(labelclean, selem = square(3) )
    AugmentedLabel = np.multiply(AugmentedLabel ,  Orig)
    

    return AugmentedLabel 




    

def SmartSeedPrediction3D( SaveDir, fname,  UnetModel, StarModel, NoiseModel = None, min_size_mask = 100, min_size = 10, 
n_tiles = (1,2,2), doMask = True, smartcorrection = None, threshold = 20, projection = False, UseProbability = True, filtersize = 0):
    
    
    
    print('Generating SmartSeed results')
    UNETResults = SaveDir + 'BinaryMask/'
    SmartSeedsResults = SaveDir + 'SmartSeedsMask/' 
    StarDistResults = SaveDir + 'StarDist/'
    Path(SaveDir).mkdir(exist_ok = True)
    Path(SmartSeedsResults).mkdir(exist_ok = True)
    Path(StarDistResults).mkdir(exist_ok = True)
    Path(UNETResults).mkdir(exist_ok = True)
    
    #Read Image
    image = imread(fname)
    sizeZ = image.shape[0]
    sizeY = image.shape[1]
    sizeX = image.shape[2]
    
    SizedMask = np.zeros([sizeZ, sizeY, sizeX], dtype = 'uint16')
    SizedSmartSeeds = np.zeros([sizeZ, sizeY, sizeX], dtype = 'uint16')
    Name = os.path.basename(os.path.splitext(fname)[0])
    if NoiseModel is not None:
         image = NoiseModel.predict(image, axes='ZYX', n_tiles=n_tiles)
    Mask = UNETPrediction3D(gaussian_filter(image, filtersize), UnetModel, n_tiles, 'ZYX')
    for i in range(0, Mask.shape[0]):
        Mask[i,:] = remove_small_objects(Mask[i,:].astype('uint16'), min_size = min_size)
    
    SizedMask[:, :Mask.shape[1], :Mask.shape[2]] = Mask
    

    SmartSeeds, _, StarImage = STARPrediction3D(gaussian_filter(image,filtersize), StarModel,  n_tiles, MaskImage = Mask, UseProbability = UseProbability, smartcorrection = smartcorrection)
    #Upsample images back to original size
    for i in range(0, Mask.shape[0]):
        SmartSeeds[i,:] = remove_small_objects(SmartSeeds[i,:].astype('uint16'), min_size = min_size)
        
    SmartSeeds = RemoveLabels(SmartSeeds)       
    SizedSmartSeeds[:, :SmartSeeds.shape[1], :SmartSeeds.shape[2]] = SmartSeeds
            
    imwrite((StarDistResults + Name+ '.tif' ) , StarImage.astype('uint16'))
    imwrite((SmartSeedsResults + Name+ '.tif' ) , SizedSmartSeeds.astype('uint16'))
    imwrite((UNETResults + Name+ '.tif' ) , SizedMask.astype('uint16')) 
    
        
    return SizedSmartSeeds, SizedMask    





def DownsampleData(image, DownsampleFactor):
                    
                if DownsampleFactor!=1:  
                    print('Downsampling Image in XY by', DownsampleFactor)
                    scale_percent = int(100/DownsampleFactor) # percent of original size
                    width = int(image.shape[2] * scale_percent / 100)
                    height = int(image.shape[1] * scale_percent / 100)
                    dim = (width, height)
                    smallimage = np.zeros([image.shape[0],  height,width])
                    for i in range(0, image.shape[0]):
                          # resize image
                          smallimage[i,:] = cv2.resize(image[i,:].astype('float32'), dim)         
         
                    return smallimage
                else:
                    
                    return image
                




def UNETPrediction3D(image, model, n_tiles, axis):
    
    
    Segmented = model.predict(image, axis, n_tiles = n_tiles)
    
    try:
       thresh = threshold_otsu(Segmented)
       Binary = Segmented > thresh
    except:
        Binary = Segmented > 0
    #Postprocessing steps
    Filled = binary_fill_holes(Binary)
    Finalimage = label(Filled)
    Finalimage = fill_label_holes(Finalimage)
    Finalimage = relabel_sequential(Finalimage)[0]
    
          
    return Finalimage

def RemoveLabels(LabelImage, minZ = 2):
    
    properties = measure.regionprops(LabelImage, LabelImage)
    for prop in properties:
                regionlabel = prop.label
                sizeZ = abs(prop.bbox[0] - prop.bbox[3])
                if sizeZ <= minZ:
                    LabelImage[LabelImage == regionlabel] = 0
    return LabelImage                

def STARPrediction3D(image, model, n_tiles, MaskImage = None, smartcorrection = None, UseProbability = True):
    
    copymodel = model
    image = normalize(image, 1, 99.8, axis = (0,1,2))
    shape = [image.shape[1], image.shape[2]]
    image = zero_pad_time(image, 64, 64)
    grid = copymodel.config.grid

    try:
         MidImage, details = model.predict_instances(image, n_tiles = n_tiles)
         SmallProbability, SmallDistance = model.predict(image, n_tiles = n_tiles)

    except:
            conf = copymodel.config
            Dummy = StarDist3D(conf)
            overlap = Dummy._axes_tile_overlap('ZYX')
            model._tile_overlap = [overlap]
            MidImage, details = model.predict_instances(image, n_tiles = n_tiles)
            SmallProbability, SmallDistance = model.predict(image, n_tiles = n_tiles)

    StarImage = MidImage[:image.shape[0],:shape[0],:shape[1]]
    SmallDistance = MaxProjectDist(SmallDistance, axis=-1)
    Probability = np.zeros([SmallProbability.shape[0] * grid[0],SmallProbability.shape[1] * grid[1], SmallProbability.shape[2] * grid[2] ])
    Distance = np.zeros([SmallDistance.shape[0] * grid[0], SmallDistance.shape[1] * grid[1], SmallDistance.shape[2] * grid[2] ])
    #We only allow for the grid parameter to be 1 along the Z axis
    for i in range(0, SmallProbability.shape[0]):
        Probability[i,:] = cv2.resize(SmallProbability[i,:], dsize=(SmallProbability.shape[2] * grid[2] , SmallProbability.shape[1] * grid[1] ))
        Distance[i,:] = cv2.resize(SmallDistance[i,:], dsize=(SmallDistance.shape[2] * grid[2] , SmallDistance.shape[1] * grid[1] ))
    
    if UseProbability:
        
        
        Probability[Probability < globalthreshold ] = 0 
             
        MaxProjectDistance = Probability[:image.shape[0],:shape[0],:shape[1]]

    else:
        
        MaxProjectDistance = Distance[:image.shape[0],:shape[0],:shape[1]]

    
          
    Watershed, Markers = WatershedwithMask3D(MaxProjectDistance.astype('uint16'), StarImage.astype('uint16'), MaskImage.astype('uint16'), grid )
    Watershed = fill_label_holes(Watershed.astype('uint16'))
  
       
       

    return Watershed, MaxProjectDistance, StarImage  
 
 
def VetoRegions(Image, Zratio = 3):
    
    Image = Image.astype('uint16')
    
    properties = measure.regionprops(Image, Image)
    
    for prop in properties:
        
        LabelImage = prop.image
        if LabelImage.shape[0] < Image.shape[0]/Zratio :
            indices = zip(*np.where(LabelImage > 0))
            for z, y, x in indices:

                 Image[z,y,x] = 0

    return Image
    

#Default method that works well with cells which are below a certain shape and do not have weak edges
    
def iou3D(boxA, centroid):
    
    ndim = len(centroid)
    inside = False
    
    Condition = [Conditioncheck(centroid, boxA, p, ndim) for p in range(0,ndim)]
        
    inside = all(Condition)
    
    return inside

def Conditioncheck(centroid, boxA, p, ndim):
    
      condition = False
    
      if centroid[p] >= boxA[p] and centroid[p] <= boxA[p + ndim]:
          
           condition = True
           
      return condition     
    

def WatershedwithMask3D(Image, Label,mask, grid): 
    properties = measure.regionprops(Label, Image) 
    binaryproperties = measure.regionprops(label(mask), Image) 
    
    
    Coordinates = [prop.centroid for prop in properties] 
    BinaryCoordinates = [prop.centroid for prop in binaryproperties]
    
    Binarybbox = [prop.bbox for prop in binaryproperties]
    Coordinates = sorted(Coordinates , key=lambda k: [k[0], k[1], k[2]]) 
    
    if len(Binarybbox) > 0:    
            for i in range(0, len(Binarybbox)):
                
                box = Binarybbox[i]
                inside = [iou3D(box, star) for star in Coordinates]
                
                if not any(inside) :
                         Coordinates.append(BinaryCoordinates[i])    
                         
    
    Coordinates.append((0,0,0))
   

    Coordinates = np.asarray(Coordinates)
    coordinates_int = np.round(Coordinates).astype(int) 
    
    markers_raw = np.zeros_like(Image) 
    markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates)) 
    markers = morphology.dilation(markers_raw.astype('uint16'), morphology.ball(2))
    
    watershedImage = watershed(-Image, markers, mask = mask.copy()) 


    
    return watershedImage, markers



def WatershedSmartCorrection3D(Image, Label, mask, grid, smartcorrection = 20, max_size = 100000):
    
    
   
    CopyDist = Image.copy()
    try:
       thresh = threshold_otsu(CopyDist)
    except:
        thresh = 0   
    CopyDist = CopyDist > thresh
    ThinCopyDist = np.zeros([CopyDist.shape[0],CopyDist.shape[1],CopyDist.shape[2]])
    for i in range(0, CopyDist.shape[0]):
       ThinCopyDist[i,:] = thin(CopyDist[i,:] , max_iter = smartcorrection//4)
  
    ThinCopyDist = label(ThinCopyDist)


    ## Use markers from Label image
    Labelproperties = measure.regionprops(Label, Image)
    LabelCoordinates = [prop.centroid for prop in Labelproperties] 
    LabelCoordinates.append((0,0,0))
    LabelCoordinates = sorted(LabelCoordinates , key=lambda k: [k[1], k[0], k[2]])
    LabelCoordinates = np.asarray(LabelCoordinates)
    sexyImage = np.zeros_like(Image)
    Labelcoordinates_int = np.round(LabelCoordinates).astype(int)
    
    Labelmarkers_raw = np.zeros([Image.shape[0], Image.shape[1],Image.shape[2] ]) 
    if(len(LabelCoordinates) > 0) :
     Labelmarkers_raw[tuple(Labelcoordinates_int.T)] = 1 + np.arange(len(LabelCoordinates))
     
     Labelmarkers = morphology.dilation(Labelmarkers_raw.astype('uint16'), morphology.ball(5))
  

   
    for i in range(0, Image.shape[0]):
        Image[i,:] = sobel(Image[i,:].astype('uint16'))


    watershedImage = watershed(Image, markers = Labelmarkers)

    TestCopyDist = np.zeros([CopyDist.shape[0],CopyDist.shape[1],CopyDist.shape[2]])
    for i in range(0, CopyDist.shape[0]):
       TestCopyDist[i,:] = thin(CopyDist[i,:] , max_iter = smartcorrection//2)

    watershedImage[TestCopyDist == 0] = 0
    sexyImage = watershedImage
    copymask = mask.copy()
    
    Binary = watershedImage > 0
   
    if smartcorrection > 0:
       indices = list(zip(*np.where(Binary>0)))
       if(len(indices) > 0):
        indices = np.asarray(indices)
        tree = spatial.cKDTree(indices)
        copymask = copymask - Binary
        maskindices = list(zip(*((np.where(copymask>0)))))
        maskindices = np.asarray(maskindices)
    
        for i in (range(0,maskindices.shape[0])):
    
           pt = maskindices[i]
           closest =  tree.query(pt)
        
           if closest[0] < smartcorrection:
               sexyImage[pt[0], pt[1]] = watershedImage[indices[closest[1]][0], indices[closest[1]][1]]  
       
    sexyImage = fill_label_holes(sexyImage)
    sexyImage, forward_map, inverse_map = relabel_sequential(sexyImage)
    
    
    return sexyImage, Labelmarkers 
    
def Integer_to_border(Label, max_size = 6400):

        SmallLabel = remove_big_objects(Label, max_size = max_size)
        BoundaryLabel =  find_boundaries(SmallLabel, mode='outer')
           
        Binary = BoundaryLabel > 0
        
        return Binary
        
def zero_pad(image, PadX, PadY):

          sizeY = image.shape[1]
          sizeX = image.shape[0]
          
          sizeXextend = sizeX
          sizeYextend = sizeY
         
 
          while sizeXextend%PadX!=0:
              sizeXextend = sizeXextend + 1
        
          while sizeYextend%PadY!=0:
              sizeYextend = sizeYextend + 1

          extendimage = np.zeros([sizeXextend, sizeYextend])
          
          extendimage[0:sizeX, 0:sizeY] = image
              
              
          return extendimage 
    
        
def zero_pad_color(image, PadX, PadY):

          sizeY = image.shape[1]
          sizeX = image.shape[0]
          color = image.shape[2]  
          
          sizeXextend = sizeX
          sizeYextend = sizeY
         
 
          while sizeXextend%PadX!=0:
              sizeXextend = sizeXextend + 1
        
          while sizeYextend%PadY!=0:
              sizeYextend = sizeYextend + 1

          extendimage = np.zeros([sizeXextend, sizeYextend, color])
          
          extendimage[0:sizeX, 0:sizeY, 0:color] = image
              
              
          return extendimage      
    
def zero_pad_time(image, PadX, PadY):

          sizeY = image.shape[2]
          sizeX = image.shape[1]
          
          sizeXextend = sizeX
          sizeYextend = sizeY
         
 
          while sizeXextend%PadX!=0:
              sizeXextend = sizeXextend + 1
        
          while sizeYextend%PadY!=0:
              sizeYextend = sizeYextend + 1

          extendimage = np.zeros([image.shape[0], sizeXextend, sizeYextend])
          
          extendimage[:,0:sizeX, 0:sizeY] = image
              
              
          return extendimage   
      
def BackGroundCorrection2D(Image, sigma):
    
    
     Blur = gaussian(Image.astype(float), sigma)
     
     
     Corrected = Image - Blur
     
     return Corrected  
 
          

def MaxProjectDist(Image, axis = -1):
    
    MaxProject = np.amax(Image, axis = axis)
        
    return MaxProject

def MidProjectDist(Image, axis = -1, slices = 1):
    
    assert len(Image.shape) >=3
    SmallImage = Image.take(indices = range(Image.shape[axis]//2 - slices, Image.shape[axis]//2 + slices), axis = axis)
    
    MaxProject = np.amax(SmallImage, axis = axis)
    return MaxProject


def multiplot(imageA, imageB, imageC, titleA, titleB, titleC, targetdir = None, File = None, plotTitle = None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()
    ax[0].imshow(imageA, cmap=cm.gray)
    ax[0].set_title(titleA)
    ax[0].set_axis_off()
    ax[1].imshow(imageB, cmap=plt.cm.nipy_spectral)
    ax[1].set_title(titleB)
    ax[1].set_axis_off()
    ax[2].imshow(imageC, cmap=plt.cm.nipy_spectral)
    ax[2].set_title(titleC)
    ax[2].set_axis_off()
    plt.tight_layout()
    plt.show()
    for a in ax:
      a.set_axis_off()
      
def doubleplot(imageA, imageB, titleA, titleB, targetdir = None, File = None, plotTitle = None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()
    ax[0].imshow(imageA, cmap=cm.gray)
    ax[0].set_title(titleA)
    ax[0].set_axis_off()
    ax[1].imshow(imageB, cmap=plt.cm.nipy_spectral)
    ax[1].set_title(titleB)
    ax[1].set_axis_off()

    plt.tight_layout()
    plt.show()
    for a in ax:
      a.set_axis_off() 

def _check_dtype_supported(ar):
    # Should use `issubdtype` for bool below, but there's a bug in numpy 1.7
    if not (ar.dtype == bool or np.issubdtype(ar.dtype, np.integer)):
        raise TypeError("Only bool or integer image types are supported. "
                        "Got %s." % ar.dtype)


    

    
    


def normalizeFloatZeroOne(x, pmin = 3, pmax = 99.8, axis = None, eps = 1e-20, dtype = np.float32):
    """Percentile based Normalization
    
    Normalize patches of image before feeding into the network
    
    Parameters
    ----------
    x : np array Image patch
    pmin : minimum percentile value for normalization
    pmax : maximum percentile value for normalization
    axis : axis along which the normalization has to be carried out
    eps : avoid dividing by zero
    dtype: type of numpy array, float 32 default
    """
    mi = np.percentile(x, pmin, axis = axis, keepdims = True)
    ma = np.percentile(x, pmax, axis = axis, keepdims = True)
    return normalizer(x, mi, ma, eps = eps, dtype = dtype)

# https://docs.python.org/3/library/itertools.html#itertools-recipes
def move_image_axes(x, fr, to, adjust_singletons=False):
    """
    x: ndarray
    fr,to: axes string (see `axes_dict`)
    """
    fr = axes_check_and_normalize(fr, length=x.ndim)
    to = axes_check_and_normalize(to)

    fr_initial = fr
    x_shape_initial = x.shape
    adjust_singletons = bool(adjust_singletons)
    if adjust_singletons:
        # remove axes not present in 'to'
        slices = [slice(None) for _ in x.shape]
        for i,a in enumerate(fr):
            if (a not in to) and (x.shape[i]==1):
                # remove singleton axis
                slices[i] = 0
                fr = fr.replace(a,'')
        x = x[slices]
        # add dummy axes present in 'to'
        for i,a in enumerate(to):
            if (a not in fr):
                # add singleton axis
                x = np.expand_dims(x,-1)
                fr += a

    if set(fr) != set(to):
        _adjusted = '(adjusted to %s and %s) ' % (x.shape, fr) if adjust_singletons else ''
        raise ValueError(
            'image with shape %s and axes %s %snot compatible with target axes %s.'
            % (x_shape_initial, fr_initial, _adjusted, to)
        )

    ax_from, ax_to = axes_dict(fr), axes_dict(to)
    if fr == to:
        return x
    return np.moveaxis(x, [ax_from[a] for a in fr], [ax_to[a] for a in fr])
def consume(iterator):
    collections.deque(iterator, maxlen=0)

def _raise(e):
    raise e
def compose(*funcs):
    return lambda x: reduce(lambda f,g: g(f), funcs, x)

def normalizeZeroOne(x):

     x = x.astype('float32')

     minVal = np.min(x)
     maxVal = np.max(x)
     
     x = ((x-minVal) / (maxVal - minVal + 1.0e-20))
     
     return x
    
def normalizeZero255(x):

     x = x.astype('float32')

     minVal = np.min(x)
     maxVal = np.max(x)
     
     x = ((x-minVal) / (maxVal - minVal + 1.0e-20))
     
     return x * 255   
    
    
def normalizer(x, mi , ma, eps = 1e-20, dtype = np.float32):


    """
    Number expression evaluation for normalization
    
    Parameters
    ----------
    x : np array of Image patch
    mi : minimum input percentile value
    ma : maximum input percentile value
    eps: avoid dividing by zero
    dtype: type of numpy array, float 32 defaut
    """


    if dtype is not None:
        x = x.astype(dtype, copy = False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy = False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy = False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi ) / (ma - mi + eps)")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

        x = normalizeZeroOne(x)
    return x    

    
def LocalThreshold2D(Image, boxsize, offset = 0, size = 10):
    
    if boxsize%2 == 0:
        boxsize = boxsize + 1
    adaptive_thresh = threshold_local(Image, boxsize, offset=offset)
    Binary  = Image > adaptive_thresh
    #Clean =  remove_small_objects(Binary, min_size=size, connectivity=4, in_place=False)

    return Binary



   ##CARE csbdeep modification of implemented function
def normalizeFloat(x, pmin = 3, pmax = 99.8, axis = None, eps = 1e-20, dtype = np.float32):
    """Percentile based Normalization
    
    Normalize patches of image before feeding into the network
    
    Parameters
    ----------
    x : np array Image patch
    pmin : minimum percentile value for normalization
    pmax : maximum percentile value for normalization
    axis : axis along which the normalization has to be carried out
    eps : avoid dividing by zero
    dtype: type of numpy array, float 32 default
    """
    mi = np.percentile(x, pmin, axis = axis, keepdims = True)
    ma = np.percentile(x, pmax, axis = axis, keepdims = True)
    return normalize_mi_ma(x, mi, ma, eps = eps, dtype = dtype)


def normalize_mi_ma(x, mi , ma, eps = 1e-20, dtype = np.float32):
    
    
    """
    Number expression evaluation for normalization
    
    Parameters
    ----------
    x : np array of Image patch
    mi : minimum input percentile value
    ma : maximum input percentile value
    eps: avoid dividing by zero
    dtype: type of numpy array, float 32 defaut
    """
    
    
    if dtype is not None:
        x = x.astype(dtype, copy = False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy = False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy = False)
        eps = dtype(eps)
        
    try: 
        import numexpr
        x = numexpr.evaluate("(x - mi ) / (ma - mi + eps)")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

        
    return x    







def backend_channels_last():
    import keras.backend as K
    assert K.image_data_format() in ('channels_first','channels_last')
    return K.image_data_format() == 'channels_last'


def move_channel_for_backend(X,channel):
    if backend_channels_last():
        return np.moveaxis(X, channel, -1)
    else:
        return np.moveaxis(X, channel,  1)
        

def axes_check_and_normalize(axes,length=None,disallowed=None,return_allowed=False):
    """
    S(ample), T(ime), C(hannel), Z, Y, X
    """
    allowed = 'STCZYX'
    axes = str(axes).upper()
    consume(a in allowed or _raise(ValueError("invalid axis '%s', must be one of %s."%(a,list(allowed)))) for a in axes)
    disallowed is None or consume(a not in disallowed or _raise(ValueError("disallowed axis '%s'."%a)) for a in axes)
    consume(axes.count(a)==1 or _raise(ValueError("axis '%s' occurs more than once."%a)) for a in axes)
    length is None or len(axes)==length or _raise(ValueError('axes (%s) must be of length %d.' % (axes,length)))
    return (axes,allowed) if return_allowed else axes
def axes_dict(axes):
    """
    from axes string to dict
    """
    axes, allowed = axes_check_and_normalize(axes,return_allowed=True)
    return { a: None if axes.find(a) == -1 else axes.find(a) for a in allowed }
    # return collections.namedt     
    
