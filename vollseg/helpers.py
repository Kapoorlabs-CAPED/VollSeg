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
from skimage.morphology import remove_small_objects, remove_small_holes, thin
from stardist.models import StarDist3D,StarDist2D
from skimage.filters import gaussian
from six.moves import reduce
from matplotlib import cm
from scipy import spatial
from skimage.filters import threshold_local, threshold_otsu, threshold_mean
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
from tqdm import tqdm
from skimage.util import random_noise
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from stardist.models.base import StarDistBase
import math
import pandas as pd
import napari
import glob
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QPushButton, QSlider
import diplib as dip

Boxname = 'ImageIDBox'


class SegCorrect(object):

     def __init__(self, imagedir, segmentationdir):
         
         self.imagedir = imagedir
         self.segmentationdir = segmentationdir
         
     def showNapari(self):
                 
                 self.viewer = napari.Viewer()
                 Raw_path = os.path.join(self.imagedir, '*tif')
                 X = glob.glob(Raw_path)
                 Imageids = []
                 Seg_path = os.path.join(self.segmentationdir, '*tif')
                 Y = glob.glob(Seg_path)
                 SegImageids = []
                 for imagename in X:
                     Imageids.append(imagename)
                 for imagename in Y:
                     SegImageids.append(imagename)   
                     
                 imageidbox = QComboBox()   
                 imageidbox.addItem(Boxname)   
                 savebutton = QPushButton(' Save Corrections')
                    
                 for i in range(0, len(Imageids)):
                     
                     
                     imageidbox.addItem(str(Imageids[i]))
                     
                     
                 
                 imageidbox.currentIndexChanged.connect(
                 lambda trackid = imageidbox: self.image_add(
                         
                         imageidbox.currentText(),
                         self.segmentationdir + "/" + os.path.basename(os.path.splitext(imageidbox.currentText())[0]) + '.tif',
                         os.path.basename(os.path.splitext(imageidbox.currentText())[0]),
                         False
                    
                )
            )            
                 
                 savebutton.clicked.connect(
                 lambda trackid = imageidbox: self.image_add(
                         
                         imageidbox.currentText(),
                         self.segmentationdir + "/" + os.path.basename(os.path.splitext(imageidbox.currentText())[0]) + '.tif',
                         os.path.basename(os.path.splitext(imageidbox.currentText())[0]),
                         True
                    
                )
            )   
                    
                  
                 
                 self.viewer.window.add_dock_widget(imageidbox, name="Image", area='bottom') 
                 self.viewer.window.add_dock_widget(savebutton, name="Save Segmentations", area='bottom') 
                 
                 
                 
     def image_add(self, image_toread, seg_image_toread, imagename,  save = False):
                
                if not save:
                        for layer in list(self.viewer.layers):

                            if 'Image' in layer.name or layer.name in 'Image':

                                                            self.viewer.layers.remove(layer)


                        self.image = imread(image_toread)
                        self.segimage =  imread(seg_image_toread)
                        
                        self.viewer.add_image(self.image, name='Image'+imagename)
                        self.viewer.add_labels(self.segimage, name ='Image'+'Integer_Labels'+imagename)

                
               

                if save:


                        ModifiedArraySeg = self.viewer.layers['Image'+'Integer_Labels' + imagename].data 
                        ModifiedArraySeg = ModifiedArraySeg.astype('uint16')
                        imwrite((self.segmentationdir  +   imagename + '.tif' ) , ModifiedArraySeg)


class StarDistBaseLite(StarDist3D):
     def __init__(self, config, name=None, basedir='.'):
        super().__init__(config=config, name=name, basedir=basedir)


     def predict_prob(self, img, axes=None, normalizer=None, n_tiles=None, show_tile_progress=True, **predict_kwargs):
       

        x, axes, axes_net, axes_net_div_by, _permute_axes, resizer, n_tiles, grid, grid_dict, channel, predict_direct, tiling_setup = \
            self._predict_setup(img, axes, normalizer, n_tiles, show_tile_progress, predict_kwargs)

        if np.prod(n_tiles) > 1:
            tile_generator, output_shape, create_empty_output = tiling_setup()

            prob = create_empty_output(1)
           
            result = (prob,prob)

            for tile, s_src, s_dst in tile_generator:
                # predict_direct -> prob, dist, [prob_class if multi_class]
                result_tile = predict_direct(tile)
                # account for grid
                s_src = [slice(s.start//grid_dict.get(a,1),s.stop//grid_dict.get(a,1)) for s,a in zip(s_src,axes_net)]
                s_dst = [slice(s.start//grid_dict.get(a,1),s.stop//grid_dict.get(a,1)) for s,a in zip(s_dst,axes_net)]
                # prob and dist have different channel dimensionality than image x
                s_src[channel] = slice(None)
                s_dst[channel] = slice(None)
                s_src, s_dst = tuple(s_src), tuple(s_dst)
                # print(s_src,s_dst)
                for part, part_tile in zip(result, result_tile):
                    try:
                       part[s_dst] = part_tile[s_src]
                    except:
                        pass
                
        else:
            # predict_direct -> prob, dist, [prob_class if multi_class]
            result = predict_direct(x)

        result = [resizer.after(part, axes_net) for part in result]

        # prob
        result[0] = np.take(result[0],0,axis=channel)
        

        return result[0]


def BinaryLabel(BinaryImageOriginal, max_size = 15000):
    
    BinaryImageOriginal = BinaryImageOriginal.astype('uint16')
    image = normalizeFloatZeroOne(BinaryImageOriginal)
    image = invertimage(image)
    IntegerImage = watershed(-image)
    AugmentedLabel = remove_big_objects(IntegerImage, max_size = max_size) 

    return AugmentedLabel 


def expand_labels(label_image, distance=1):
    """Expand labels in label image by ``distance`` pixels without overlapping.
    Given a label image, ``expand_labels`` grows label regions (connected components)
    outwards by up to ``distance`` pixels without overflowing into neighboring regions.
    More specifically, each background pixel that is within Euclidean distance
    of <= ``distance`` pixels of a connected component is assigned the label of that
    connected component.
    Where multiple connected components are within ``distance`` pixels of a background
    pixel, the label value of the closest connected component will be assigned (see
    Notes for the case of multiple labels at equal distance).
    Parameters
    ----------
    label_image : ndarray of dtype int
        label image
    distance : float
        Euclidean distance in pixels by which to grow the labels. Default is one.
    Returns
    -------
    enlarged_labels : ndarray of dtype int
        Labeled array, where all connected regions have been enlarged
    Notes
    -----
    Where labels are spaced more than ``distance`` pixels are apart, this is
    equivalent to a morphological dilation with a disc or hyperball of radius ``distance``.
    However, in contrast to a morphological dilation, ``expand_labels`` will
    not expand a label region into a neighboring region.  
    This implementation of ``expand_labels`` is derived from CellProfiler [1]_, where
    it is known as module "IdentifySecondaryObjects (Distance-N)" [2]_.
    There is an important edge case when a pixel has the same distance to
    multiple regions, as it is not defined which region expands into that
    space. Here, the exact behavior depends on the upstream implementation
    of ``scipy.ndimage.distance_transform_edt``.
    See Also
    --------
    :func:`skimage.measure.label`, :func:`skimage.segmentation.watershed`, :func:`skimage.morphology.dilation`
    References
    ----------
    .. [1] https://cellprofiler.org
    .. [2] https://github.com/CellProfiler/CellProfiler/blob/082930ea95add7b72243a4fa3d39ae5145995e9c/cellprofiler/modules/identifysecondaryobjects.py#L559
    Examples
    --------
    >>> labels = np.array([0, 1, 0, 0, 0, 0, 2])
    >>> expand_labels(labels, distance=1)
    array([1, 1, 1, 0, 0, 2, 2])
    Labels will not overwrite each other:
    >>> expand_labels(labels, distance=3)
    array([1, 1, 1, 1, 2, 2, 2])
    In case of ties, behavior is undefined, but currently resolves to the
    label closest to ``(0,) * ndim`` in lexicographical order.
    >>> labels_tied = np.array([0, 1, 0, 2, 0])
    >>> expand_labels(labels_tied, 1)
    array([1, 1, 1, 2, 2])
    >>> labels2d = np.array(
    ...     [[0, 1, 0, 0],
    ...      [2, 0, 0, 0],
    ...      [0, 3, 0, 0]]
    ... )
    >>> expand_labels(labels2d, 1)
    array([[2, 1, 1, 0],
           [2, 2, 0, 0],
           [2, 3, 3, 0]])
    """

    distances, nearest_label_coords = distance_transform_edt(
        label_image == 0, return_indices=True
    )
    labels_out = np.zeros_like(label_image)
    dilate_mask = distances <= distance
    # build the coordinates to find nearest labels,
    # in contrast to [1] this implementation supports label arrays
    # of any dimension
    masked_nearest_label_coords = [
        dimension_indices[dilate_mask]
        for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels
    return labels_out

def SimplePrediction(x, UnetModel, StarModel, n_tiles = (2,2), UseProbability = True, min_size = 20, axis = 'ZYX', globalthreshold = 1.0E-5):
    
               
    
             
                                           
                      Mask = UNETPrediction3D(x, UnetModel, n_tiles, axis)
                      
                      SmartSeeds, _, StarImage, _ = STARPrediction3D(x, StarModel, n_tiles, MaskImage = Mask, smartcorrection = None, UseProbability = UseProbability, globalthreshold = globalthreshold)
                      
                      SmartSeeds = SmartSeeds.astype('uint16') 
                     
                
                
                      return SmartSeeds

def crappify_flou_G_P(x, y, lam, savedirx, savediry, name):
    x = x.astype('float32')
    gaussiannoise = np.random.poisson(lam, x.shape)
    x = x + gaussiannoise 
        
    #add noise to original image
    imwrite(savedirx + '/' + name + 'pg' + str(lam) + '.tif', x.astype('float32'))    
    #keep the label the same
    imwrite(savediry + '/' + name + 'pg' + str(lam)  + '.tif', y.astype('uint16'))     


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





def NotumQueen(save_dir, fname, denoising_model, projection_model, mask_model, star_model, min_size = 5, n_tiles = (1,1,1), UseProbability = True):
    
    Path(save_dir).mkdir(exist_ok = True)
    projection_results = save_dir + 'Projected/'
    Path(projection_results).mkdir(exist_ok = True)
    Name = os.path.basename(os.path.splitext(fname)[0])
    print('Denoising Image')
    image = imread(fname)
    image = denoising_model.predict(image,'ZYX', n_tiles = n_tiles)
    image = projection_model.predict(image,'YX', n_tiles = (n_tiles[1], n_tiles[2]))

    imwrite((projection_results + Name+ '.tif' ) , image.astype('float32'))
    
    NotumSegmentation2D(save_dir,image,fname, mask_model, star_model, min_size = min_size, n_tiles = (n_tiles[1], n_tiles[2]), UseProbability = UseProbability)
    
    
def NotumSeededQueen(save_dir, fname, denoising_model, projection_model, unet_model, mask_model, star_model, min_size = 5, n_tiles = (1,1,1), UseProbability = True):
    
    Path(save_dir).mkdir(exist_ok = True)
    projection_results = save_dir + 'Projected/'
    Path(projection_results).mkdir(exist_ok = True)
    Name = os.path.basename(os.path.splitext(fname)[0])
    print('Denoising Image')
    image = imread(fname)
    image = denoising_model.predict(image,'ZYX', n_tiles = n_tiles)
    image = projection_model.predict(image,'YX', n_tiles = (n_tiles[1], n_tiles[2]))

    imwrite((projection_results + Name+ '.tif' ) , image.astype('float32'))
    
    SeededNotumSegmentation2D(save_dir,image,fname, unet_model, mask_model, star_model, min_size = min_size, n_tiles = (n_tiles[1], n_tiles[2]), UseProbability = UseProbability)

def SeededNotumSegmentation2D(SaveDir,image, fname, UnetModel, MaskModel, StarModel, min_size = 5, n_tiles = (2,2), UseProbability = True):
    
    print('Generating SmartSeed results')
    
    MASKResults = SaveDir + 'OverAllMask/'
    UNETResults = SaveDir + 'UnetMask/'
    StarImageResults = SaveDir + 'StarDistMask/'
    SmartSeedsResults = SaveDir + 'SmartSeedsMask/' 
    SmartSeedsLabelResults = SaveDir + 'SmartSeedsLabels/' 
    ProbResults = SaveDir + 'Probability/' 
    
    Path(SaveDir).mkdir(exist_ok = True)
    Path(SmartSeedsResults).mkdir(exist_ok = True)
    Path(StarImageResults).mkdir(exist_ok = True)
    Path(UNETResults).mkdir(exist_ok = True)
    Path(MASKResults).mkdir(exist_ok = True)
    Path(SmartSeedsLabelResults).mkdir(exist_ok = True)
    Path(ProbResults).mkdir(exist_ok = True)
    #Read Image
    Name = os.path.basename(os.path.splitext(fname)[0])
    
    #U-net prediction
    
    OverAllMask = SuperUNETPrediction(image, MaskModel, n_tiles, 'YX')
    Mask = SuperUNETPrediction(image, UnetModel, n_tiles, 'YX')
    
    #Smart Seed prediction 
    SmartSeeds, Markers, StarImage, ProbImage = SuperSTARPrediction(image, StarModel, n_tiles, MaskImage = Mask, OverAllMaskImage = OverAllMask, UseProbability = UseProbability)
    
    
    SmartSeedsLabels = SmartSeeds.copy()
    
    OverAllMask = CleanMask(StarImage, OverAllMask)    #For avoiding pixel level error 
    SmartSeedsLabels = np.multiply(SmartSeedsLabels, OverAllMask)
    SegimageB = find_boundaries(SmartSeedsLabels)
    invertProbimage = 1 - ProbImage
    image_max = np.add(invertProbimage,SegimageB)
    indices = np.where(image_max < 1.2)
    image_max[indices] = 0
    SmartSeeds = np.array(dip.UpperSkeleton2D(image_max.astype('float32')))
           
    #Save results, we only need smart seeds finale results but hey!
    imwrite((ProbResults + Name+ '.tif' ) , ProbImage.astype('float32'))
    imwrite((SmartSeedsResults + Name+ '.tif' ) , SmartSeeds.astype('uint8'))
    imwrite((SmartSeedsLabelResults + Name+ '.tif' ) , SmartSeedsLabels.astype('uint16'))
    imwrite((StarImageResults + Name+ '.tif' ) , StarImage.astype('uint16'))
    imwrite((UNETResults + Name+ '.tif' ) , Mask.astype('uint8'))  
    imwrite((MASKResults + Name+ '.tif' ) , OverAllMask.astype('uint8'))  
 
      



def CreateTrackMate_CSV( Label,Name,savedir):


    
    


    TimeList = []

    XList = []
    YList = []
    TrackIDList = []
    QualityList = []
    print('Image has shape:', Label.shape)
    print('Image Dimensions:', len(Label.shape))

    CurrentSegimage = Label.astype('uint16')
    properties = measure.regionprops(CurrentSegimage)
    for prop in properties:

            T = prop.centroid[0]
            Y = prop.centroid[1]
            X = prop.centroid[2]
            regionlabel = prop.label
            sizeZ = abs(prop.bbox[0] - prop.bbox[3])
            sizeY = abs(prop.bbox[1] - prop.bbox[4])
            sizeX = abs(prop.bbox[2] - prop.bbox[5])
            volume = sizeZ * sizeX * sizeY
            radius = math.pow(3 * volume / (4 * math.pi), 1.0 / 3.0)
            perimeter = 2 * math.pi * radius
            TimeList.append(int(T))
            XList.append(int(X))
            YList.append(int(Y))
            TrackIDList.append(regionlabel)
            QualityList.append(radius)

    df = pd.DataFrame(
        list(
            zip(
                XList,
                YList,
                TimeList,
                TrackIDList,
                QualityList
            )
        ),
        index=None,
        columns=[
            'POSITION_X',
            'POSITION_Y',
            'FRAME',
            'TRACK_ID',
            'QUALITY'
        ],
    )

    df.to_csv(savedir + '/' + 'TrackMate_csv' + Name + '.csv', index=False)    
 
    
 

    
def NotumKing(save_dir, filesRaw, denoising_model, projection_model, mask_model, star_model, n_tiles = (1,1,1), UseProbability = True, dounet = True, seedpool = True, min_size_mask = 100, min_size = 5, max_size = 10000000):
    
    Path(save_dir).mkdir(exist_ok = True)
    projection_results = save_dir + 'Projected/'
    Path(projection_results).mkdir(exist_ok = True)
    
    print('Denoising Image')
    time_lapse = []
    for fname in filesRaw:
            image = imread(fname)
            image = denoising_model.predict(image,'ZYX', n_tiles = n_tiles)
            image = projection_model.predict(image,'YX', n_tiles = (n_tiles[1], n_tiles[2]))
            time_lapse.append(image)
    Name = os.path.basename(os.path.splitext(fname)[0])
    time_lapse = np.asarray(time_lapse)        
    imwrite((projection_results + Name+ '.tif' ) , time_lapse.astype('float32'))   
    
    Raw_path = os.path.join(projection_results, '*tif')
    filesRaw = glob.glob(Raw_path)
    for fname in filesRaw:
        
         SmartSeedPrediction3D( save_dir, fname,  mask_model, star_model,  min_size_mask = min_size_mask, min_size = min_size, max_size = max_size, n_tiles = n_tiles, UseProbability = UseProbability, dounet = dounet, seedpool = seedpool)    

def NotumSegmentation2D(save_dir,image,fname, mask_model, star_model, min_size = 5, n_tiles = (2,2), UseProbability = True):
    
    print('Generating SmartSeed results')
    Path(save_dir).mkdir(exist_ok = True)
    MASKResults = save_dir + 'OverAllMask/'
    StarImageResults = save_dir + 'StarDistMask/'
    SmartSeedsResults = save_dir + 'SmartSeedsMask/' 
    SmartSeedsLabelResults = save_dir + 'SmartSeedsLabels/' 
    ProbResults = save_dir + 'Probability/' 
    
    Path(SmartSeedsResults).mkdir(exist_ok = True)
    Path(StarImageResults).mkdir(exist_ok = True)
   
    Path(MASKResults).mkdir(exist_ok = True)
    Path(SmartSeedsLabelResults).mkdir(exist_ok = True)
    Path(ProbResults).mkdir(exist_ok = True)
    #Read Image
    Name = os.path.basename(os.path.splitext(fname)[0])
    
    #U-net prediction
    
    OverAllMask = SuperUNETPrediction(image, mask_model, n_tiles, 'YX')
    
    
    #Smart Seed prediction 
    SmartSeeds, Markers, StarImage, ProbImage = SuperSTARPrediction(image, star_model, n_tiles, MaskImage = OverAllMask, OverAllMaskImage = OverAllMask, UseProbability = UseProbability)
    
    
    SmartSeedsLabels = SmartSeeds.copy()
    
    OverAllMask = CleanMask(StarImage, OverAllMask)    #For avoiding pixel level error 
    SmartSeedsLabels = np.multiply(SmartSeedsLabels, OverAllMask)
    SegimageB = find_boundaries(SmartSeedsLabels)
    invertProbimage = 1 - ProbImage
    image_max = np.add(invertProbimage,SegimageB)
    indices = np.where(image_max < 1.2)
    image_max[indices] = 0
    SmartSeeds = np.array(dip.UpperSkeleton2D(image_max.astype('float32')))
           
    #Save results, we only need smart seeds finale results but hey!
    imwrite((ProbResults + Name+ '.tif' ) , ProbImage.astype('float32'))
    imwrite((SmartSeedsResults + Name+ '.tif' ) , SmartSeeds.astype('uint8'))
    imwrite((SmartSeedsLabelResults + Name+ '.tif' ) , SmartSeedsLabels.astype('uint16'))
    imwrite((StarImageResults + Name+ '.tif' ) , StarImage.astype('uint16'))
     
    imwrite((MASKResults + Name+ '.tif' ) , OverAllMask.astype('uint8')) 
    
    

def SmartSeedPrediction2D( SaveDir, fname, UnetModel, StarModel, min_size = 5, n_tiles = (2,2), UseProbability = True):
    
    print('Generating SmartSeed results')
    UNETResults = SaveDir + 'BinaryMask/'
    StarImageResults = SaveDir + 'StarDist/'
    SmartSeedsResults = SaveDir + 'SmartSeedsMask/' 
    SmartSeedsIntegerResults = SaveDir + 'SmartSeedsInteger/'
    Path(SaveDir).mkdir(exist_ok = True)
    Path(SmartSeedsResults).mkdir(exist_ok = True)
    Path(StarImageResults).mkdir(exist_ok = True)
    Path(UNETResults).mkdir(exist_ok = True)
    Path(SmartSeedsIntegerResults).mkdir(exist_ok = True)
    #Read Image
    image = imread(fname)
    Name = os.path.basename(os.path.splitext(fname)[0])
    #U-net prediction
    Mask = SuperUNETPrediction(image, UnetModel, n_tiles, 'YX')
  
    #Smart Seed prediction 
    SmartSeeds, Markers, StarImage, ProbImage = SuperSTARPrediction(image, StarModel, n_tiles, MaskImage = Mask, UseProbability = UseProbability)
    labelmax = np.amax(StarImage)
    Mask[StarImage > 0] == labelmax + 1
    #For avoiding pixel level error 
    Mask = expand_labels(Mask, distance = 1)
    SmartSeeds = expand_labels(SmartSeeds, distance = 1)
    
    SmartSeedsInteger = SmartSeeds
    
    BinaryMask = Integer_to_border(Mask.astype('uint16'))  
    SmartSeeds = Integer_to_border(SmartSeeds.astype('uint16'))
    #Missing edges of one network prevented by others
    SmartSeeds = skeletonize(SmartSeeds)
    BinaryMask = skeletonize(BinaryMask)
    SmartSeeds = np.logical_or(SmartSeeds, BinaryMask)
    SmartSeeds = skeletonize(SmartSeeds)
    
    #Could create double pixels and new pockets, use watershed and skeletonize to remove again
    SmartSeeds = BinaryLabel(SmartSeeds)
   
    SmartSeeds = Integer_to_border(SmartSeeds.astype('uint16'))
    SmartSeeds = remove_small_holes(SmartSeeds, min_size)
    SmartSeeds = skeletonize(SmartSeeds)
    #Save results, we only need smart seeds finale results but hey!
    imwrite((SmartSeedsResults + Name+ '.tif' ) , SmartSeeds.astype('uint8'))
    imwrite((SmartSeedsIntegerResults + Name+ '.tif' ) , SmartSeedsInteger.astype('uint16'))
    imwrite((StarImageResults + Name+ '.tif' ) , StarImage.astype('uint16'))
    imwrite((UNETResults + Name+ '.tif' ) , BinaryMask.astype('uint8'))   
    
 
    return SmartSeeds, Mask
  
def SuperWatershedwithMask(Image, Label,mask, grid):
    
    
   
    properties = measure.regionprops(Label, Image)
    binaryproperties = measure.regionprops(label(mask), Image) 
    
    Coordinates = [prop.centroid for prop in properties]
    BinaryCoordinates = [prop.centroid for prop in binaryproperties]
    Binarybbox = [prop.bbox for prop in binaryproperties]
    
    if len(Binarybbox) > 0:    
            for i in range(0, len(Binarybbox)):
                
                box = Binarybbox[i]
                inside = [iouNotum(box, star) for star in Coordinates]
                
                if not any(inside) :
                         Coordinates.append(BinaryCoordinates[i])
    Coordinates = sorted(Coordinates , key=lambda k: [k[1], k[0]])
    Coordinates.append((0,0))
    Coordinates = np.asarray(Coordinates)

    coordinates_int = np.round(Coordinates).astype(int)
    markers_raw = np.zeros_like(Image)  
    markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))
    
    markers = morphology.dilation(markers_raw, morphology.disk(2))
    watershedImage = watershed(-Image, markers, mask = mask.copy())
    
    return watershedImage, markers
#If there are neighbouring seeds we do not put more seeds
def ConditioncheckNotum(centroid, boxA, p, ndim):
    
      condition = False
    
      if centroid[p] >=  boxA[p]  and centroid[p] <=  boxA[p + ndim]:
          
           condition = True
           
      return condition     
 
def iouNotum(boxA, centroid):
    
    ndim = len(centroid)
    inside = False
    
    Condition = [ConditioncheckNotum(centroid, boxA, p, ndim) for p in range(0,ndim)]
        
    inside = all(Condition)
    
    return inside


def SuperWatershedwithoutMask(Image, Label,mask, grid):
    
    
   
    properties = measure.regionprops(Label, Image)
    binaryproperties = measure.regionprops(label(mask), Image) 
    
    Coordinates = [prop.centroid for prop in properties]
    BinaryCoordinates = [prop.centroid for prop in binaryproperties]
    Binarybbox = [prop.bbox for prop in binaryproperties]
    
    if len(Binarybbox) > 0:    
            for i in range(0, len(Binarybbox)):
                
                box = Binarybbox[i]
                inside = [iouNotum(box, star) for star in Coordinates]
                
                if not any(inside) :
                         Coordinates.append(BinaryCoordinates[i])
    Coordinates = sorted(Coordinates , key=lambda k: [k[1], k[0]])
    Coordinates.append((0,0))
    Coordinates = np.asarray(Coordinates)

    coordinates_int = np.round(Coordinates).astype(int)
    markers_raw = np.zeros_like(Image)  
    markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))
    
    markers = morphology.dilation(markers_raw, morphology.disk(2))
    watershedImage = watershed(-Image, markers)
    
    return watershedImage, markers

#Default method that works well with cells which are below a certain shape and do not have weak edges    
    
def SmartSeedPredictionSliced(SaveDir, fname, UnetModel, StarModel, NoiseModel = None, min_size = 5, n_tiles = (1,1), UseProbability = True, threshold = 20):
    
    print('Generating SmartSeed results')
    UNETResults = SaveDir + 'BinaryMask/'
    StarImageResults = SaveDir + 'StarDist/'
    SmartSeedsResults = SaveDir + 'SmartSeedsMask/' 
    SmartSeedsIntegerResults = SaveDir + 'SmartSeedsInteger/'
    DenoiseResults = SaveDir + 'Denoised/'
    
    Path(SaveDir).mkdir(exist_ok = True)
    Path(DenoiseResults).mkdir(exist_ok = True)
    Path(SmartSeedsResults).mkdir(exist_ok = True)
    Path(SmartSeedsIntegerResults).mkdir(exist_ok = True)
    Path(StarImageResults).mkdir(exist_ok = True)
    Path(UNETResults).mkdir(exist_ok = True)
    
    #Read Image
    image = imread(fname)
    Name = os.path.basename(os.path.splitext(fname)[0])
    
    if NoiseModel is not None:
         image = NoiseModel.predict(image, axes='ZYX', n_tiles=(1,n_tiles[0], n_tiles[1]))
         imwrite((DenoiseResults + Name+ '.tif' ) , image.astype('float32'))
    BinaryTime = np.zeros([image.shape[0], image.shape[1], image.shape[2]])
    StarTime = np.zeros([image.shape[0], image.shape[1], image.shape[2]])
    SmartSeedsTime = np.zeros([image.shape[0], image.shape[1], image.shape[2]])
    SmartSeedsIntegerTime = np.zeros([image.shape[0], image.shape[1], image.shape[2]])
    for i in range(0, image.shape[0]):
        #U-net prediction
        Mask = SuperUNETPrediction(image[i,:], UnetModel, n_tiles, 'YX')
      
        #Smart Seed prediction 
        SmartSeeds, Markers, StarImage, ProbImage = SuperSTARPrediction(image[i,:], StarModel, n_tiles, MaskImage = Mask, UseProbability = UseProbability)
        
        labelmax = np.amax(StarImage)
        Mask[StarImage > 0] == labelmax + 1
        #For avoiding pixel level error 
        Mask = expand_labels(Mask, distance = 1)
        SmartSeeds = expand_labels(SmartSeeds, distance = 1)
        SmartSeedsInteger = SmartSeeds
        BinaryMask = Integer_to_border(Mask.astype('uint16'))  
        SmartSeeds = Integer_to_border(SmartSeeds.astype('uint16'))

        #Missing edges of one network prevented by others
        SmartSeeds = skeletonize(SmartSeeds)
        BinaryMask = skeletonize(BinaryMask)
        SmartSeeds = np.logical_or(SmartSeeds, BinaryMask)
        SmartSeeds = skeletonize(SmartSeeds)
    
        #Could create double pixels and new pockets, use watershed and skeletonize to remove again
        SmartSeeds = BinaryLabel(SmartSeeds)
      
        SmartSeeds = Integer_to_border(SmartSeeds.astype('uint16'))
        SmartSeeds = remove_small_holes(SmartSeeds, min_size)
        SmartSeeds = skeletonize(SmartSeeds)
        BinaryTime[i,:] = BinaryMask
        StarTime[i,:] = StarImage
        SmartSeedsTime[i,:] = SmartSeeds
        SmartSeedsIntegerTime[i,:] = SmartSeedsInteger
        
    #Save results, we only need smart seeds finale results but hey!
    SmartSeedsIntegerTime = merge_labels_across_volume(SmartSeedsIntegerTime.astype('uint16'), RelabelZ, threshold= threshold)
    imwrite((SmartSeedsResults + Name+ '.tif' ) , SmartSeedsTime.astype('uint8'))
    imwrite((SmartSeedsIntegerResults + Name+ '.tif' ) , SmartSeedsIntegerTime.astype('uint16'))
    imwrite((StarImageResults + Name+ '.tif' ) , StarTime.astype('uint16'))
    imwrite((UNETResults + Name+ '.tif' ) , BinaryTime.astype('uint8'))   
    
    
    

    
    
def SmartSeedPrediction3D( SaveDir, fname,  UnetModel, StarModel, NoiseModel = None, min_size_mask = 100, min_size = 100, max_size = 100000,
n_tiles = (1,2,2), UseProbability = True,  globalthreshold = 1.0E-5, extent = 0, dounet = True, seedpool = True, otsu = False, startZ = 0):
    
    
    
    print('Generating SmartSeed results')
    UNETResults = SaveDir + 'BinaryMask/'
    
    SmartSeedsResults = SaveDir + 'SmartSeedsMask/' 
    StarDistResults = SaveDir + 'StarDist/'
    DenoiseResults = SaveDir + 'Denoised/'
    ProbabilityResults = SaveDir + 'Probability/'
    MarkerResults = SaveDir + 'Markers/'
    Path(SaveDir).mkdir(exist_ok = True)
    Path(DenoiseResults).mkdir(exist_ok = True)
    Path(SmartSeedsResults).mkdir(exist_ok = True)
    Path(StarDistResults).mkdir(exist_ok = True)
    Path(UNETResults).mkdir(exist_ok = True)
    Path(ProbabilityResults).mkdir(exist_ok = True)
    Path(MarkerResults).mkdir(exist_ok = True)
    #Read Image
    image = imread(fname)
    
    sizeZ = image.shape[0]
    sizeY = image.shape[1]
    sizeX = image.shape[2]
    
    SizedMask = np.zeros([sizeZ, sizeY, sizeX], dtype = 'uint16')
    SizedSmartSeeds = np.zeros([sizeZ, sizeY, sizeX], dtype = 'uint16')
    SizedProbabilityMap = np.zeros([sizeZ, sizeY, sizeX], dtype = 'float32')
    Name = os.path.basename(os.path.splitext(fname)[0])
    if NoiseModel is not None:
         print('Denoising Image')
        
         image = NoiseModel.predict(image, axes='ZYX', n_tiles=n_tiles)
         newimage = np.zeros(image.shape, dtype = 'float32')
         if otsu:
             for i in range(0, image.shape[0]): 
                 thresh = threshold_otsu(image[i,:])
                 image[i,:][image[i,:] <= thresh] = 0
                 newimage[i,:] = image[i,:]
             image = newimage  
         imwrite((DenoiseResults + Name+ '.tif' ) , image.astype('float32'))   
    
    if dounet:
        print('UNET segmentation on Image')     

        Mask = UNETPrediction3D(image, UnetModel, n_tiles, 'ZYX')
    else:
      
      Mask = np.zeros(image.shape)
      
      for i in range(0, Mask.shape[0]):

                 thresh = threshold_otsu(image[i,:])
                 Mask[i,:] = image[i,:] > thresh  
                 Mask[i,:] = label(Mask[i,:]) 
    
                 Mask[i,:] = remove_small_objects(Mask[i,:].astype('uint16'), min_size = min_size)
                 Mask[i,:] = remove_big_objects(Mask[i,:].astype('uint16'), max_size = max_size)
    
    Mask = label(Mask > 0)       
    SizedMask[:, :Mask.shape[1], :Mask.shape[2]] = Mask
    imwrite((UNETResults + Name+ '.tif' ) , SizedMask.astype('uint16')) 
    print('Stardist segmentation on Image')  
    SmartSeeds, ProbabilityMap, StarImage, Markers = STARPrediction3D(image, StarModel,  n_tiles, MaskImage = Mask, UseProbability = UseProbability, globalthreshold = globalthreshold, extent = extent, seedpool = seedpool)
   
    for i in range(0, SmartSeeds.shape[0]):
       SmartSeeds[i,:] = remove_small_objects(SmartSeeds[i,:].astype('uint16'), min_size = min_size)
       SmartSeeds[i,:] = remove_big_objects(SmartSeeds[i,:].astype('uint16'), max_size = max_size)
    SmartSeeds = fill_label_holes(SmartSeeds.astype('uint16'))
    if startZ > 0:
         SmartSeeds[0:startZ,:,:] = 0
         
         
    SmartSeeds = RemoveLabels(SmartSeeds) 
    SizedSmartSeeds[:, :SmartSeeds.shape[1], :SmartSeeds.shape[2]] = SmartSeeds
    SizedProbabilityMap[:, :ProbabilityMap.shape[1], :ProbabilityMap.shape[2]] = ProbabilityMap           
    
    imwrite((StarDistResults + Name+ '.tif' ) , StarImage.astype('uint16'))
    imwrite((SmartSeedsResults + Name+ '.tif' ) , SizedSmartSeeds.astype('uint16'))
    imwrite((ProbabilityResults + Name+ '.tif' ) , ProbabilityMap.astype('float32'))
    imwrite((MarkerResults + Name+ '.tif' ) , Markers.astype('uint16'))
        
def VollSeg2D(image, unet_model, star_model, noise_model = None, prob_thresh = None, nms_thresh = None, axes = 'YX', min_size = 5,max_size = 10000000, dounet = True, n_tiles = (2,2), UseProbability = True):
    
    print('Generating SmartSeed results')
    
    if noise_model is not None:
         print('Denoising Image')
        
         image = noise_model.predict(image, axes=axes, n_tiles=n_tiles)
    if dounet:
        print('UNET segmentation on Image')     

        Mask = UNETPrediction3D(image, unet_model, n_tiles, )
    else:
      
      Mask = np.zeros(image.shape)
      
      thresh = threshold_otsu(image)
      Mask = image > thresh  
      Mask = label(Mask) 
      Mask = remove_small_objects(Mask.astype('uint16'), min_size = min_size)
      Mask = remove_big_objects(Mask.astype('uint16'), max_size = max_size)
    
      Mask = label(Mask > 0)       
  
    #Smart Seed prediction 
    SmartSeeds, Markers, StarImage, ProbabilityMap = SuperSTARPrediction(image, star_model, n_tiles, MaskImage = Mask, UseProbability = UseProbability, prob_thresh = prob_thresh, nms_thresh = nms_thresh)
    labelmax = np.amax(StarImage)
    Mask[StarImage > 0] == labelmax + 1
    #For avoiding pixel level error 
    Mask = expand_labels(Mask, distance = 1)
    SmartSeeds = expand_labels(SmartSeeds, distance = 1)
    
        
    BinaryMask = Integer_to_border(Mask.astype('uint16'))  
    SmartSeeds = Integer_to_border(SmartSeeds.astype('uint16'))
    #Missing edges of one network prevented by others
    SmartSeeds = skeletonize(SmartSeeds)
    BinaryMask = skeletonize(BinaryMask)
    SmartSeeds = np.logical_or(SmartSeeds, BinaryMask)
    SmartSeeds = skeletonize(SmartSeeds)
    
    #Could create double pixels and new pockets, use watershed and skeletonize to remove again
    SmartSeeds = BinaryLabel(SmartSeeds)
   
    SmartSeeds = Integer_to_border(SmartSeeds.astype('uint16'))
    SmartSeeds = remove_small_holes(SmartSeeds, min_size)
    SmartSeeds = skeletonize(SmartSeeds)
    if noise_model == None:
        return SmartSeeds, Mask, StarImage, ProbabilityMap, Markers 
    else:
        return SmartSeeds, Mask, StarImage, ProbabilityMap, Markers, image

    
def VollSeg3D( image,  unet_model, star_model, axes='ZYX', noise_model = None, prob_thresh = None, nms_thresh = None, min_size_mask = 100, min_size = 100, max_size = 10000000,
n_tiles = (1,2,2), UseProbability = True, globalthreshold = 1.0E-5, extent = 0, dounet = True, seedpool = True,  startZ = 0):
    
    print('Generating VollSeg results')
    sizeZ = image.shape[0]
    sizeY = image.shape[1]
    sizeX = image.shape[2]
    
    SizedMask = np.zeros([sizeZ, sizeY, sizeX], dtype = 'uint16')
    SizedSmartSeeds = np.zeros([sizeZ, sizeY, sizeX], dtype = 'uint16')
    SizedProbabilityMap = np.zeros([sizeZ, sizeY, sizeX], dtype = 'float32')
   
    if noise_model is not None:
         print('Denoising Image')
        
         image = noise_model.predict(image, axes=axes, n_tiles=n_tiles)
           
    
    if dounet:
        print('UNET segmentation on Image')     

        Mask = UNETPrediction3D(image, unet_model, n_tiles, )
    else:
      
      Mask = np.zeros(image.shape)
      
      for i in range(0, Mask.shape[0]):

                 thresh = threshold_otsu(image[i,:])
                 Mask[i,:] = image[i,:] > thresh  
                 Mask[i,:] = label(Mask[i,:]) 
    
                 Mask[i,:] = remove_small_objects(Mask[i,:].astype('uint16'), min_size = min_size)
                 Mask[i,:] = remove_big_objects(Mask[i,:].astype('uint16'), max_size = max_size)
    
    Mask = label(Mask > 0)       
    SizedMask[:, :Mask.shape[1], :Mask.shape[2]] = Mask
    print('Stardist segmentation on Image')  
    SmartSeeds, ProbabilityMap, StarImage, Markers = STARPrediction3D(image, star_model,  n_tiles, MaskImage = Mask, UseProbability = UseProbability, globalthreshold = globalthreshold, extent = extent, seedpool = seedpool, prob_thresh= prob_thresh, nms_thresh = nms_thresh)
   
    for i in range(0, SmartSeeds.shape[0]):
       SmartSeeds[i,:] = remove_small_objects(SmartSeeds[i,:].astype('uint16'), min_size = min_size)
       SmartSeeds[i,:] = remove_big_objects(SmartSeeds[i,:].astype('uint16'), max_size = max_size)
    SmartSeeds = fill_label_holes(SmartSeeds.astype('uint16'))
    if startZ > 0:
         SmartSeeds[0:startZ,:,:] = 0
         
         
    SmartSeeds = RemoveLabels(SmartSeeds) 
    SizedSmartSeeds[:, :SmartSeeds.shape[1], :SmartSeeds.shape[2]] = SmartSeeds
    SizedProbabilityMap[:, :ProbabilityMap.shape[1], :ProbabilityMap.shape[2]] = ProbabilityMap           
    
        
    if noise_model == None:
        return SizedSmartSeeds, SizedMask, StarImage, ProbabilityMap, Markers 
    else:
        return SizedSmartSeeds, SizedMask, StarImage, ProbabilityMap, Markers, image 



def Integer_to_border(Label):

        BoundaryLabel =  find_boundaries(Label, mode='outer')
           
        Binary = BoundaryLabel > 0
        
        return Binary
        

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
                


def SuperUNETPrediction(image, model, n_tiles, axis, threshold = 20):
    
    
    Segmented = model.predict(image, axis, n_tiles = n_tiles)
    try:
       thresh = threshold_otsu(Segmented)
       Binary = Segmented > thresh
    except:
        Binary = Segmented > 0
        
    Binary= binary_erosion(Binary)    
    #Postprocessing steps
    
    
    Finalimage = label(Binary)
    Finalimage = fill_label_holes(Finalimage)
    Finalimage = relabel_sequential(Finalimage)[0]
    
    return  Finalimage
def merge_labels_across_volume(labelvol, relabelfunc, threshold=3):
    nz, ny, nx = labelvol.shape
    res = np.zeros_like(labelvol)
    res[0,...] = labelvol[0,...]
    backup = labelvol.copy() # kapoors code modifies the input array
    for i in tqdm(range(nz-1)):
        
        res[i+1,...] = relabelfunc(res[i,...], labelvol[i+1,...],threshold=threshold)
        labelvol = backup.copy() # restore the input array
    return res

def RelabelZ(previousImage, currentImage,threshold):
      # This line ensures non-intersecting label sets
      copyImage = currentImage.copy()
      copypreviousImage = previousImage.copy()
      copyImage = relabel_sequential(copyImage,offset=copypreviousImage.max()+1)[0]
        # I also don't like modifying the input image, so we take a copy
      relabelimage = copyImage.copy()
      waterproperties = measure.regionprops(copypreviousImage, copypreviousImage)
      indices = [] 
      labels = []
      for prop in waterproperties:
        if prop.label > 0:
                 
                  labels.append(prop.label)
                  indices.append(prop.centroid) 
     
      if len(indices) > 0:
        tree = spatial.cKDTree(indices)
        currentwaterproperties = measure.regionprops(copyImage, copyImage)
        currentindices = [prop.centroid for prop in currentwaterproperties] 
        currentlabels = [prop.label for prop in currentwaterproperties] 
        if len(currentindices) > 0: 
            for i in range(0,len(currentindices)):
                index = currentindices[i]
                currentlabel = currentlabels[i] 
                if currentlabel > 0:
                        previouspoint = tree.query(index)
                        for prop in waterproperties:
                               
                                      if int(prop.centroid[0]) == int(indices[previouspoint[1]][0]) and int(prop.centroid[1]) == int(indices[previouspoint[1]][1]):
                                                previouslabel = prop.label
                                                break
                        
                        if previouspoint[0] > threshold:
                              relabelimage[np.where(copyImage == currentlabel)] = currentlabel
                        else:
                              relabelimage[np.where(copyImage == currentlabel)] = previouslabel
      
                              

    
      return relabelimage

def SuperSTARPrediction(image, model, n_tiles, MaskImage, OverAllMaskImage, UseProbability = True, prob_thresh = None, nms_thresh = None):
    
    
    image = normalize(image, 1, 99.8, axis = (0,1))
    shape = [image.shape[0], image.shape[1]]
    image = zero_pad(image, 64, 64)
    
    
    if prob_thresh is not None and nms_thresh is not None:
        
        MidImage, details = model.predict_instances(image, n_tiles = n_tiles, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
    else:
        MidImage, details = model.predict_instances(image, n_tiles = n_tiles)
    
    StarImage = MidImage[:shape[0],:shape[1]]
    
    SmallProbability, SmallDistance = model.predict(image, n_tiles = n_tiles)
    grid = model.config.grid
    Probability = cv2.resize(SmallProbability, dsize=(SmallProbability.shape[1] * grid[1] , SmallProbability.shape[0] * grid[0] ))
    Distance = MaxProjectDist(SmallDistance, axis=-1)
    Distance = cv2.resize(Distance, dsize=(Distance.shape[1] * grid[1] , Distance.shape[0] * grid[0] ))
    if UseProbability:
        
        MaxProjectDistance = Probability[:shape[0],:shape[1]]

    else:
        
        MaxProjectDistance = Distance[:shape[0],:shape[1]]

    
    
    OverAllMaskImage = CleanMask( StarImage, OverAllMaskImage)
    Watershed, Markers = SuperWatershedwithMask(MaxProjectDistance, StarImage.astype('uint16'), MaskImage.astype('uint16'), OverAllMaskImage.astype('uint16'), grid)
    Watershed = fill_label_holes(Watershed.astype('uint16'))
    

    return Watershed, Markers, StarImage, MaxProjectDistance  

def CleanMask( StarImage, OverAllMaskImage):
    OverAllMaskImage = np.logical_or(OverAllMaskImage > 0, StarImage > 0)
    OverAllMaskImage = binary_erosion(OverAllMaskImage)
    OverAllMaskImage = label(OverAllMaskImage)
    OverAllMaskImage = fill_label_holes(OverAllMaskImage.astype('uint16'))
    
    return  OverAllMaskImage

def UNETPrediction3D(image, model, n_tiles, axis):
    
    
    Segmented = model.predict(image, axis, n_tiles = n_tiles)
    
    try:
       thresh = threshold_mean(Segmented)
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

def STARPrediction3D(image, model, n_tiles, MaskImage = None, smartcorrection = None, UseProbability = True, globalthreshold = 1.0E-5, extent = 0, seedpool = True, prob_thresh = None, nms_thresh = None):
    
    copymodel = model
    image = normalize(image, 1, 99.8, axis = (0,1,2))
    shape = [image.shape[1], image.shape[2]]
    image = zero_pad_time(image, 64, 64)
    grid = copymodel.config.grid

    print('Predicting Instances')
    if prob_thresh is not None and nms_thresh is not None:
       MidImage, details = model.predict_instances(image, n_tiles = n_tiles , prob_thresh = prob_thresh, nms_thresh = nms_thresh)
    else:    
      MidImage, details = model.predict_instances(image, n_tiles = n_tiles)
    print('Predicting Probabilities')
    if UseProbability:
       SmallProbability  = model.predict_prob(image, n_tiles = n_tiles)
    else:
       SmallProbability, SmallDistance  = model.predict(image, n_tiles = n_tiles) 

    print('Predictions Done')
    StarImage = MidImage[:image.shape[0],:shape[0],:shape[1]]
         
    StarImage = RemoveLabels(StarImage)    
    if UseProbability == False:
        
        SmallDistance = MaxProjectDist(SmallDistance, axis=-1)
        Distance = np.zeros([SmallDistance.shape[0] * grid[0], SmallDistance.shape[1] * grid[1], SmallDistance.shape[2] * grid[2] ])
    
    Probability = np.zeros([SmallProbability.shape[0] * grid[0],SmallProbability.shape[1] * grid[1], SmallProbability.shape[2] * grid[2] ])
    
    print('Reshaping')
    #We only allow for the grid parameter to be 1 along the Z axis
    for i in range(0, SmallProbability.shape[0]):
        Probability[i,:] = cv2.resize(SmallProbability[i,:], dsize=(SmallProbability.shape[2] * grid[2] , SmallProbability.shape[1] * grid[1] ))
        if UseProbability == False:
            Distance[i,:] = cv2.resize(SmallDistance[i,:], dsize=(SmallDistance.shape[2] * grid[2] , SmallDistance.shape[1] * grid[1] ))
    
    if UseProbability:
        
        print('Using Probability maps')
        Probability[Probability < globalthreshold ] = 0 
             
        MaxProjectDistance = Probability[:image.shape[0],:shape[0],:shape[1]]

    else:
        
        print('Using Distance maps')
        MaxProjectDistance = Distance[:image.shape[0],:shape[0],:shape[1]]

    
    print('Doing Watershedding')      
    Watershed, Markers = WatershedwithMask3D(MaxProjectDistance.astype('uint16'), StarImage.astype('uint16'), MaskImage.astype('uint16'), grid, extent,seedpool )
    Watershed = fill_label_holes(Watershed.astype('uint16'))
  
       
       

    return Watershed, MaxProjectDistance, StarImage, Markers  
 
 
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
    
def iou3D(boxA, centroid, extent = 0):
    
    ndim = len(centroid)
    inside = False
    
    Condition = [Conditioncheck(centroid, boxA, p, ndim, extent) for p in range(0,ndim)]
        
    inside = all(Condition)
    
    return inside

def Conditioncheck(centroid, boxA, p, ndim, extent):
    
      condition = False
     
      vol = extent * ( boxA[p + ndim] - boxA[p] ) / 2
      
      if centroid[p] >=  boxA[p] - vol and centroid[p] <= boxA[p + ndim] + vol:
          
           condition = True
           
      return condition     
    

def WatershedwithMask3D(Image, Label,mask, grid, extent = 0, seedpool = True): 
    properties = measure.regionprops(Label, Image) 
    binaryproperties = measure.regionprops(label(mask), Image) 
    print(seedpool)
    
    Coordinates = [prop.centroid for prop in properties] 
    BinaryCoordinates = [prop.centroid for prop in binaryproperties]
    
    Binarybbox = [prop.bbox for prop in binaryproperties]
    Coordinates = sorted(Coordinates , key=lambda k: [k[0], k[1], k[2]]) 
    
    if seedpool:
      if len(Binarybbox) > 0:    
            for i in range(0, len(Binarybbox)):
                
                box = Binarybbox[i]
                inside = [iou3D(box, star, extent) for star in Coordinates]
                
                if not any(inside) :
                         Coordinates.append(BinaryCoordinates[i])    
                         
    
    Coordinates.append((0,0,0))
   

    Coordinates = np.asarray(Coordinates)
    coordinates_int = np.round(Coordinates).astype(int) 
    
    markers_raw = np.zeros_like(Image) 
    markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates)) 
    markers = morphology.dilation(markers_raw.astype('uint16'), morphology.ball(2))
    mask = np.logical_or(mask, Label > 0)
    watershedImage = watershed(-Image, markers, mask = mask.copy() )
    
    
    return watershedImage, markers




    
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
