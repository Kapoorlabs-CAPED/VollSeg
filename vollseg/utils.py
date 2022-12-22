#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:08:41 2019
@author: vkapoor
"""

from __future__ import print_function, unicode_literals, absolute_import, division
from pickle import GLOBAL
# import matplotlib.pyplot as plt
import numpy as np
import os
from six import string_types
from tifffile import imread, imwrite
from skimage import morphology
from skimage.morphology import dilation, square
from scipy.ndimage import zoom
from skimage.morphology import remove_small_objects
from matplotlib import cm
from scipy import spatial
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from pathlib import Path
from skimage.segmentation import relabel_sequential
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.ndimage.morphology import binary_fill_holes
from skimage.util import invert as invertimage
from skimage import measure
from skimage.measure import label
from csbdeep.utils import normalize
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
import math
import pandas as pd
import napari
import glob
from skimage.util import map_array
from vollseg.matching import matching
from vollseg.seedpool import SeedPool
from vollseg.unetstarmask import UnetStarMask
from vollseg.nmslabel import NMSLabel
from skimage.measure import regionprops
from qtpy.QtWidgets import QComboBox, QPushButton
from skimage.filters import threshold_otsu
from scipy.ndimage.measurements import find_objects
from cellpose import models

Boxname = 'ImageIDBox'
GLOBAL_THRESH = 1.0E-2
GLOBAL_ERODE = 8

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
            lambda trackid=imageidbox: self.image_add(

                imageidbox.currentText(),
                self.segmentationdir + "/" +
                os.path.basename(os.path.splitext(
                    imageidbox.currentText())[0]) + '.tif',
                os.path.basename(os.path.splitext(
                    imageidbox.currentText())[0]),
                False

            )
        )

        savebutton.clicked.connect(
            lambda trackid=imageidbox: self.image_add(

                imageidbox.currentText(),
                self.segmentationdir + "/" +
                os.path.basename(os.path.splitext(
                    imageidbox.currentText())[0]) + '.tif',
                os.path.basename(os.path.splitext(
                    imageidbox.currentText())[0]),
                True

            )
        )

        self.viewer.window.add_dock_widget(
            imageidbox, name="Image", area='bottom')
        self.viewer.window.add_dock_widget(
            savebutton, name="Save Segmentations", area='bottom')

    def image_add(self, image_toread, seg_image_toread, imagename,  save=False):

        if not save:
            for layer in list(self.viewer.layers):

                if 'Image' in layer.name or layer.name in 'Image':

                    self.viewer.layers.remove(
                        layer)

            self.image = imread(image_toread)
            self.segimage = imread(seg_image_toread)

            self.viewer.add_image(
                self.image, name='Image'+imagename)
            self.viewer.add_labels(
                self.segimage, name='Image'+'Integer_Labels'+imagename)

        if save:

            ModifiedArraySeg = self.viewer.layers['Image' +
                                                  'Integer_Labels' + imagename].data
            ModifiedArraySeg = ModifiedArraySeg.astype('uint16')
            imwrite((self.segmentationdir + imagename +
                    '.tif'), ModifiedArraySeg)


def BinaryLabel(BinaryImageOriginal, max_size=15000):

    BinaryImageOriginal = BinaryImageOriginal.astype('uint16')
    image = normalizeFloatZeroOne(BinaryImageOriginal)
    image = invertimage(image)
    IntegerImage = watershed(-image)
    AugmentedLabel = remove_big_objects(IntegerImage, max_size=max_size)

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


def SimplePrediction(x, UnetModel, StarModel, n_tiles=(2, 2), UseProbability=True, min_size=20, axes='ZYX', ExpandLabels = True):

    Mask = UNETPrediction3D(x, UnetModel, n_tiles, axes, ExpandLabels)

    smart_seeds, _, _, _ = STARPrediction3D(
        x, axes, StarModel, n_tiles, unet_mask=Mask, smartcorrection=None, UseProbability=UseProbability)

    smart_seeds = smart_seeds.astype('uint16')

    return smart_seeds



def fill_label_holes(lbl_img, **kwargs):
    """Fill small holes in label image."""
    # TODO: refactor 'fill_label_holes' and 'edt_prob' to share code
    def grow(sl, interior):
        return tuple(slice(s.start-int(w[0]), s.stop+int(w[1])) for s, w in zip(sl, interior))

    def shrink(interior):
        return tuple(slice(int(w[0]), (-1 if w[1] else None)) for w in interior)
    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)
    for i, sl in enumerate(objects, 1):
        if sl is None:
            continue
        interior = [(s.start > 0, s.stop < sz)
                    for s, sz in zip(sl, lbl_img.shape)]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl, interior)] == i
        mask_filled = binary_fill_holes(grown_mask, **kwargs)[shrink_slice]
        lbl_img_filled[sl][mask_filled] = i
    return lbl_img_filled


def dilate_label_holes(lbl_img, iterations):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in (range(np.min(lbl_img), np.max(lbl_img) + 1)):
        mask = lbl_img == l
        mask_filled = binary_dilation(mask, iterations=iterations)
        lbl_img_filled[mask_filled] = l
    return lbl_img_filled


def match_labels(ys, iou_threshold=0.5):
    """
    Matches object ids in a list of label images based on a matching criterion.
    For i=0..len(ys)-1 consecutively matches ys[i+1] with ys[i],
    matching objects retain their id, non matched objects will be assigned a new id
    Example
    -------
    import numpy as np
    from stardist.data import test_image_nuclei_2d
    from stardist.matching import match_labels
    _y = test_image_nuclei_2d(return_mask=True)[1]
    labels = np.stack([_y, 2*np.roll(_y,10)], axis=0)
    labels_new = match_labels(labels)
    Parameters
    ----------
    ys : np.ndarray, tuple of np.ndarray
          list/array of integer labels (2D or 3D)
    """
    print('Stitching')
    ys = np.asarray(ys)
    if not ys.ndim in (3, 4):
        raise ValueError('label image y should be 3 or 4 dimensional!')

    def _match_single(x, y):
        res = matching(x, y, report_matches=True)

        pairs = tuple(p for p, s in zip(res.matched_pairs,
                      res.matched_scores) if s >= iou_threshold)
        map_dict = dict((i2, i1) for i1, i2 in pairs)

        y2 = np.zeros_like(y)
        y_labels = set(np.unique(y)) - {0}

        # labels that can be used for non-matched objects
        label_reservoir = list(
            set(np.arange(1, len(y_labels)+1)) - set(map_dict.values()))
        for r in regionprops(y):
            m = (y[r.slice] == r.label)
            if r.label in map_dict:
                y2[r.slice][m] = map_dict[r.label]
            else:
                y2[r.slice][m] = label_reservoir.pop(0)

        return y2

    ys_new = ys.copy()
    
    
    for i in tqdm(range(len(ys)-1)):
           ys_new[i+1] = _match_single(ys_new[i], ys[i+1])

    return ys_new



def remove_big_objects(ar: np.ndarray, max_size):

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





def BinaryDilation(Image, iterations=1):

    DilatedImage = binary_dilation(Image, iterations=iterations)

    return DilatedImage


def CCLabels(fname, max_size=15000):

    BinaryImageOriginal = imread(fname)
    Orig = normalizeFloatZeroOne(BinaryImageOriginal)
    InvertedBinaryImage = invertimage(BinaryImageOriginal)
    BinaryImage = normalizeFloatZeroOne(InvertedBinaryImage)
    image = binary_dilation(BinaryImage)
    image = invertimage(image)
    IntegerImage = label(image)
    labelclean = remove_big_objects(IntegerImage, max_size=max_size)
    AugmentedLabel = dilation(labelclean, selem=square(3))
    AugmentedLabel = np.multiply(AugmentedLabel,  Orig)

    return AugmentedLabel






def CreateTrackMate_CSV(Label, Name, savedir):

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




def SmartSkel(smart_seedsLabels, ProbImage, RGB = False):

    if RGB:
        return smart_seedsLabels > 0
    SegimageB = find_boundaries(smart_seedsLabels)
    invertProbimage = 1 - ProbImage
    image_max = np.add(invertProbimage, SegimageB)

    pixel_condition = (image_max < 1.2)
    pixel_replace_condition = 0
    image_max = image_conditionals(image_max,pixel_condition,pixel_replace_condition )

    Skeleton = skeletonize(image_max.astype('uint16') > 0)

    return Skeleton

def Skel(smart_seedsLabels, RGB = False):

    if RGB:
        return smart_seedsLabels > 0
    image_max = find_boundaries(smart_seedsLabels)
    
    Skeleton = skeletonize(image_max.astype('uint16') > 0)

    return Skeleton

# If there are neighbouring seeds we do not put more seeds




def Region_embedding(image, region, sourceimage, RGB = False):

    returnimage = np.zeros(image.shape)
    if region is not None:
            if len(region) == 4 and len(image.shape) == 2:
                rowstart = region[0]
                colstart = region[1]
                endrow = region[2]
                endcol = region[3]
                returnimage[rowstart:endrow, colstart:endcol] = sourceimage
            if len(image.shape) == 3 and len(region) == 6  and RGB == False:
                zstart = region[0]
                rowstart = region[1]
                colstart = region[2]
                zend = region[3]
                endrow = region[4]
                endcol = region[5]
                returnimage[zstart:zend, rowstart:endrow,
                            colstart:endcol] = sourceimage

            if len(image.shape) == 3 and len(region) == 4  and RGB == False:
                rowstart = region[0]
                colstart = region[1]
                endrow = region[2]
                endcol = region[3]
                returnimage[0:image.shape[0], rowstart:endrow,
                            colstart:endcol] = sourceimage

            if len(image.shape) == 3 and len(region) == 4 and RGB:
                returnimage = returnimage[:,:,0]
                rowstart = region[0]
                colstart = region[1]
                endrow = region[2]
                endcol = region[3]
                returnimage[rowstart:endrow,
                            colstart:endcol] = sourceimage

    else:
        returnimage = image
    return returnimage


def VollSeg2D(image, unet_model, star_model, noise_model=None, roi_model=None,  prob_thresh=None, nms_thresh=None, axes='YX', min_size_mask=5, min_size=5,
              max_size=10000000, dounet=True, n_tiles=(2, 2), ExpandLabels = True,  donormalize=True, lower_perc=1, upper_perc=99.8, UseProbability=True, RGB=False, 
              seedpool=True):

    print('Generating SmartSeed results')

    
    if star_model is not None:
        nms_thresh = star_model.thresholds[1]
    elif nms_thresh is not None:
        nms_thresh = nms_thresh
    else:
        nms_thresh = 0

    if RGB:
        axes = 'YXC'
    if 'T' in axes:
        axes = 'YX'
        if RGB:
              axes = 'YXC'  
    if noise_model is not None:
        print('Denoising Image')

        image = noise_model.predict(image.astype('float32'), axes=axes, n_tiles=n_tiles)
        pixel_condition = (image < 0)
        pixel_replace_condition = 0
        image = image_conditionals(image,pixel_condition,pixel_replace_condition ) 

    Mask = None
    Mask_patch = None
    roi_image=None
    if roi_model is not None:
        model_dim = roi_model.config.n_dim
        assert model_dim == len(
            image.shape), f'For 2D images the region of interest model has to be 2D, model provided had {model_dim} instead'
        roi_image = UNETPrediction3D(
            image, roi_model, n_tiles, axes,ExpandLabels = ExpandLabels)
        roi_bbox = Bbox_region(roi_image)
        if roi_bbox is not None:
                rowstart = roi_bbox[0]
                colstart = roi_bbox[1]
                endrow = roi_bbox[2]
                endcol = roi_bbox[3]
                region = (slice(rowstart, endrow),
                        slice(colstart, endcol))
                # The actual pixels in that region.
                patch = image[region]
        else:

                 patch = image        

    

    else:

        patch = image

        region = (slice(0, image.shape[0]),
                  slice(0, image.shape[1]))
        rowstart = 0
        colstart = 0
        endrow = image.shape[1]
        endcol = image.shape[0]
        roi_bbox = [colstart, rowstart, endcol, endrow]
    if dounet:

        if unet_model is not None:
                print('UNET segmentation on Image')

                Segmented = unet_model.predict(image.astype('float32'), axes, n_tiles=n_tiles)
        else:
                Segmented = image
        if RGB:
                Segmented = Segmented[:, :, 0]

        thresholds = threshold_otsu(Segmented)

        Binary = Segmented > thresholds
        Mask = Binary.copy() 
            
        Mask = Region_embedding(image, roi_bbox, Mask, RGB = RGB)
        Mask_patch = Mask.copy()
    elif noise_model is not None and dounet == False:

        Mask = np.zeros(patch.shape)
        thresholds = threshold_otsu(patch)
        Mask = patch > thresholds
        Mask = label(Mask)
        Mask = remove_small_objects(
            Mask.astype('uint16'), min_size=min_size_mask)
        Mask = remove_big_objects(Mask.astype('uint16'), max_size=max_size)
        
        if RGB:
                Mask = Mask[:, :, 0]
                Mask_patch = Mask_patch[:,:,0]
        Mask = Region_embedding(image, roi_bbox, Mask, RGB = RGB)
        Mask_patch = Mask.copy()    
    # Smart Seed prediction
    print('Stardist segmentation on Image')
    if RGB:
        axis = (0,1,2)
    else:
        axis = (0,1)    
    if donormalize:
        patch_star = normalize(patch.astype('float32'), lower_perc, upper_perc, axis=axis) 
    else:
        patch_star = patch
    smart_seeds, Markers, star_labels, proabability_map = SuperSTARPrediction(
        patch_star, star_model, n_tiles, unet_mask=Mask_patch, UseProbability=UseProbability, prob_thresh=prob_thresh, nms_thresh=nms_thresh, seedpool = seedpool)
    smart_seeds = remove_small_objects(
        smart_seeds.astype('uint16'), min_size=min_size)
    smart_seeds = remove_big_objects(
        smart_seeds.astype('uint16'), max_size=max_size)
    Skeleton = SmartSkel(smart_seeds, proabability_map, RGB)
    Skeleton = Skeleton > 0
    # For avoiding pixel level error
    if Mask is not None:
       Mask = expand_labels(Mask, distance=1)
   

    smart_seeds = expand_labels(smart_seeds, distance=1)

    smart_seeds = Region_embedding(image, roi_bbox, smart_seeds, RGB = RGB)
    Markers = Region_embedding(image, roi_bbox, Markers, RGB = RGB)
    star_labels = Region_embedding(image, roi_bbox, star_labels, RGB = RGB)
    proabability_map = Region_embedding(image, roi_bbox, proabability_map, RGB = RGB)
    Skeleton = Region_embedding(image, roi_bbox, Skeleton, RGB = RGB)
    if Mask is None:
        Mask = smart_seeds > 0
   

    if noise_model is None and roi_image is not None:
        return smart_seeds.astype('uint16'), Mask.astype('uint16'), star_labels.astype('uint16'), proabability_map, Markers.astype('uint16'), Skeleton.astype('uint16'), roi_image.astype('uint16')
    
    if noise_model is None and roi_image is None:
        return smart_seeds.astype('uint16'), Mask.astype('uint16'), star_labels.astype('uint16'), proabability_map, Markers.astype('uint16'), Skeleton.astype('uint16')


    if noise_model is not None and roi_image is not None:
        return smart_seeds.astype('uint16'), Mask.astype('uint16'), star_labels.astype('uint16'), proabability_map, Markers.astype('uint16'), Skeleton.astype('uint16'), image, roi_image.astype('uint16')

    if noise_model is not None and roi_image is None:
        return smart_seeds.astype('uint16'), Mask.astype('uint16'), star_labels.astype('uint16'), proabability_map, Markers.astype('uint16'), Skeleton.astype('uint16'), image



def VollSeg_nolabel_precondition(image, Finalimage):

   
    ndim = len(image.shape) 
    if ndim == 3:
        for i in range(image.shape[0]):
              Finalimage[i] = expand_labels(Finalimage[i], distance = GLOBAL_ERODE)

    return Finalimage    
             

def VollSeg_label_precondition(image, overall_mask, Finalimage):

        ndim = len(image.shape)
        if ndim == 3:
          for i in range(image.shape[0]):
              Finalimage[i] = expand_labels(Finalimage[i], distance = 50)
          pixel_condition = (overall_mask == 0)
          pixel_replace_condition = 0
          Finalimage = image_conditionals(Finalimage,pixel_condition,pixel_replace_condition )  
       
        return Finalimage

def VollSeg_label_expansion(image, overall_mask, Finalimage, Skeleton, RGB):

    for i in range(image.shape[0]):
                            Finalimage[i,:] = expand_labels(Finalimage[i,:], distance = 50)
                            Skeleton[i, :] = Skel(Finalimage[i,:], RGB)
                            Skeleton[i, :] = Skeleton[i, :] > 0
    pixel_condition = (overall_mask == 0)
    pixel_replace_condition = 0
    Finalimage = image_conditionals(Finalimage,pixel_condition,pixel_replace_condition )        
    Skeleton = image_conditionals(Skeleton,pixel_condition,pixel_replace_condition )

    return Finalimage, Skeleton 

def VollSeg_nolabel_expansion(image, Finalimage, Skeleton, RGB):
    
    for i in range(image.shape[0]):
                   Finalimage[i,:] = expand_labels(Finalimage[i,:], distance = GLOBAL_ERODE) 
                   Skeleton[i, :] = Skel(Finalimage[i,:], RGB)
                   Skeleton[i, :] = Skeleton[i, :] > 0 
                   
    return Finalimage, Skeleton 


def VollSeg_unet(image, unet_model=None, roi_model=None, n_tiles=(2, 2), axes='YX', ExpandLabels = True, noise_model=None, min_size_mask=100, max_size=10000000,  RGB=False, iou_threshold=0.3, slice_merge=False, dounet=True, erosion_iterations = 15):

    ndim = len(image.shape)    
    if len(n_tiles)!=ndim:
        if ndim == 3:
            n_tiles = (n_tiles[-3], n_tiles[-2], n_tiles[-1])
        if ndim == 2:
            n_tiles = (n_tiles[-2], n_tiles[-1])    

    if roi_model is None:
        if RGB:
            if n_tiles is not None:
                n_tiles = (n_tiles[0], n_tiles[1], 1)

        

        if noise_model is not None:
            image = noise_model.predict(image.astype('float32'), axes, n_tiles=n_tiles)
            pixel_condition = (image < 0)
            pixel_replace_condition = 0
            image = image_conditionals(image,pixel_condition,pixel_replace_condition )
            
        if dounet and unet_model is not None:
            Segmented = unet_model.predict(image.astype('float32'), axes, n_tiles=n_tiles)
        else:
            Segmented = image
        if RGB:
            Segmented = Segmented[:, :, 0]

        thresholds = threshold_otsu(Segmented)

        Binary = Segmented > thresholds
        overall_mask = Binary.copy()
        
        if ndim == 3:
                for i in range(image.shape[0]):
                    overall_mask[i] = binary_dilation(overall_mask[i], iterations = erosion_iterations)
                    overall_mask[i] = binary_erosion(overall_mask[i], iterations = erosion_iterations)
                    overall_mask[i] = fill_label_holes(overall_mask[i])
    
       

        Binary = label(Binary)

        if ndim == 2:
            Binary = remove_small_objects(
                Binary.astype('uint16'), min_size=min_size_mask)
            Binary = remove_big_objects(
                Binary.astype('uint16'), max_size=max_size)
            Binary = fill_label_holes(Binary)
            Finalimage = relabel_sequential(Binary)[0]
            Skeleton = Skel(Finalimage, RGB)
            Skeleton = Skeleton > 0
        if ndim == 3 and slice_merge:
            for i in range(image.shape[0]):
                Binary[i] = label(Binary[i])
            
            Binary = match_labels(Binary, iou_threshold=iou_threshold)
            Binary = fill_label_holes(Binary)

        if ndim == 3: 
            for i in range(image.shape[0]):
                Binary[i]  = remove_small_objects(
                    Binary[i] .astype('uint16'), min_size=min_size_mask)
                Binary[i]  = remove_big_objects(
                    Binary[i] .astype('uint16'), max_size=max_size)    
            Finalimage = relabel_sequential(Binary)[0]
            Skeleton = np.zeros_like(Finalimage)


            if ExpandLabels:
                       
                    Finalimage, Skeleton = VollSeg_label_expansion(image, overall_mask, Finalimage, Skeleton, RGB)   
                    
    elif roi_model is not None:

        if noise_model is not None:
            image = noise_model.predict(image.astype('float32'), axes, n_tiles=n_tiles)

            pixel_condition = (image < 0)
            pixel_replace_condition = 0
            image = image_conditionals(image,pixel_condition,pixel_replace_condition) 

        model_dim = roi_model.config.n_dim
        if model_dim < len(image.shape):
            if len(n_tiles) == len(image.shape):
                tiles = (n_tiles[1], n_tiles[2])
            else:
                tiles = n_tiles
            maximage = np.amax(image, axis=0)
            s_Binary = UNETPrediction3D(
                maximage, roi_model, tiles, 'YX', ExpandLabels = ExpandLabels)

            s_Binary = label(s_Binary)
            s_Binary = remove_small_objects(
                s_Binary.astype('uint16'), min_size=min_size_mask)
            s_Binary = remove_big_objects(
                s_Binary.astype('uint16'), max_size=max_size)
            s_Binary = fill_label_holes(s_Binary)

            s_Finalimage = relabel_sequential(s_Binary)[0]

            s_Skeleton = skeletonize(find_boundaries(s_Finalimage > 0))
            Binary = np.zeros_like(image)
            Skeleton = np.zeros_like(image)
            Finalimage = np.zeros_like(image)
            for i in range(0, image.shape[0]):

               Binary[i] = s_Binary
               Skeleton[i] = s_Skeleton
               Finalimage[i] = s_Finalimage

        elif model_dim == len(image.shape):
            Binary = UNETPrediction3D(
                image, roi_model, n_tiles, axes, ExpandLabels = ExpandLabels)

            Binary = label(Binary)
            if ndim == 3 and slice_merge:
                for i in range(image.shape[0]):
                    Binary[i] = label(Binary[i])
                    

                Binary = match_labels(Binary, iou_threshold=iou_threshold)
                Binary = fill_label_holes(Binary)
                for i in range(image.shape[0]):
                    Binary[i] = remove_small_objects(
                        Binary[i].astype('uint16'), min_size=min_size_mask)
                    Binary[i] = remove_big_objects(
                        Binary[i].astype('uint16'), max_size=max_size)

            Finalimage = relabel_sequential(Binary)[0]

            Skeleton = skeletonize(find_boundaries(Finalimage > 0))
        



    return Finalimage.astype('uint16'), Skeleton, image

def _cellpose_star_time_block(cellpose_model,
                        custom_cellpose_model,
                        cellpose_model_name,
                        image_membrane,
                        image_nuclei,
                        diameter_cellpose,
                        flow_threshold,
                        cellprob_threshold,
                        stitch_threshold,
                        anisotropy,
                        pretrained_cellpose_model_path,
                        gpu,
                        unet_model,
                        unet_membrane_model,
                        star_model, 
                        roi_model,
                        ExpandLabels,
                        axes,
                        noise_model,
                        prob_thresh,
                        nms_thresh,
                        donormalize,
                        n_tiles,
                        UseProbability,
                        dounet,
                        seedpool,
                        startZ,
                        slice_merge,
                        iou_threshold,
                        lower_perc,
                        upper_perc,
                        min_size_mask,
                        min_size,
                        max_size,
                        do_3D):
    
    
    
    if cellpose_model is not None:
                
                if custom_cellpose_model:
                    cellpose_model = models.Cellpose(gpu=gpu, model_type = cellpose_model_name)
                    if anisotropy is not None:
                            cellres = tuple(
                             zip(
                                *tuple(cellpose_model.eval(_x, diameter=diameter_cellpose,  flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold, stitch_threshold=stitch_threshold, anisotropy=anisotropy, do_3D=do_3D)
                                       for _x in tqdm(image_nuclei))))
                    else:

                        cellres = tuple(
                            zip(
                                *tuple(
                                    cellpose_model.eval(_x, diameter=diameter_cellpose, flow_threshold=flow_threshold,
                                                        cellprob_threshold=cellprob_threshold,
                                                        stitch_threshold=stitch_threshold,
                                                        do_3D=do_3D)
                                    for _x in tqdm(image_nuclei))))
                    
               
                else:   
                    cellpose_model = models.CellposeModel(gpu=gpu, pretrained_model = pretrained_cellpose_model_path)
                    if anisotropy is not None:
                        cellres = tuple(
                         zip(
                            *tuple(cellpose_model.eval(image_membrane, diameter=diameter_cellpose,  flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold, stitch_threshold=stitch_threshold, anisotropy=anisotropy, do_3D=do_3D)
                                   for _x in tqdm(image_nuclei))))
                    else:
                        cellres = tuple(
                            zip(
                                *tuple(cellpose_model.eval(image_membrane, diameter=diameter_cellpose,
                                                           flow_threshold=flow_threshold,
                                                           cellprob_threshold=cellprob_threshold,
                                                           stitch_threshold=stitch_threshold,
                                                           do_3D=do_3D)
                                       for _x in tqdm(image_nuclei))))

            
    if star_model is not None:
                  if prob_thresh is None and nms_thresh is None:
                        prob_thresh = star_model.thresholds.prob
                        nms_thresh = star_model.thresholds.nms
                  res = tuple(
                     zip(
                        *tuple(VollSeg3D(image_nuclei[i],  unet_model, star_model, axes=axes, noise_model=noise_model, roi_model=roi_model,ExpandLabels= ExpandLabels,  prob_thresh=prob_thresh, nms_thresh=nms_thresh, donormalize=donormalize, lower_perc=lower_perc, upper_perc=upper_perc, min_size_mask=min_size_mask, min_size=min_size, max_size=max_size,
                                        n_tiles=n_tiles, image_membrane = image_membrane[i], UseProbability=UseProbability, unet_membrane_model = unet_membrane_model,
                                        dounet=dounet, seedpool=seedpool, startZ=startZ, slice_merge=slice_merge, iou_threshold=iou_threshold) for i in tqdm(range(image_nuclei.shape[0])))))
    
    return cellres, res

def _cellpose_star_block(cellpose_model,
                        custom_cellpose_model,
                        cellpose_model_name,
                        image_membrane,
                        image_nuclei,
                        diameter_cellpose,
                        flow_threshold,
                        cellprob_threshold,
                        stitch_threshold,
                        anisotropy,
                        pretrained_cellpose_model_path,
                        gpu,
                        unet_model,
                        unet_membrane_model,
                        star_model, 
                        roi_model,
                        ExpandLabels,
                        axes,
                        noise_model,
                        prob_thresh,
                        nms_thresh,
                        donormalize,
                        n_tiles,
                        UseProbability,
                        dounet,
                        seedpool,
                        startZ,
                        slice_merge,
                        iou_threshold,
                        lower_perc,
                        upper_perc,
                        min_size_mask,
                        min_size,
                        max_size,
                        do_3D
                        ):
    
    cellres = None
    res = None
    if cellpose_model is not None:
                
                if custom_cellpose_model:
                    cellpose_model = models.Cellpose(gpu=gpu, model_type = cellpose_model_name)
                    if anisotropy is not None:
                       cellres = cellpose_model.eval(image_membrane, diameter=diameter_cellpose,  flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold, stitch_threshold=stitch_threshold, anisotropy=anisotropy, do_3D=do_3D)
                    else:
                        cellres = cellpose_model.eval(image_membrane, diameter=diameter_cellpose,
                                                      flow_threshold=flow_threshold,
                                                      cellprob_threshold=cellprob_threshold,
                                                      stitch_threshold=stitch_threshold,
                                                      do_3D=do_3D)

                else:   
                    cellpose_model = models.CellposeModel(gpu=gpu, pretrained_model = pretrained_cellpose_model_path)
                    if anisotropy is not None:
                        cellres = cellpose_model.eval(image_membrane, diameter=diameter_cellpose,  flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold, stitch_threshold=stitch_threshold, anisotropy=anisotropy, do_3D=do_3D)
                    else:
                        cellres = cellpose_model.eval(image_membrane, diameter=diameter_cellpose,  flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold, stitch_threshold=stitch_threshold, do_3D=do_3D)

    if star_model is not None:
                
                res = VollSeg3D(image_nuclei,  unet_model,  star_model, roi_model=roi_model,ExpandLabels= ExpandLabels,  axes=axes, noise_model=noise_model, prob_thresh=prob_thresh, nms_thresh=nms_thresh, donormalize=donormalize, lower_perc=lower_perc, upper_perc=upper_perc, min_size_mask=min_size_mask, min_size=min_size, max_size=max_size,
                                    n_tiles=n_tiles, image_membrane = image_membrane, UseProbability=UseProbability,unet_membrane_model = unet_membrane_model,  dounet=dounet, seedpool=seedpool, startZ=startZ, slice_merge=slice_merge, iou_threshold=iou_threshold)
                    
    return cellres, res

def VollCellSeg(image: np.ndarray, 
                diameter_cellpose: float = 34.6,
                stitch_threshold: float = 0.5,
                channel_membrane: int = 0,
                channel_nuclei: int = 1,
                flow_threshold: float = 0.4, 
                cellprob_threshold: float = 0.0,
                anisotropy = None,
                star_model = None, 
                unet_model = None,
                unet_membrane_model = None,
                roi_model = None,
                noise_model=None,
                cellpose_model = None, 
                custom_cellpose_model: bool = False, 
                pretrained_cellpose_model_path: str = None,
                cellpose_model_name = 'cyto2',
                gpu: bool = False,
                axes: str ='ZYX',  
                prob_thresh: float =None, 
                ExpandLabels: bool = False, 
                nms_thresh: float=None, 
                min_size_mask: int =10, 
                min_size: int =10, 
                max_size: int = 10000,
                erosion_iterations: int = 5,
                n_tiles: tuple = (1, 1, 1), 
                UseProbability: bool =True,  
                donormalize: bool =True, 
                lower_perc: float = 1.0, 
                upper_perc: float =99.8, 
                dounet: bool = True, 
                seedpool: bool = True, 
                save_dir: str = None, 
                Name : str ='Result',  
                startZ : int = 0, 
                slice_merge : bool = False, 
                iou_threshold: float = 0.3, 
                do_3D: bool =False,
                ):
    
    
    if prob_thresh is None and nms_thresh is None:
                        prob_thresh = star_model.thresholds.prob
                        nms_thresh = star_model.thresholds.nms
    
    if len(image.shape) == 3 and 'T' not in axes:
        #Just a 3D image
        image_membrane = image
        image_nuclei = image
                            
        cellres, res = _cellpose_star_block(cellpose_model,
                        custom_cellpose_model,
                        cellpose_model_name,
                        image_membrane,
                        image_nuclei,
                        diameter_cellpose,
                        flow_threshold,
                        cellprob_threshold,
                        stitch_threshold,
                        anisotropy,
                        pretrained_cellpose_model_path,
                        gpu,
                        unet_model,
                        unet_membrane_model,
                        star_model, 
                        roi_model,
                        ExpandLabels,
                        axes,
                        noise_model,
                        prob_thresh,
                        nms_thresh,
                        donormalize,
                        n_tiles,
                        UseProbability,
                        dounet,
                        seedpool,
                        startZ,
                        slice_merge,
                        iou_threshold,
                        lower_perc,
                        upper_perc,
                        min_size_mask,
                        min_size,
                        do_3D)
                            
    if len(image.shape) == 4 and 'T' not in axes:
            image_membrane = image[:,channel_membrane,:,:]
            image_nuclei = image[:,channel_nuclei,:,:]
            
            
            cellres, res = _cellpose_star_block(cellpose_model,
                        custom_cellpose_model,
                        cellpose_model_name,
                        image_membrane,
                        image_nuclei,
                        diameter_cellpose,
                        flow_threshold,
                        cellprob_threshold,
                        stitch_threshold,
                        anisotropy,
                        pretrained_cellpose_model_path,
                        gpu,
                        unet_model,
                        unet_membrane_model,
                        star_model, 
                        roi_model,
                        ExpandLabels,
                        axes,
                        noise_model,
                        prob_thresh,
                        nms_thresh,
                        donormalize,
                        n_tiles,
                        UseProbability,
                        dounet,
                        seedpool,
                        startZ,
                        slice_merge,
                        iou_threshold,
                        lower_perc,
                        upper_perc,
                        min_size_mask,
                        min_size,
                        max_size,
                        do_3D)
            
                
    if len(image.shape) > 4 and 'T' in axes:
           
            if len(n_tiles) == 4:
                  n_tiles = (n_tiles[1], n_tiles[2], n_tiles[3])
            image_membrane = image[:,:,channel_membrane,:,:]
            image_nuclei = image[:,:,channel_nuclei,:,:]
            cellres, res = _cellpose_star_time_block(cellpose_model,
                        custom_cellpose_model,
                        cellpose_model_name,
                        image_membrane,
                        image_nuclei,
                        diameter_cellpose,
                        flow_threshold,
                        cellprob_threshold,
                        stitch_threshold,
                        anisotropy,
                        pretrained_cellpose_model_path,
                        gpu,
                        unet_model,
                        unet_membrane_model,
                        star_model, 
                        roi_model,
                        ExpandLabels,
                        axes,
                        noise_model,
                        prob_thresh,
                        nms_thresh,
                        donormalize,
                        n_tiles,
                        UseProbability,
                        dounet,
                        seedpool,
                        startZ,
                        slice_merge,
                        iou_threshold,
                        lower_perc,
                        upper_perc,
                        min_size_mask,
                        min_size,
                        max_size,
                        do_3D)

  
    if cellpose_model is not None and custom_cellpose_model:
        cellpose_masks, flows, styles, diams = cellres
    if cellpose_model is not None and not custom_cellpose_model:
        cellpose_masks, flows, styles = cellres    


    cellpose_masks_copy = cellpose_masks.copy()
    if noise_model is None and star_model is not None and  roi_model is not None and cellpose_model is None and unet_membrane_model is None:
        Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton, roi_image = res
        
    if noise_model is None and star_model is not None and  roi_model is not None and cellpose_model is not None and unet_membrane_model is None:
        Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton, roi_image = res    
        cellpose_base, vollcellseg = _cellpose_block(axes, flows, lower_perc, upper_perc, cellpose_masks_copy, Sizedsmart_seeds, SizedMask, min_size_mask, max_size, nms_thresh, image_membrane)
    
    if noise_model is None and star_model is not None and  roi_model is not None and cellpose_model is not None and unet_membrane_model is not None:
        Sizedsmart_seeds, SizedMask, SizedMembraneMask, star_labels, proabability_map, Markers, Skeleton, roi_image = res    
        cellpose_base, vollcellseg = _cellpose_block(axes, flows, lower_perc, upper_perc, cellpose_masks_copy, Sizedsmart_seeds, SizedMembraneMask, min_size_mask, max_size, nms_thresh, image_membrane)



    if noise_model is None and star_model is not None and  roi_model is None and cellpose_model is None and unet_membrane_model is None:
        Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton = res  
        
    if noise_model is None and star_model is not None and  roi_model is None and cellpose_model is not None and unet_membrane_model is None:
        Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton = res
        cellpose_base, vollcellseg = _cellpose_block(axes, flows, lower_perc, upper_perc, cellpose_masks_copy, Sizedsmart_seeds, SizedMask, min_size_mask, max_size, nms_thresh, image_membrane)
    
    if noise_model is None and star_model is not None and  roi_model is None and cellpose_model is not None and unet_membrane_model is not None:
        Sizedsmart_seeds, SizedMask, SizedMembraneMask, star_labels, proabability_map, Markers, Skeleton = res
        cellpose_base, vollcellseg = _cellpose_block(axes, flows, lower_perc, upper_perc, cellpose_masks_copy, Sizedsmart_seeds, SizedMembraneMask, min_size_mask, max_size, nms_thresh, image_membrane)
  

    if noise_model is not None and star_model is not None and  roi_model is not None and cellpose_model is None and unet_membrane_model is None:
        Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton,  image, roi_image = res
        
    if noise_model is not None and star_model is not None and  roi_model is not None and cellpose_model is not None and unet_membrane_model is None:
        Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton,  image, roi_image = res    
        cellpose_base, vollcellseg = _cellpose_block(axes, flows, lower_perc, upper_perc, cellpose_masks_copy, Sizedsmart_seeds, SizedMask, min_size_mask, max_size, nms_thresh, image_membrane)
   
    if noise_model is not None and star_model is not None and  roi_model is not None and cellpose_model is not None and unet_membrane_model is not None:
        Sizedsmart_seeds, SizedMask, SizedMembraneMask, star_labels, proabability_map, Markers, Skeleton,  image, roi_image = res    
        cellpose_base, vollcellseg = _cellpose_block(axes, flows, lower_perc, upper_perc, cellpose_masks_copy, Sizedsmart_seeds, SizedMembraneMask, min_size_mask, max_size, nms_thresh, image_membrane)


    if noise_model is not None and star_model is not None and  roi_model is None and cellpose_model is not None and unet_membrane_model is None:
        Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton,  image = res    
        
    if noise_model is None and star_model is not None and  roi_model is not None and cellpose_model is not None and unet_membrane_model is None:
        
              Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton, roi_image = res
              cellpose_base, vollcellseg = _cellpose_block(axes, flows, lower_perc, upper_perc, cellpose_masks_copy, Sizedsmart_seeds, SizedMask, min_size_mask, max_size, nms_thresh, image_membrane)
      
    if noise_model is None and star_model is not None and  roi_model is not None and cellpose_model is not None and unet_membrane_model is not None:
        
              Sizedsmart_seeds, SizedMask, SizedMembraneMask, star_labels, proabability_map, Markers, Skeleton, roi_image = res
              cellpose_base, vollcellseg = _cellpose_block(axes, flows, lower_perc, upper_perc, cellpose_masks_copy, Sizedsmart_seeds, SizedMembraneMask, min_size_mask, max_size, nms_thresh, image_membrane)
            
        
    elif noise_model is not None and star_model is None and roi_model is None and unet_model is None and cellpose_model is not None and unet_membrane_model is None:

         SizedMask, Skeleton, image = res
   
    
    elif star_model is None and  roi_model is None and unet_model is not None and noise_model is not None and cellpose_model is not None and unet_membrane_model is None:

        SizedMask, Skeleton, image = res

    elif star_model is None and  roi_model is not None and unet_model is not None and noise_model is not None and cellpose_model is not None and unet_membrane_model is None:

        SizedMask, Skeleton, image = res    

    elif star_model is None and  roi_model is None and unet_model is not None and noise_model is None and cellpose_model is not None and unet_membrane_model is None:

        SizedMask, Skeleton, image = res    

    elif star_model is None and  roi_model is not None and unet_model is None and noise_model is None and cellpose_model is not None and unet_membrane_model is None:

        roi_image, Skeleton, image = res
        SizedMask = roi_image

    elif star_model is None and  roi_model is not None and unet_model is None and noise_model is not None and cellpose_model is not None and unet_membrane_model is None:

        roi_image, Skeleton, image = res
        SizedMask = roi_image    

    elif star_model is None and  roi_model is not None and unet_model is not None and noise_model is None and cellpose_model is not None and unet_membrane_model is None:

        roi_image, Skeleton, image = res
        SizedMask = roi_image

    if save_dir is not None:
        print('Saving Results ...')
        Path(save_dir).mkdir(exist_ok=True)

        if cellpose_model is not None:
            cellpose_results = save_dir + 'CellPose/'
            Path(cellpose_results).mkdir(exist_ok=True)
            imwrite((cellpose_results + Name + '.tif'),
                    np.asarray(cellpose_masks).astype('uint16'))
            
            vollcellpose_results = save_dir + 'VollCellPose/'
            Path(vollcellpose_results).mkdir(exist_ok=True)
            imwrite((vollcellpose_results + Name + '.tif'),
                    np.asarray(vollcellseg).astype('uint16'))
            
        if unet_membrane_model is not None:
            
            unet_membrane_results = save_dir + 'BinaryMembraneMask/'
            Path(unet_membrane_results).mkdir(exist_ok=True)
             
            imwrite((unet_membrane_results + Name + '.tif'),
                    np.asarray(SizedMembraneMask).astype('uint16'))
        
        if  roi_model is not None:
            roi_results = save_dir + 'Roi/'
            Path(roi_results).mkdir(exist_ok=True)
            imwrite((roi_results + Name + '.tif'),
                    np.asarray(roi_image).astype('uint16'))

        if unet_model is not None:
            unet_results = save_dir + 'BinaryMask/'
            skel_unet_results = save_dir + 'Skeleton/'
            Path(unet_results).mkdir(exist_ok=True)
            Path(skel_unet_results).mkdir(exist_ok=True)
             
            imwrite((unet_results + Name + '.tif'),
                    np.asarray(SizedMask).astype('uint16'))
            imwrite((skel_unet_results + Name + '.tif'),
                    np.asarray(Skeleton).astype('uint16'))        
        if star_model is not None:
            vollseg_results = save_dir + 'VollSeg/'
            stardist_results = save_dir + 'StarDist/'
            probability_results = save_dir + 'Probability/'
            marker_results = save_dir + 'Markers/'
            skel_results = save_dir + 'Skeleton/'
            Path(skel_results).mkdir(exist_ok=True)
            Path(vollseg_results).mkdir(exist_ok=True)
            Path(stardist_results).mkdir(exist_ok=True)
            Path(probability_results).mkdir(exist_ok=True)
            Path(marker_results).mkdir(exist_ok=True)
            imwrite((stardist_results + Name + '.tif'),
                    np.asarray(star_labels).astype('uint16'))
            imwrite((vollseg_results + Name + '.tif'),
                    np.asarray(Sizedsmart_seeds).astype('uint16'))
            imwrite((probability_results + Name + '.tif'),
                    np.asarray(proabability_map).astype('float32'))
            imwrite((marker_results + Name + '.tif'),
                    np.asarray(Markers).astype('uint16'))
            imwrite((skel_results + Name + '.tif'), np.asarray(Skeleton))
        if noise_model is not None:
            denoised_results = save_dir + 'Denoised/'
            Path(denoised_results).mkdir(exist_ok=True)
            imwrite((denoised_results + Name + '.tif'),
                    np.asarray(image).astype('float32'))

      
    # If denoising is not done but stardist and unet models are supplied we return the stardist, vollseg and semantic segmentation maps
    if noise_model is None and star_model is not None and  roi_model is not None and cellpose_model is None:

        return Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton, roi_image
    
    if noise_model is None and star_model is not None and  roi_model is not None and cellpose_model is not None:

        return Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton, roi_image, cellpose_masks, vollcellseg
    
    if noise_model is None and star_model is not None and  roi_model is not None and cellpose_model is not None and unet_membrane_model is not None:

        return Sizedsmart_seeds, SizedMask, SizedMembraneMask, star_labels, proabability_map, Markers, Skeleton, roi_image, cellpose_masks, vollcellseg

    elif noise_model is None and star_model is not None and  roi_model is  None and cellpose_model is None:

        return Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton 
    
    elif noise_model is None and star_model is not None and  roi_model is  None and cellpose_model is not None:

        return Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton, cellpose_masks, vollcellseg 
    
    elif noise_model is None and star_model is not None and  roi_model is  None and cellpose_model is not None and unet_membrane_model is not None:

        return Sizedsmart_seeds, SizedMask, SizedMembraneMask, star_labels, proabability_map, Markers, Skeleton, cellpose_masks, vollcellseg  
    

    # If denoising is done and stardist and unet models are supplied we return the stardist, vollseg, denoised image and semantic segmentation maps
    elif noise_model is not None and star_model is not None and  roi_model is not None and cellpose_model is None:
      
        return Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton,  image, roi_image
    
    elif noise_model is not None and star_model is not None and  roi_model is not None and cellpose_model is not None:
      
        return Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton,  image, roi_image, cellpose_masks, vollcellseg
    
    elif noise_model is not None and star_model is not None and  roi_model is not None and cellpose_model is not None and unet_membrane_model is not None:
      
        return Sizedsmart_seeds, SizedMask, SizedMembraneMask, star_labels, proabability_map, Markers, Skeleton,  image, roi_image, cellpose_masks, vollcellseg
    

    elif noise_model is not None and star_model is not None and  roi_model is None and cellpose_model is None:

        return Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton,  image
    
    elif noise_model is not None and star_model is not None and  roi_model is None and cellpose_model is not None:

        return Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton,  image, cellpose_masks, vollcellseg   
    
    elif noise_model is not None and star_model is not None and  roi_model is None and cellpose_model is not None and unet_membrane_model is not None:

        return Sizedsmart_seeds, SizedMask, SizedMembraneMask, star_labels, proabability_map, Markers, Skeleton,  image, cellpose_masks, vollcellseg
     

    # If the stardist model is not supplied but only the unet and noise model we return the denoised result and the semantic segmentation map
    elif star_model is None and  roi_model is not None and noise_model is not None and cellpose_model is None:

        return SizedMask, Skeleton, image

    elif star_model is None and  roi_model is not None and noise_model is None and cellpose_model is None:

        return  roi_image.astype('uint16')  , Skeleton, image 

    elif star_model is None and  roi_model is not None and noise_model is not None and cellpose_model is None:

        return  roi_image.astype('uint16')  , Skeleton, image     
    
    elif noise_model is not None and star_model is None and roi_model is None and unet_model is None and cellpose_model is None:
            
            return  SizedMask , Skeleton, image

    elif star_model is None and  roi_model is  None and noise_model is None and unet_model is not None and cellpose_model is None:

        return SizedMask, Skeleton, image    


def _cellpose_block(axes, flows, lower_perc, upper_perc, cellpose_masks, Sizedsmart_seeds, SizedMask, min_size_mask, max_size, nms_thresh, image_membrane):
    
    if 'T' not in axes:   
            cellpose_base = np.max(flows[0], axis = -1)
            cellpose_base = normalize(cellpose_base, lower_perc, upper_perc, axis= (0,1,2)) 
            vollcellseg = CellPoseWater(cellpose_base, cellpose_masks, Sizedsmart_seeds, SizedMask, min_size_mask, max_size,nms_thresh)
            
    if 'T' in axes:
                
            cellpose_base = []
            vollcellseg = []
            for time in range(image_membrane.shape[0]):
                cellpose_base_time = np.max(flows[0], axis = -1)[time]
                cellpose_masks_time = cellpose_masks[time]
                cellpose_base_time = normalize(cellpose_base_time, lower_perc, upper_perc, axis= (0,1,2))
               
                vollcellseg_time = CellPoseWater(cellpose_base_time, cellpose_masks_time, Sizedsmart_seeds[time], SizedMask, min_size_mask, max_size,nms_thresh)
                cellpose_base.append(cellpose_base_time)
                vollcellseg.append(vollcellseg_time)
            cellpose_base = np.asarray(cellpose_base)
            vollcellseg = np.asarray(vollcellseg_time) 
       
    return cellpose_base, vollcellseg        


def VollSeg(image,  unet_model=None, star_model=None, roi_model=None,  axes='ZYX', noise_model=None, prob_thresh=None, ExpandLabels = False, nms_thresh=None, min_size_mask=100, min_size=100, max_size=10000000,
            n_tiles=(1, 1, 1), UseProbability=True,  donormalize=True, lower_perc=1, upper_perc=99.8, dounet=True, seedpool=True, save_dir=None, Name='Result',  startZ=0, slice_merge=False, iou_threshold=0.3, RGB=False):

    if len(image.shape) == 2:

        # if the default tiling of the function is not changed by the user, we use the last two tuples
        if len(n_tiles) == 3:
            n_tiles = (n_tiles[1], n_tiles[2])

        # If stardist model is supplied we use this method
        if star_model is not None:

            res = VollSeg2D(image, unet_model, star_model, roi_model=roi_model, ExpandLabels= ExpandLabels, noise_model=noise_model, prob_thresh=prob_thresh, nms_thresh=nms_thresh,  donormalize=donormalize, lower_perc=lower_perc, upper_perc=upper_perc, axes=axes, min_size_mask=min_size_mask, min_size=min_size,
                            max_size=max_size, dounet=dounet, n_tiles=n_tiles, UseProbability=UseProbability, RGB=RGB)

        # If there is no stardist model we use unet model or denoising model or both to get the semantic segmentation
        if star_model is None:

            res = VollSeg_unet(image, unet_model=unet_model, roi_model=roi_model,ExpandLabels= ExpandLabels, n_tiles=n_tiles, axes=axes, min_size_mask=min_size_mask,
                               max_size=max_size,  noise_model=noise_model, RGB=RGB, iou_threshold=iou_threshold, slice_merge=slice_merge, dounet=dounet)
    if len(image.shape) == 3 and 'T' not in axes and RGB == False:
        # this is a 3D image and if stardist model is supplied we use this method
        if star_model is not None:
            res = VollSeg3D(image,  unet_model, star_model, roi_model=roi_model,ExpandLabels= ExpandLabels,  axes=axes, noise_model=noise_model, prob_thresh=prob_thresh, nms_thresh=nms_thresh, donormalize=donormalize, lower_perc=lower_perc, upper_perc=upper_perc, min_size_mask=min_size_mask, min_size=min_size, max_size=max_size,
                            n_tiles=n_tiles, UseProbability=UseProbability,  dounet=dounet, seedpool=seedpool, startZ=startZ, slice_merge=slice_merge, iou_threshold=iou_threshold)

        # If there is no stardist model we use unet model with or without denoising model
        if star_model is None:

            res = VollSeg_unet(image, unet_model=unet_model, roi_model=roi_model,ExpandLabels= ExpandLabels, n_tiles=n_tiles, axes=axes, min_size_mask=min_size_mask,
                               max_size=max_size,  noise_model=noise_model, RGB=RGB, iou_threshold=iou_threshold, slice_merge=slice_merge, dounet=dounet)
    if len(image.shape) == 3 and 'T' not in axes and RGB:
        # this is a 3D image and if stardist model is supplied we use this method
        if star_model is not None:
            res = VollSeg2D(image, unet_model, star_model, roi_model=roi_model,ExpandLabels= ExpandLabels, noise_model=noise_model, prob_thresh=prob_thresh, nms_thresh=nms_thresh,  donormalize=donormalize, lower_perc=lower_perc, upper_perc=upper_perc, axes=axes, min_size_mask=min_size_mask, min_size=min_size,
                            max_size=max_size, dounet=dounet, n_tiles=n_tiles, UseProbability=UseProbability, RGB=RGB)
        # If there is no stardist model we use unet model with or without denoising model
        if star_model is None:
           
            res = VollSeg_unet(image, unet_model=unet_model, roi_model=roi_model,ExpandLabels= ExpandLabels, n_tiles=n_tiles, axes=axes, min_size_mask=min_size_mask,
                               max_size=max_size,  noise_model=noise_model, RGB=RGB, iou_threshold=iou_threshold, slice_merge=slice_merge, dounet=dounet)
                 
    if len(image.shape) == 3 and 'T' in axes:
        if len(n_tiles) == 3:
            n_tiles = (n_tiles[1], n_tiles[2])
        if star_model is not None:
            res = tuple(
                zip(
                    *tuple(VollSeg2D(_x, unet_model, star_model, noise_model=noise_model,ExpandLabels= ExpandLabels, roi_model=roi_model,  prob_thresh=prob_thresh, nms_thresh=nms_thresh, donormalize=donormalize, lower_perc=lower_perc, upper_perc=upper_perc, axes=axes, min_size_mask=min_size_mask, min_size=min_size,
                                     max_size=max_size, dounet=dounet, n_tiles=n_tiles, UseProbability=UseProbability, RGB=RGB) for _x in tqdm(image))))
        if star_model is None:

            res = tuple(zip(*tuple(VollSeg_unet(_x, unet_model=unet_model, roi_model=roi_model,ExpandLabels= ExpandLabels, n_tiles=n_tiles, axes=axes, noise_model=noise_model, RGB=RGB, iou_threshold=iou_threshold, slice_merge=slice_merge, dounet=dounet)
                                   for _x in tqdm(image))))

    if len(image.shape) == 4:
        if len(n_tiles) == 4:
            n_tiles = (n_tiles[1], n_tiles[2], n_tiles[3])
        res = tuple(
            zip(
                *tuple(VollSeg3D(_x,  unet_model, star_model, axes=axes, noise_model=noise_model, roi_model=roi_model,ExpandLabels= ExpandLabels,  prob_thresh=prob_thresh, nms_thresh=nms_thresh, donormalize=donormalize, lower_perc=lower_perc, upper_perc=upper_perc, min_size_mask=min_size_mask, min_size=min_size, max_size=max_size,
                                 n_tiles=n_tiles, UseProbability=UseProbability,
                                 dounet=dounet, seedpool=seedpool, startZ=startZ, slice_merge=slice_merge, iou_threshold=iou_threshold) for _x in tqdm(image))))

 

    if noise_model is None and star_model is not None and  roi_model is not None:
        Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton, roi_image = res

    if noise_model is None and star_model is not None and  roi_model is None:
        Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton = res    

    elif noise_model is not None and star_model is not None and  roi_model is not None:
        Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton,  image, roi_image = res

    elif noise_model is not None and star_model is not None and  roi_model is None:
        Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton,  image = res    
        
    elif noise_model is not None and star_model is None and roi_model is None and unet_model is None:

         SizedMask, Skeleton, image = res
   
    
    elif star_model is None and  roi_model is None and unet_model is not None and noise_model is not None :

        SizedMask, Skeleton, image = res

    elif star_model is None and  roi_model is not None and unet_model is not None and noise_model is not None :

        SizedMask, Skeleton, image = res    

    elif star_model is None and  roi_model is None and unet_model is not None and noise_model is None :

        SizedMask, Skeleton, image = res    

    elif star_model is None and  roi_model is not None and unet_model is None and noise_model is None :

        roi_image, Skeleton, image = res
        SizedMask = roi_image

    elif star_model is None and  roi_model is not None and unet_model is None and noise_model is not None :

        roi_image, Skeleton, image = res
        SizedMask = roi_image    

    elif star_model is None and  roi_model is not None and unet_model is not None and noise_model is None :

        roi_image, Skeleton, image = res
        SizedMask = roi_image

    if save_dir is not None:
        print('Saving Results ...')
        Path(save_dir).mkdir(exist_ok=True)

        if  roi_model is not None:
            roi_results = save_dir + 'Roi/'
            Path(roi_results).mkdir(exist_ok=True)
            imwrite((roi_results + Name + '.tif'),
                    np.asarray(roi_image).astype('uint16'))

        if unet_model is not None:
            unet_results = save_dir + 'BinaryMask/'
            skel_unet_results = save_dir + 'Skeleton/'
            Path(unet_results).mkdir(exist_ok=True)
            Path(skel_unet_results).mkdir(exist_ok=True)
             
            imwrite((unet_results + Name + '.tif'),
                    np.asarray(SizedMask).astype('uint16'))
            imwrite((skel_unet_results + Name + '.tif'),
                    np.asarray(Skeleton).astype('uint16'))        
        if star_model is not None:
            vollseg_results = save_dir + 'VollSeg/'
            stardist_results = save_dir + 'StarDist/'
            probability_results = save_dir + 'Probability/'
            marker_results = save_dir + 'Markers/'
            skel_results = save_dir + 'Skeleton/'
            Path(skel_results).mkdir(exist_ok=True)
            Path(vollseg_results).mkdir(exist_ok=True)
            Path(stardist_results).mkdir(exist_ok=True)
            Path(probability_results).mkdir(exist_ok=True)
            Path(marker_results).mkdir(exist_ok=True)
            imwrite((stardist_results + Name + '.tif'),
                    np.asarray(star_labels).astype('uint16'))
            imwrite((vollseg_results + Name + '.tif'),
                    np.asarray(Sizedsmart_seeds).astype('uint16'))
            imwrite((probability_results + Name + '.tif'),
                    np.asarray(proabability_map).astype('float32'))
            imwrite((marker_results + Name + '.tif'),
                    np.asarray(Markers).astype('uint16'))
            imwrite((skel_results + Name + '.tif'), np.asarray(Skeleton))
        if noise_model is not None:
            denoised_results = save_dir + 'Denoised/'
            Path(denoised_results).mkdir(exist_ok=True)
            imwrite((denoised_results + Name + '.tif'),
                    np.asarray(image).astype('float32'))

      
    # If denoising is not done but stardist and unet models are supplied we return the stardist, vollseg and semantic segmentation maps
    if noise_model is None and star_model is not None and  roi_model is not None:

        return Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton, roi_image

    elif noise_model is None and star_model is not None and  roi_model is  None:

        return Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton    

    # If denoising is done and stardist and unet models are supplied we return the stardist, vollseg, denoised image and semantic segmentation maps
    elif noise_model is not None and star_model is not None and  roi_model is not None:
      
        return Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton,  image, roi_image

    elif noise_model is not None and star_model is not None and  roi_model is None:

        return Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton,  image    

    # If the stardist model is not supplied but only the unet and noise model we return the denoised result and the semantic segmentation map
    elif star_model is None and  roi_model is not None and noise_model is not None:

        return SizedMask, Skeleton, image

    elif star_model is None and  roi_model is not None and noise_model is None:

        return  roi_image.astype('uint16')  , Skeleton, image 

    elif star_model is None and  roi_model is not None and noise_model is not None:

        return  roi_image.astype('uint16')  , Skeleton, image     
    
    elif noise_model is not None and star_model is None and roi_model is None and unet_model is None:
            
            return  SizedMask , Skeleton, image
                  

    

    elif star_model is None and  roi_model is  None and noise_model is None and unet_model is not None:

        return SizedMask, Skeleton, image 



def VollSeg3D(image,  unet_model, star_model, axes='ZYX', noise_model=None, roi_model=None, prob_thresh=None, nms_thresh=None, min_size_mask=100, min_size=100, max_size=10000000,
              n_tiles=(1, 2, 2), image_membrane = None, unet_membrane_model = None, UseProbability=True, ExpandLabels = True, dounet=True, seedpool=True, donormalize=True, lower_perc=1, upper_perc=99.8, startZ=0, slice_merge=False, iou_threshold=0.3):

   

    print('Generating VollSeg results')
    sizeZ = image.shape[0]
    sizeY = image.shape[1]
    sizeX = image.shape[2]
    if len(n_tiles) >= len(image.shape):
                n_tiles = (n_tiles[-3],n_tiles[-2], n_tiles[-1])
    else:
                tiles = n_tiles
    SizedMask = np.zeros([sizeZ, sizeY, sizeX], dtype='uint16')
    if unet_membrane_model is not None:
        SizedMembraneMask = np.zeros([sizeZ, sizeY, sizeX], dtype='uint16')
    Sizedsmart_seeds = np.zeros([sizeZ, sizeY, sizeX], dtype='uint16')
    Sizedproabability_map = np.zeros([sizeZ, sizeY, sizeX], dtype='float32')
    Sizedmarkers = np.zeros([sizeZ, sizeY, sizeX], dtype='uint16')
    Sizedstardist = np.zeros([sizeZ, sizeY, sizeX], dtype='uint16')
    Mask = None
    Mask_patch = None
    roi_image = None
    if noise_model is not None:
        print('Denoising Image')

        image = noise_model.predict(image.astype('float32'), axes=axes, n_tiles=n_tiles)
        pixel_condition = (image < 0)
        pixel_replace_condition = 0
        image = image_conditionals(image,pixel_condition,pixel_replace_condition)
        

    if roi_model is not None:
        
        print('Creating overall mask for the tissue') 
        model_dim = roi_model.config.n_dim
        if model_dim < len(image.shape):
            if len(n_tiles) >= len(image.shape):
                tiles = (n_tiles[-2], n_tiles[-1])
            else:
                tiles = n_tiles
            maximage = np.amax(image, axis=0)
            roi_image = UNETPrediction3D(
                maximage, roi_model, tiles, 'YX', iou_threshold=nms_thresh, ExpandLabels = ExpandLabels)
            roi_bbox = Bbox_region(roi_image)
            if roi_bbox is not None:
                rowstart = roi_bbox[0]
                colstart = roi_bbox[1]
                endrow = roi_bbox[2]
                endcol = roi_bbox[3]
                region = (slice(0, image.shape[0]), slice(rowstart, endrow),
                        slice(colstart, endcol))
        elif model_dim == len(image.shape):
            roi_image = UNETPrediction3D(
                image, roi_model, n_tiles, axes, ExpandLabels = ExpandLabels)
          
            roi_bbox = Bbox_region(roi_image)
            if roi_bbox is not None:
                zstart = roi_bbox[0]
                rowstart = roi_bbox[1]
                colstart = roi_bbox[2]
                zend = roi_bbox[3]
                endrow = roi_bbox[4]
                endcol = roi_bbox[5]
                region = (slice(zstart, zend), slice(rowstart, endrow),
                        slice(colstart, endcol))
        # The actual pixels in that region.
        if roi_bbox is not None:
            patch = image[region]
            if image_membrane is not None:
                patch_membrane = image_membrane[region]
        else:
            patch = image  
            if image_membrane is not None:
                patch_membrane = image_membrane      

    else:

        patch = image
        if image_membrane is not None:
                patch_membrane = image_membrane
        
        region = (slice(0, image.shape[0]), slice(0, image.shape[1]),
                  slice(0, image.shape[2]))
        rowstart = 0
        colstart = 0
        endrow = image.shape[2]
        endcol = image.shape[1]
        roi_bbox = [colstart, rowstart, endcol, endrow]


    if unet_membrane_model is not None and image_membrane is not None:
            print('UNET membrane segmentation on Image',  patch_membrane.shape)

            MembraneMask = UNETPrediction3D(patch_membrane, unet_membrane_model, n_tiles, axes,
                                    iou_threshold=iou_threshold, slice_merge=slice_merge, ExpandLabels = ExpandLabels)
            print('done here')
            for i in tqdm(range(0, MembraneMask.shape[0])):
                   
                    MembraneMask[i] = remove_small_objects(
                            MembraneMask[i].astype('uint16'), min_size=min_size_mask)
                    MembraneMask[i] = remove_big_objects(
                            MembraneMask[i].astype('uint16'), max_size=max_size)
            MembraneMask_patch = MembraneMask.copy()
            print('embedding membrane')
            MembraneMask = Region_embedding(image, roi_bbox, MembraneMask)
            if slice_merge:
                MembraneMask = match_labels(MembraneMask.astype('uint16'), iou_threshold=iou_threshold)
            else:
                MembraneMask = label(MembraneMask > 0)
            SizedMembraneMask[:, :MembraneMask.shape[1], :MembraneMask.shape[2]] = MembraneMask

    if dounet:

        if unet_model is not None:
            print('UNET segmentation on Image',  patch.shape)

            Mask = UNETPrediction3D(patch, unet_model, n_tiles, axes,
                                    iou_threshold=iou_threshold, slice_merge=slice_merge, ExpandLabels = ExpandLabels)
            for i in range(0, Mask.shape[0]):
                    Mask[i] = remove_small_objects(
                            Mask[i].astype('uint16'), min_size=min_size_mask)
                    Mask[i] = remove_big_objects(
                            Mask[i].astype('uint16'), max_size=max_size)
            Mask_patch = Mask.copy()
            Mask = Region_embedding(image, roi_bbox, Mask)
            if slice_merge:
                Mask = match_labels(Mask.astype('uint16'), iou_threshold=iou_threshold)
            else:
                Mask = label(Mask > 0)
            SizedMask[:, :Mask.shape[1], :Mask.shape[2]] = Mask

    elif noise_model is not None and dounet == False:

        Mask = np.zeros(patch.shape)

        for i in range(0, Mask.shape[0]):

            thresholds = threshold_otsu(patch[i])
            Mask[i] = patch[i] > thresholds
            Mask[i] = label(Mask[i, :])

            Mask[i] = remove_small_objects(
                Mask[i].astype('uint16'), min_size=min_size_mask)
            Mask[i] = remove_big_objects(
                Mask[i].astype('uint16'), max_size=max_size)
        if slice_merge:
            Mask = match_labels(Mask, iou_threshold=iou_threshold)
        else:
            Mask = label(Mask > 0)
        Mask_patch = Mask.copy()    
        Mask = Region_embedding(image, roi_bbox, Mask)
        SizedMask[:, :Mask.shape[1], :Mask.shape[2]] = Mask
    if star_model is not None:
            print('Stardist segmentation on Image')
            if donormalize:
                
                patch_star = normalize(patch, lower_perc, upper_perc, axis= (0,1,2)) 
            else:
                patch_star = patch

            smart_seeds, proabability_map, star_labels, Markers = STARPrediction3D(
                patch_star, axes, star_model,  n_tiles, unet_mask=Mask_patch, UseProbability=UseProbability,seedpool=seedpool, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
            print('Removing small/large objects')
            for i in tqdm(range(0, smart_seeds.shape[0])):
                smart_seeds[i] = remove_small_objects(
                    smart_seeds[i, :].astype('uint16'), min_size=min_size)
                smart_seeds[i] = remove_big_objects(
                    smart_seeds[i, :].astype('uint16'), max_size=max_size)
            smart_seeds = fill_label_holes(smart_seeds.astype('uint16'))
            if startZ > 0:
                smart_seeds[0:startZ] = 0
            smart_seeds = Region_embedding(image, roi_bbox, smart_seeds)
            Sizedsmart_seeds[:, :smart_seeds.shape[1],
                            :smart_seeds.shape[2]] = smart_seeds
            Markers = Region_embedding(image, roi_bbox, Markers)
            Sizedmarkers[:, :smart_seeds.shape[1],
                        :smart_seeds.shape[2]] = Markers
            proabability_map = Region_embedding(image, roi_bbox, proabability_map)
            Sizedproabability_map[:, :proabability_map.shape[1],
                                :proabability_map.shape[2]] = proabability_map
            star_labels = Region_embedding(image, roi_bbox, star_labels)
            Sizedstardist[:, :star_labels.shape[1],
                        :star_labels.shape[2]] = star_labels
            Skeleton = np.zeros_like(Sizedsmart_seeds)
            for i in range(0, Sizedsmart_seeds.shape[0]):
                Skeleton[i] = SmartSkel(Sizedsmart_seeds[i],
                                        Sizedproabability_map[i])
            Skeleton = Skeleton > 0


    if noise_model is None and roi_image is not None and star_model is not None and unet_membrane_model is None:
        return Sizedsmart_seeds.astype('uint16'), SizedMask.astype('uint16'), star_labels.astype('uint16'), proabability_map, Markers.astype('uint16'), Skeleton.astype('uint16'), roi_image.astype('uint16')
    if noise_model is None and roi_image is None and star_model is not None and unet_membrane_model is None:
        return Sizedsmart_seeds.astype('uint16'), SizedMask.astype('uint16'), star_labels.astype('uint16'), proabability_map, Markers.astype('uint16'), Skeleton.astype('uint16')    
    if noise_model is not None and roi_image is None and star_model is not None and unet_membrane_model is None:
        return Sizedsmart_seeds.astype('uint16'), SizedMask.astype('uint16'), star_labels.astype('uint16'), proabability_map, Markers.astype('uint16'), Skeleton.astype('uint16'),  image
    if noise_model is not None and roi_image is not None and star_model is not None and unet_membrane_model is None:
        return Sizedsmart_seeds.astype('uint16'), SizedMask.astype('uint16'), star_labels.astype('uint16'), proabability_map, Markers.astype('uint16'), Skeleton.astype('uint16'),  image, roi_image.astype('uint16')
    
    if noise_model is not None and roi_image is not None and star_model is None and unet_membrane_model is None:
        return  SizedMask.astype('uint16'), Skeleton, image
    
    if noise_model is not None and roi_image is None and star_model is None and unet_model is None and unet_membrane_model is None:
         return SizedMask.astype('uint16'), Skeleton, image

    if noise_model is None and roi_image is None and star_model is None and unet_model is not None and unet_membrane_model is None:
         return SizedMask.astype('uint16'), Skeleton, image
     
    
    #With membrane model
    
    if noise_model is None and roi_image is not None and star_model is not None and unet_membrane_model is not None:
        return Sizedsmart_seeds.astype('uint16'), SizedMask.astype('uint16'), SizedMembraneMask.astype('uint16') ,star_labels.astype('uint16'), proabability_map, Markers.astype('uint16'), Skeleton.astype('uint16'), roi_image.astype('uint16')
    if noise_model is None and roi_image is None and star_model is not None and unet_membrane_model is not None:
        return Sizedsmart_seeds.astype('uint16'), SizedMask.astype('uint16'), SizedMembraneMask.astype('uint16') , star_labels.astype('uint16'), proabability_map, Markers.astype('uint16'), Skeleton.astype('uint16')    
    if noise_model is not None and roi_image is None and star_model is not None and unet_membrane_model is not None:
        return Sizedsmart_seeds.astype('uint16'), SizedMask.astype('uint16'), SizedMembraneMask.astype('uint16') , star_labels.astype('uint16'), proabability_map, Markers.astype('uint16'), Skeleton.astype('uint16'),  image
    if noise_model is not None and roi_image is not None and star_model is not None and unet_membrane_model is not None:
        return Sizedsmart_seeds.astype('uint16'), SizedMask.astype('uint16'),SizedMembraneMask.astype('uint16') , star_labels.astype('uint16'), proabability_map, Markers.astype('uint16'), Skeleton.astype('uint16'),  image, roi_image.astype('uint16')
    
    if noise_model is not None and roi_image is not None and star_model is None and unet_membrane_model is not None:
        return  SizedMask.astype('uint16'),SizedMembraneMask.astype('uint16') , Skeleton, image
    
    if noise_model is not None and roi_image is None and star_model is None and unet_model is None and unet_membrane_model is not None:
         return SizedMask.astype('uint16'),SizedMembraneMask.astype('uint16') , Skeleton, image

    if noise_model is None and roi_image is None and star_model is None and unet_model is not None and unet_membrane_model is not None:
         return SizedMask.astype('uint16'),SizedMembraneMask.astype('uint16') , Skeleton, image

    
     

           
def image_pixel_duplicator(image, size):

    assert len(image.shape) == len(size), f'The provided size {len(size)} should match the image dimensions {len(image.shape)}'
    
    ndim = len(size)



    if ndim == 3:
                    size_y = size[0]
                    size_x = size[1]
                    size_z = size[2]
                    if size_y <= image.shape[0]:
                        size_y =  image.shape[0]
                    if size_x <= image.shape[1]:
                        size_x =  image.shape[1]
                    if size_z <= image.shape[2]:
                        size_z =  image.shape[2]    

                    size = (size_y, size_x, size_z)
                    ResizeImage = np.zeros(size)
                    j = 0
                    for i in range(0, ResizeImage.shape[1]):
                        
                        if j < image.shape[1]:
                            ResizeImage[:image.shape[0],i,:image.shape[2]] = image[:image.shape[0],j,:image.shape[2]]
                            j = j + 1
                        else:
                            j = 0   
                        
                    j = 0
                    for i in range(0, ResizeImage.shape[2]):
                        
                        if j < image.shape[2]:
                            ResizeImage[:,:,i] = ResizeImage[:,:,j]
                            j = j + 1
                        else:
                            j = 0     

                    j = 0
                    for i in range(0, ResizeImage.shape[0]):
                        
                        if j < image.shape[0]:
                            ResizeImage[i,:,:] = ResizeImage[j,:,:]
                            j = j + 1
                        else:
                            j = 0  

                      

    if ndim == 2:


                    size_y = size[0]
                    size_x = size[1]
                    if size_y <= image.shape[0]:
                        size_y =  image.shape[0]
                    if size_x <= image.shape[1]:
                        size_x =  image.shape[1]
                      

                    size = (size_y, size_x)

                    ResizeImage = np.zeros(size)
                    j = 0
                    for i in range(0, ResizeImage.shape[1]):
                        
                        if j < image.shape[1]:
                            ResizeImage[:image.shape[0],i] = image[:image.shape[0],j]
                            j = j + 1
                        else:
                            j = 0   
                        

                    j = 0
                    for i in range(0, ResizeImage.shape[0]):
                        
                        if j < image.shape[0]:
                            ResizeImage[i,:] = ResizeImage[j,:]
                            j = j + 1
                        else:
                            j = 0  
           

              

    return ResizeImage


def image_conditionals(image, pixel_condition, pixel_replace_condition):

  
    indices = zip(*np.where(pixel_condition))
    for index in indices:
        
        image[index] = pixel_replace_condition
     
    return image

def image_embedding(image, size):

    
    ndim = len(image.shape)
    if ndim == 2:
        assert len(image.shape) == len(size), f'The provided size {len(size)} should match the image dimensions {len(image.shape)}'
        for i in range(len(size)):
          assert image.shape[i] <= size[i] , f'The image size should be smaller than the volume it is to be embedded in'
          width = []
          for i in range(len(size)):
                width.append(size[i] - image.shape[i])
          width = np.asarray(width)
    
          ResizeImage = np.pad(image, width, 'constant', constant_values = 0)
    if ndim == 3:
        ResizeImage = []
        width = []
        for i in range(len(size)):
                width.append(size[i] - image.shape[i + 1])
        width = np.asarray(width)
        for i in range(image.shape[0]):
             
           ResizeImage.append(np.pad(image[i,:], width, 'constant', constant_values = 0))   
        ResizeImage = np.asarray(ResizeImage)
    return ResizeImage


def Integer_to_border(Label):

    BoundaryLabel = find_boundaries(Label, mode='outer')

    Binary = BoundaryLabel > 0

    return Binary





def SuperUNETPrediction(image, model, n_tiles, axis):

    Segmented = model.predict(image.astype('float32'), axis, n_tiles=n_tiles)

    thresholds = threshold_otsu(Segmented)
    Binary = Segmented > thresholds
    Finalimage = label(Binary)

    Finalimage = relabel_sequential(Finalimage)[0]

    return Finalimage


def merge_labels_across_volume(labelvol, relabelfunc, threshold=3):
    nz, ny, nx = labelvol.shape
    res = np.zeros_like(labelvol)
    res[0, ...] = labelvol[0, ...]
    backup = labelvol.copy()  # kapoors code modifies the input array
    for i in tqdm(range(nz-1)):

        res[i+1, ...] = relabelfunc(res[i, ...],
                                    labelvol[i+1, ...], threshold=threshold)
        labelvol = backup.copy()  # restore the input array
    res = res.astype('uint16')
    return res


def RelabelZ(previousImage, currentImage, threshold):

    currentImage = currentImage.astype('uint16')
    relabelimage = currentImage
    previousImage = previousImage.astype('uint16')
    waterproperties = measure.regionprops(previousImage)
    indices = [prop.centroid for prop in waterproperties]
    if len(indices) > 0:
        tree = spatial.cKDTree(indices)
        currentwaterproperties = measure.regionprops(currentImage)
        currentindices = [prop.centroid for prop in currentwaterproperties]
        if len(currentindices) > 0:
            for i in range(0, len(currentindices)):
                index = currentindices[i]
                currentlabel = currentImage[int(index[0]), int(index[1])]
                if currentlabel > 0:
                    previouspoint = tree.query(index)
                    previouslabel = previousImage[int(indices[previouspoint[1]][0]), int(
                        indices[previouspoint[1]][1])]
                    if previouspoint[0] > threshold:
                        
                        pixel_condition = (currentImage == currentlabel)
                        pixel_replace_condition = currentlabel
                        relabelimage = image_conditionals(relabelimage,pixel_condition,pixel_replace_condition )

                    else:
                        pixel_condition = (currentImage == currentlabel)
                        pixel_replace_condition = previouslabel
                        relabelimage = image_conditionals(relabelimage,pixel_condition,pixel_replace_condition )

    return relabelimage




def CleanMask(star_labels, OverAllunet_mask):
    OverAllunet_mask = np.logical_or(OverAllunet_mask > 0, star_labels > 0)
    OverAllunet_mask = binary_erosion(OverAllunet_mask)
    OverAllunet_mask = label(OverAllunet_mask)
    OverAllunet_mask = fill_label_holes(OverAllunet_mask.astype('uint16'))

    return OverAllunet_mask


def UNETPrediction3D(image, model, n_tiles, axis, iou_threshold=0.3, slice_merge=False, erosion_iterations = 15, ExpandLabels = True):

    model_dim = model.config.n_dim
    
    if model_dim < len(image.shape):
        Segmented = np.zeros_like(image)
        
        for i in range(image.shape[0]):
            Segmented[i] = model.predict(image[i].astype('float32'), axis.replace('Z', ''), n_tiles= (n_tiles[-2], n_tiles[-1]))
                
    else:
        
        Segmented = model.predict(image.astype('float32'), axis, n_tiles=n_tiles)

    thresholds = threshold_otsu(Segmented)
    Binary = Segmented > thresholds
    overall_mask = Binary.copy()
    ndim = len(image.shape)
    if ndim == 3:
                for i in range(image.shape[0]):
                    overall_mask[i] = binary_dilation(overall_mask[i], iterations = erosion_iterations)
                    overall_mask[i] = binary_erosion(overall_mask[i], iterations = erosion_iterations)
                    overall_mask[i] = fill_label_holes(overall_mask[i])
    
    Binary = label(Binary)
    
        
    if ndim == 3 and slice_merge:
        for i in range(image.shape[0]):
            Binary[i] = label(Binary[i])
            Binary[i] = expand_labels(Binary[i], distance = GLOBAL_ERODE//2)
            Binary[i] = fill_label_holes(Binary[i])
        Binary = match_labels(Binary.astype('uint16'),
                              iou_threshold=iou_threshold)
    if ndim == 3 and model_dim < len(image.shape):    
         for i in range(image.shape[0]):
            Binary[i] = label(Binary[i])
            Binary[i] = expand_labels(Binary[i], distance = GLOBAL_ERODE//2)
            Binary[i] = fill_label_holes(Binary[i])
            Binary = match_labels(Binary.astype('uint16'),
                              iou_threshold=iou_threshold)
    # Postprocessing steps
    Finalimage = fill_label_holes(Binary)
    Finalimage = relabel_sequential(Finalimage)[0]
    if ExpandLabels:
        Finalimage = VollSeg_label_precondition(image, overall_mask, Finalimage)
  
    

    return Finalimage


def Bbox_region(image):

    props = measure.regionprops(image)
    area = [prop.area for prop in props]
    if len(area) > 0:
            largest_blob_ind = np.argmax(area)
            largest_bbox = props[largest_blob_ind].bbox
            return largest_bbox
           



def SuperSTARPrediction(image, model, n_tiles, unet_mask=None, OverAllunet_mask=None, UseProbability=True, 
prob_thresh=None, nms_thresh=None, seedpool = True):

    if prob_thresh is None and nms_thresh is None:
        prob_thresh = model.thresholds.prob
        nms_thresh = model.thresholds.nms 
    if prob_thresh is not None and nms_thresh is not None:

        star_labels,  SmallProbability, SmallDistance = model.predict_vollseg(
            image.astype('float32'), n_tiles=n_tiles, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
    else:
        star_labels,  SmallProbability, SmallDistance = model.predict_vollseg(
            image.astype('float32'), n_tiles=n_tiles)


    grid = model.config.grid
    Probability = zoom(SmallProbability, zoom=(
        grid[1], grid[0]))
    Distance = MaxProjectDist(SmallDistance, axis=-1)
    Distance = zoom(Distance, zoom=(
        grid[1], grid[0]))

    pixel_condition = (Probability < GLOBAL_THRESH)
    pixel_replace_condition = 0
    Probability = image_conditionals(Probability, pixel_condition, pixel_replace_condition) 

    if UseProbability:

        MaxProjectDistance = Probability[:star_labels.shape[0],:star_labels.shape[1]] 

    else:

        MaxProjectDistance = Distance[:star_labels.shape[0],:star_labels.shape[1]]

    if OverAllunet_mask is None:
        OverAllunet_mask = unet_mask
    if OverAllunet_mask is not None:    
       OverAllunet_mask = CleanMask(star_labels, OverAllunet_mask)

    if unet_mask is None:
        unet_mask = star_labels > 0
    Watershed, Markers = SuperWatershedwithMask(
        MaxProjectDistance, star_labels.astype('uint16'), unet_mask.astype('uint16'), nms_thresh = nms_thresh, seedpool = seedpool)
    Watershed = fill_label_holes(Watershed.astype('uint16'))

    return Watershed, Markers, star_labels, MaxProjectDistance


def STARPrediction3D(image, axes, model, n_tiles, unet_mask=None,  UseProbability=True,  seedpool=True, prob_thresh=None, nms_thresh=None):

    copymodel = model
    
    grid = copymodel.config.grid
    print('Predicting Instances')
    if prob_thresh is None and nms_thresh is None:
        prob_thresh = model.thresholds.prob
        nms_thresh = model.thresholds.nms 
    if prob_thresh is not None and nms_thresh is not None:

        print(f'Using user choice of prob_thresh = {prob_thresh} and nms_thresh = {nms_thresh}')
        res = model.predict_vollseg(
            image.astype('float32'), axes = axes, n_tiles=n_tiles, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
    else:
        res = model.predict_vollseg(image.astype('float32'), axes = axes, n_tiles=n_tiles)
    star_labels, SmallProbability, SmallDistance = res
    print('Predictions Done')
    
    if UseProbability == False:

        SmallDistance = MaxProjectDist(SmallDistance, axis=-1)
        Distance = np.zeros([SmallDistance.shape[0] * grid[0],
                            SmallDistance.shape[1] * grid[1], SmallDistance.shape[2] * grid[2]])

    Probability = np.zeros([SmallProbability.shape[0] * grid[0],
                           SmallProbability.shape[1] * grid[1], SmallProbability.shape[2] * grid[2]])

    # We only allow for the grid parameter to be 1 along the Z axis
    for i in range(0, SmallProbability.shape[0]):
        Probability[i, :] = zoom(SmallProbability[i, :], zoom=(
            grid[2], grid[1]))
        if UseProbability == False:
            Distance[i, :] = zoom(SmallDistance[i, :], zoom=(
                grid[2], grid[1]))

    if UseProbability:

        print('Using Probability maps')
        MaxProjectDistance = Probability[:star_labels.shape[0],:star_labels.shape[1],:star_labels.shape[2]]

    else:

        print('Using Distance maps')
        MaxProjectDistance = Distance[:star_labels.shape[0],:star_labels.shape[1],:star_labels.shape[2]]

    print('Doing Watershedding')
 
    if unet_mask is None:
         unet_mask = star_labels > 0

    Watershed, Markers = WatershedwithMask3D(MaxProjectDistance.astype(
        'uint16'), star_labels.astype('uint16'), unet_mask.astype('uint16'), nms_thresh = nms_thresh, seedpool = seedpool)
    Watershed = fill_label_holes(Watershed.astype('uint16'))

    return Watershed, MaxProjectDistance, star_labels, Markers





def CellPoseWater(Image, Masks, Seeds, membrane_mask, min_size, max_size,nms_thresh):
    
   
    CopyMasks = Masks.copy()
    properties = measure.regionprops(CopyMasks)
    starproperties = measure.regionprops(Seeds)
    
    bbox = [prop.bbox for prop in properties]
    Coordinates = [prop.centroid for prop in starproperties]
    KeepCoordinates = []
    if len(Coordinates) > 0:
            for i in range(0, len(Coordinates)):

                star = Coordinates[i]
                value=CopyMasks[int(star[0]),int(star[1]),int(star[2])]

                if value==0:
                    KeepCoordinates.append(Coordinates[i])
                    
                    
    KeepCoordinates.append((0, 0, 0))
    KeepCoordinates = np.asarray(KeepCoordinates)

    coordinates_int = np.round(KeepCoordinates).astype(int)
    markers_raw = np.zeros_like(Image)
    markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(KeepCoordinates))

    markers = morphology.dilation(
        markers_raw.astype('uint16'), morphology.ball(2))                
    watershed_image = watershed(-Image, markers, mask = membrane_mask)
    watershed_image = fill_label_holes(watershed_image)
    
    empy_region_indices = zip(*np.where(CopyMasks == 0))
    watershed_image =  NMSLabel(image= watershed_image, nms_thresh=nms_thresh * 2).supresslabels()
    watershed_image =  NMSLabel(image= watershed_image, nms_thresh=nms_thresh * 2).supressregions()
    
    for index in empy_region_indices:
        
        CopyMasks[index] = watershed_image[index]
        
    for i in range(CopyMasks.shape[0]):
       CopyMasks[i] = remove_small_objects(CopyMasks[i], min_size = min_size) 
       CopyMasks[i] = remove_big_objects(CopyMasks[i], max_size = max_size)
    
    return CopyMasks

def SuperWatershedwithMask(Image, Label, mask, nms_thresh, seedpool):

    properties = measure.regionprops(Label)
    Coordinates = [prop.centroid for prop in properties]
    binaryproperties = measure.regionprops(label(mask), Image)
    BinaryCoordinates = [prop.centroid for prop in binaryproperties]
    Binarybbox = [prop.bbox for prop in binaryproperties]
    
    Starbbox = [prop.bbox for prop in properties]
    Starlabel = [prop.label for prop in properties]
    if len(Starbbox) > 0:
        for i in range(0, len(Starbbox)):

            box = Starbbox[i]
            starlabel = Starlabel[i]
            include = [UnetStarMask(box, unet).masking() for unet in BinaryCoordinates] 
            if False not in include:
                indices = zip(*np.where(Label == starlabel))
                for index in indices:
        
                      mask[index] = 1

    binaryproperties = measure.regionprops(label(mask))
    BinaryCoordinates = [prop.centroid for prop in binaryproperties]
    Binarybbox = [prop.bbox for prop in binaryproperties]
    if seedpool:
        if len(Binarybbox) > 0:
            for i in range(0, len(Binarybbox)):

                box = Binarybbox[i]
                include = [SeedPool(box, star).pooling() for star in Coordinates]

                if False not in include:
                    Coordinates.append(BinaryCoordinates[i])
    Coordinates.append((0, 0))
    Coordinates = np.asarray(Coordinates)

    coordinates_int = np.round(Coordinates).astype(int)
    markers_raw = np.zeros_like(Image)
    markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))

    markers = morphology.dilation(markers_raw, morphology.disk(2))
    watershedImage = watershed(-Image, markers, mask=mask.copy())
    watershedImage =  NMSLabel(image= watershedImage, nms_thresh=nms_thresh).supresslabels()
    #watershedImage =  NMSLabel(image= watershedImage, nms_thresh=nms_thresh).supressregions()
    return watershedImage, markers


def WatershedwithMask3D(Image, Label, mask, nms_thresh, seedpool=True):
    properties = measure.regionprops(Label)
    

    Coordinates = [prop.centroid for prop in properties]
    binaryproperties = measure.regionprops(label(mask))
    BinaryCoordinates = [prop.centroid for prop in binaryproperties]
    Binarybbox = [prop.bbox for prop in binaryproperties]
    Coordinates = sorted(Coordinates, key=lambda k: [k[0], k[1], k[2]])
    Starbbox = [prop.bbox for prop in properties]
    Starlabel = [prop.label for prop in properties]
    if len(Starbbox) > 0:
        for i in range(0, len(Starbbox)):

            box = Starbbox[i]
            starlabel = Starlabel[i]
            include = [UnetStarMask(box, unet).masking() for unet in BinaryCoordinates] 
            if False not in include:
                indices = zip(*np.where(Label == starlabel))
                for index in indices:
                      mask[index] = 1
    binaryproperties = measure.regionprops(label(mask))
    BinaryCoordinates = [prop.centroid for prop in binaryproperties]
    Binarybbox = [prop.bbox for prop in binaryproperties]

    if seedpool:


        if len(Binarybbox) > 0:
            for i in range(0, len(Binarybbox)):

                box = Binarybbox[i]
                include = [SeedPool(box, star).pooling() for star in Coordinates]

                if False not in include:
                    Coordinates.append(BinaryCoordinates[i])

    Coordinates.append((0, 0, 0))

    Coordinates = np.asarray(Coordinates)
    coordinates_int = np.round(Coordinates).astype(int)

    markers_raw = np.zeros_like(Image)
    markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))
    markers = morphology.dilation(
        markers_raw.astype('uint16'), morphology.ball(2))
    watershedImage = watershed(-Image, markers, mask=mask.copy())
    watershedImage =  NMSLabel(image= watershedImage, nms_thresh=nms_thresh).supresslabels()
    #watershedImage =  NMSLabel(image= watershedImage, nms_thresh=nms_thresh).supressregions()
    return watershedImage, markers


def MaxProjectDist(Image, axis=-1):

    MaxProject = np.amax(Image, axis=axis)

    return MaxProject


def MidProjectDist(Image, axis=-1, slices=1):

    assert len(Image.shape) >= 3
    SmallImage = Image.take(indices=range(
        Image.shape[axis]//2 - slices, Image.shape[axis]//2 + slices), axis=axis)

    MaxProject = np.amax(SmallImage, axis=axis)
    return MaxProject







def normalizeFloatZeroOne(x, pmin=3, pmax=99.8, axis=None, eps=1e-20, dtype=np.float32):
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
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalizer(x, mi, ma, eps=eps, dtype=dtype)

# https://docs.python.org/3/library/itertools.html#itertools-recipes


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


def normalizer(x, mi, ma, eps=1e-20, dtype=np.float32):
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
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

        x = (x - mi) / (ma - mi + eps)

        x = normalizeZeroOne(x)
    return x




   # CARE csbdeep modification of implemented function


def normalizeFloat(x, pmin=3, pmax=99.8, axis=None, eps=1e-20, dtype=np.float32):
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
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, eps=1e-20, dtype=np.float32):
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
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    x = (x - mi) / (ma - mi + eps)

    return x




def plot_train_history(history, savedir, modelname, *keys,**kwargs):
    """Plot (Keras) training history returned by :func:`CARE.train`."""
    import matplotlib.pyplot as plt

    logy = kwargs.pop('logy',False)

    if all(( isinstance(k,string_types) for k in keys )):
        w, keys = 1, [keys]
    else:
        w = len(keys)

    plt.gcf()
    for i, group in enumerate(keys):
        plt.subplot(1,w,i+1)
        for k in ([group] if isinstance(group,string_types) else group):
            plt.plot(history.epoch,history.history[k],'.-',label=k,**kwargs)
            if logy:
                plt.gca().set_yscale('log', nonposy='clip')
        plt.xlabel('epoch')
        plt.legend(loc='best')
    plt.savefig(savedir + '/' + modelname + 'train_accuracy' + '.png', dpi = 600)    