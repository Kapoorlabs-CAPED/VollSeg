#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:08:41 2019
@author: vkapoor
"""

from __future__ import print_function, unicode_literals, absolute_import, division
# import matplotlib.pyplot as plt
import numpy as np
import os
import collections
from tifffile import imread, imwrite
from skimage import morphology
from skimage.morphology import dilation, square
import cv2
from skimage.morphology import remove_small_objects, remove_small_holes
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
from skimage.segmentation import relabel_sequential
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.morphology import binary_dilation, binary_erosion
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
from vollseg.matching import matching
from skimage.measure import regionprops
from qtpy.QtWidgets import QComboBox, QPushButton
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


def SimplePrediction(x, UnetModel, StarModel, n_tiles=(2, 2), UseProbability=True, min_size=20, axis='ZYX', globalthreshold=1.0E-5):

                      Mask = UNETPrediction3D(x, UnetModel, n_tiles, axis)

                      smart_seeds, _, star_labels, _ = STARPrediction3D(
                          x, StarModel, n_tiles, unet_mask=Mask, smartcorrection=None, UseProbability=UseProbability, globalthreshold=globalthreshold)

                      smart_seeds = smart_seeds.astype('uint16')

                      return smart_seeds


def crappify_flou_G_P(x, y, lam, savedirx, savediry, name):
    x = x.astype('float32')
    gaussiannoise = np.random.poisson(lam, x.shape)
    x = x + gaussiannoise

    # add noise to original image
    imwrite(savedirx + '/' + name + 'pg' +
            str(lam) + '.tif', x.astype('float32'))
    # keep the label the same
    imwrite(savediry + '/' + name + 'pg' +
            str(lam) + '.tif', y.astype('uint16'))


def _fill_label_holes(lbl_img, **kwargs):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in (set(np.unique(lbl_img)) - set([0])):
        mask = lbl_img == l
        mask_filled = binary_fill_holes(mask, **kwargs)
        lbl_img_filled[mask_filled] = l
    return lbl_img_filled


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
        if sl is None: continue
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


def match_labels(ys, iou_threshold=0):
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

    ys = np.asarray(ys)
    if not ys.ndim in (3, 4):
        raise ValueError('label image y should be 3 or 4 dimensional!')

    def _match_single(x, y):
        res = matching(x, y, report_matches=True, thresh=0)

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

    for i in range(len(ys)-1):
       ys_new[i+1] = _match_single(ys_new[i], ys[i+1])

    return ys_new
    # ys_new = merge_labels_across_volume(ys, RelabelZ, threshold = iou_threshold)

    return ys_new


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


def multiplotline(plotA, plotB, plotC, titleA, titleB, titleC, targetdir=None, File=None, plotTitle=None):
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
    else:
      Title = 'MultiPlot'
    if targetdir is not None and File is not None:
      plt.savefig(targetdir + Title + File + '.png')
    if targetdir is not None and File is None:
      plt.savefig(targetdir + Title + File + '.png')
    plt.show()


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


def NotumQueen(save_dir, fname, denoising_model, projection_model, mask_model, star_model, min_size=5, n_tiles=(1, 1, 1), UseProbability=True):

    Path(save_dir).mkdir(exist_ok=True)
    projection_results = save_dir + 'Projected/'
    Path(projection_results).mkdir(exist_ok=True)
    Name = os.path.basename(os.path.splitext(fname)[0])
    print('Denoising Image')
    image = imread(fname)
    image = denoising_model.predict(image, 'ZYX', n_tiles=n_tiles)
    image = projection_model.predict(
        image, 'YX', n_tiles=(n_tiles[1], n_tiles[2]))

    imwrite((projection_results + Name + '.tif'), image.astype('float32'))

    NotumSegmentation2D(save_dir, image, fname, mask_model, star_model, min_size=min_size, n_tiles=(
        n_tiles[1], n_tiles[2]), UseProbability=UseProbability)


def NotumSeededQueen(save_dir, fname, denoising_model, projection_model, unet_model, mask_model, star_model, min_size=5, n_tiles=(1, 1, 1), UseProbability=True):

    Path(save_dir).mkdir(exist_ok=True)
    projection_results = save_dir + 'Projected/'
    Path(projection_results).mkdir(exist_ok=True)
    Name = os.path.basename(os.path.splitext(fname)[0])
    print('Denoising Image')
    image = imread(fname)
    image = denoising_model.predict(image, 'ZYX', n_tiles=n_tiles)
    image = projection_model.predict(
        image, 'YX', n_tiles=(n_tiles[1], n_tiles[2]))

    imwrite((projection_results + Name + '.tif'), image.astype('float32'))

    SeededNotumSegmentation2D(save_dir, image, fname, unet_model, mask_model, star_model,
                              min_size=min_size, n_tiles=(n_tiles[1], n_tiles[2]), UseProbability=UseProbability)


def SeededNotumSegmentation2D(SaveDir, image, fname, UnetModel, MaskModel, StarModel, min_size=5, n_tiles=(2, 2), UseProbability=True):

    print('Generating SmartSeed results')

    MASKResults = SaveDir + 'OverAllMask/'
    unet_results = SaveDir + 'UnetMask/'
    star_labelsResults = SaveDir + 'StarDistMask/'
    smart_seedsResults = SaveDir + 'smart_seedsMask/'
    smart_seedsLabelResults = SaveDir + 'smart_seedsLabels/'
    ProbResults = SaveDir + 'Probability/'

    Path(SaveDir).mkdir(exist_ok=True)
    Path(smart_seedsResults).mkdir(exist_ok=True)
    Path(star_labelsResults).mkdir(exist_ok=True)
    Path(unet_results).mkdir(exist_ok=True)
    Path(MASKResults).mkdir(exist_ok=True)
    Path(smart_seedsLabelResults).mkdir(exist_ok=True)
    Path(ProbResults).mkdir(exist_ok=True)
    # Read Image
    Name = os.path.basename(os.path.splitext(fname)[0])

    # U-net prediction

    OverAllMask = SuperUNETPrediction(image, MaskModel, n_tiles, 'YX')
    Mask = SuperUNETPrediction(image, UnetModel, n_tiles, 'YX')

    # Smart Seed prediction
    smart_seeds, Markers, star_labels, ProbImage = SuperSTARPrediction(
        image, StarModel, n_tiles, unet_mask=Mask, OverAllunet_mask=OverAllMask, UseProbability=UseProbability)

    smart_seedsLabels = smart_seeds.copy()

    # For avoiding pixel level error
    OverAllMask = CleanMask(star_labels, OverAllMask)
    smart_seedsLabels = np.multiply(smart_seedsLabels, OverAllMask)
    SegimageB = find_boundaries(smart_seedsLabels)
    invertProbimage = 1 - ProbImage
    image_max = np.add(invertProbimage, SegimageB)
    indices = np.where(image_max < 1.2)
    image_max[indices] = 0
    smart_seeds = np.array(dip.UpperSkeleton2D(image_max.astype('float32')))

    # Save results, we only need smart seeds finale results but hey!
    imwrite((ProbResults + Name + '.tif'), ProbImage.astype('float32'))
    imwrite((smart_seedsResults + Name + '.tif'), smart_seeds.astype('uint8'))
    imwrite((smart_seedsLabelResults + Name + '.tif'),
            smart_seedsLabels.astype('uint16'))
    imwrite((star_labelsResults + Name + '.tif'), star_labels.astype('uint16'))
    imwrite((unet_results + Name + '.tif'), Mask.astype('uint8'))
    imwrite((MASKResults + Name + '.tif'), OverAllMask.astype('uint8'))


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


def NotumKing(save_dir, filesRaw, denoising_model, projection_model, unet_model, star_model, n_tiles=(1, 1, 1), UseProbability=True, dounet=True, seedpool=True, min_size_mask=100, min_size=5, max_size=10000000):

    Path(save_dir).mkdir(exist_ok=True)
    projection_results = save_dir + 'Projected/'
    Path(projection_results).mkdir(exist_ok=True)

    print('Denoising Image')
    time_lapse = []
    for fname in filesRaw:
            image = imread(fname)
            image = denoising_model.predict(image, 'ZYX', n_tiles=n_tiles)
            image = projection_model.predict(
                image, 'YX', n_tiles=(n_tiles[1], n_tiles[2]))
            time_lapse.append(image)
    Name = os.path.basename(os.path.splitext(fname)[0])
    time_lapse = np.asarray(time_lapse)
    imwrite((projection_results + Name + '.tif'), time_lapse.astype('float32'))

    Raw_path = os.path.join(projection_results, '*tif')
    filesRaw = glob.glob(Raw_path)
    for fname in filesRaw:
         image = imread(fname)
         Name = os.path.basename(os.path.splittext(fname)[0])
         VollSeg3D(image,  unet_model, star_model,  min_size_mask=min_size_mask, min_size=min_size, max_size=max_size, n_tiles=n_tiles,
                   UseProbability=UseProbability, dounet=dounet, seedpool=seedpool, axes='ZYX', save_dir=save_dir, Name=Name)


def NotumSegmentation2D(save_dir, image, fname, mask_model, star_model, min_size=5, n_tiles=(2, 2), UseProbability=True):

    print('Generating SmartSeed results')
    Path(save_dir).mkdir(exist_ok=True)
    MASKResults = save_dir + 'OverAllMask/'
    star_labelsResults = save_dir + 'StarDistMask/'
    smart_seedsResults = save_dir + 'smart_seedsMask/'
    smart_seedsLabelResults = save_dir + 'smart_seedsLabels/'
    ProbResults = save_dir + 'Probability/'

    Path(smart_seedsResults).mkdir(exist_ok=True)
    Path(star_labelsResults).mkdir(exist_ok=True)

    Path(MASKResults).mkdir(exist_ok=True)
    Path(smart_seedsLabelResults).mkdir(exist_ok=True)
    Path(ProbResults).mkdir(exist_ok=True)
    # Read Image
    Name = os.path.basename(os.path.splitext(fname)[0])

    # U-net prediction

    OverAllMask = SuperUNETPrediction(image, mask_model, n_tiles, 'YX')

    # Smart Seed prediction
    smart_seeds, Markers, star_labels, ProbImage = SuperSTARPrediction(
        image, star_model, n_tiles, unet_mask=OverAllMask, OverAllunet_mask=OverAllMask, UseProbability=UseProbability)

    smart_seedsLabels = smart_seeds.copy()

    # For avoiding pixel level error
    OverAllMask = CleanMask(star_labels, OverAllMask)
    smart_seedsLabels = np.multiply(smart_seedsLabels, OverAllMask)
    SegimageB = find_boundaries(smart_seedsLabels)
    invertProbimage = 1 - ProbImage
    image_max = np.add(invertProbimage, SegimageB)
    indices = np.where(image_max < 1.2)
    image_max[indices] = 0
    smart_seeds = np.array(dip.UpperSkeleton2D(image_max.astype('float32')))

    # Save results, we only need smart seeds finale results but hey!
    imwrite((ProbResults + Name + '.tif'), ProbImage.astype('float32'))
    imwrite((smart_seedsResults + Name + '.tif'), smart_seeds.astype('uint8'))
    imwrite((smart_seedsLabelResults + Name + '.tif'),
            smart_seedsLabels.astype('uint16'))
    imwrite((star_labelsResults + Name + '.tif'), star_labels.astype('uint16'))

    imwrite((MASKResults + Name + '.tif'), OverAllMask.astype('uint8'))


def SmartSkel(smart_seedsLabels, ProbImage):

    SegimageB = find_boundaries(smart_seedsLabels)
    invertProbimage = 1 - ProbImage
    image_max = np.add(invertProbimage, SegimageB)
    indices = np.where(image_max < 1.2)
    image_max[indices] = 0
    Skeleton = np.array(dip.UpperSkeleton2D(image_max.astype('float32')))

    return Skeleton





def SuperWatershedwithMask(Image, Label, mask, grid):

    properties = measure.regionprops(Label, Image)
    binaryproperties = measure.regionprops(label(mask), Image)

    Coordinates = [prop.centroid for prop in properties]
    BinaryCoordinates = [prop.centroid for prop in binaryproperties]
    Binarybbox = [prop.bbox for prop in binaryproperties]

    if len(Binarybbox) > 0:
            for i in range(0, len(Binarybbox)):

                box = Binarybbox[i]
                inside = [iouNotum(box, star) for star in Coordinates]

                if not any(inside):
                         Coordinates.append(BinaryCoordinates[i])
    Coordinates = sorted(Coordinates, key=lambda k: [k[1], k[0]])
    Coordinates.append((0, 0))
    Coordinates = np.asarray(Coordinates)

    coordinates_int = np.round(Coordinates).astype(int)
    markers_raw = np.zeros_like(Image)
    markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))

    markers = morphology.dilation(markers_raw, morphology.disk(2))
    watershedImage = watershed(-Image, markers, mask=mask.copy())

    return watershedImage, markers
# If there are neighbouring seeds we do not put more seeds


def ConditioncheckNotum(centroid, boxA, p, ndim):

      condition = False

      if centroid[p] >= boxA[p] and centroid[p] <= boxA[p + ndim]:

           condition = True

      return condition


def iouNotum(boxA, centroid):

    ndim = len(centroid)
    inside = False

    Condition = [ConditioncheckNotum(centroid, boxA, p, ndim)
                                     for p in range(0, ndim)]

    inside = all(Condition)

    return inside


def SuperWatershedwithoutMask(Image, Label, mask, grid):

    properties = measure.regionprops(Label, Image)
    binaryproperties = measure.regionprops(label(mask), Image)

    Coordinates = [prop.centroid for prop in properties]
    BinaryCoordinates = [prop.centroid for prop in binaryproperties]
    Binarybbox = [prop.bbox for prop in binaryproperties]

    if len(Binarybbox) > 0:
            for i in range(0, len(Binarybbox)):

                box = Binarybbox[i]
                inside = [iouNotum(box, star) for star in Coordinates]

                if not any(inside):
                         Coordinates.append(BinaryCoordinates[i])
    Coordinates = sorted(Coordinates, key=lambda k: [k[1], k[0]])
    Coordinates.append((0, 0))
    Coordinates = np.asarray(Coordinates)

    coordinates_int = np.round(Coordinates).astype(int)
    markers_raw = np.zeros_like(Image)
    markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))

    markers = morphology.dilation(markers_raw, morphology.disk(2))
    watershedImage = watershed(-Image, markers)

    return watershedImage, markers


def VollSeg2D(image, unet_model, star_model, noise_model=None, prob_thresh=None, nms_thresh=None, axes='YX', min_size_mask=5, min_size=5,
              max_size=10000000, dounet=True, n_tiles=(2, 2), UseProbability=True, RGB=False, save_dir=None, Name='Result'):

    print('Generating SmartSeed results')

    if save_dir is not None:

        unet_results = save_dir + 'BinaryMask/'
        vollseg_results = save_dir + 'VollSeg/'
        stardist_results = save_dir + 'StarDist/'
        denoised_results = save_dir + 'Denoised/'
        probability_results = save_dir + 'Probability/'
        marker_results = save_dir + 'Markers/'
        skel_results = save_dir + 'Skeleton/'
        Path(save_dir).mkdir(exist_ok=True)
        Path(skel_results).mkdir(exist_ok=True)
        Path(denoised_results).mkdir(exist_ok=True)
        Path(vollseg_results).mkdir(exist_ok=True)
        Path(stardist_results).mkdir(exist_ok=True)
        Path(unet_results).mkdir(exist_ok=True)
        Path(probability_results).mkdir(exist_ok=True)
        Path(marker_results).mkdir(exist_ok=True)

    if star_model is not None:
        nms_thresh = star_model.thresholds[1]
    elif nms_thresh is not None:
        nms_thresh = nms_thresh
    else:
        nms_thresh = 0

    if RGB:
        axes = 'YXC'
    if noise_model is not None:
         print('Denoising Image')

         image = noise_model.predict(image, axes=axes, n_tiles=n_tiles)
         if save_dir is not None:
             imwrite((denoised_results + Name + '.tif'),
                     image.astype('float32'))
    if dounet:

        if unet_model is not None:
            print('UNET segmentation on Image')

            Mask = UNETPrediction3D(
                image, unet_model, n_tiles, axes, iou_threshold=nms_thresh)
            Mask = remove_small_objects(
                Mask.astype('uint16'), min_size=min_size_mask)
            Mask = remove_big_objects(Mask.astype('uint16'), max_size=max_size)

        else:

                Mask = np.zeros(image.shape)
                try:
                    thresh = threshold_otsu(image)
                except:
                    thresh = 0
                Mask = image > thresh
                Mask = label(Mask)
                Mask = remove_small_objects(
                    Mask.astype('uint16'), min_size=min_size_mask)
                Mask = remove_big_objects(Mask.astype('uint16'), max_size=max_size)
    else:

          Mask = np.zeros(image.shape)
          try:
            thresh = threshold_otsu(image)
          except:
            thresh = 0
          Mask = image > thresh
          Mask = label(Mask)
          Mask = remove_small_objects(
              Mask.astype('uint16'), min_size=min_size_mask)
          Mask = remove_big_objects(Mask.astype('uint16'), max_size=max_size)

    if save_dir is not None:
             imwrite((unet_results + Name + '.tif'), Mask.astype('uint16'))
    # Smart Seed prediction
    print('Stardist segmentation on Image')
    if RGB:
        Mask = Mask[:, :, 0]
    smart_seeds, Markers, star_labels, proabability_map = SuperSTARPrediction(
        image, star_model, n_tiles, unet_mask=Mask, UseProbability=UseProbability, prob_thresh=prob_thresh, nms_thresh=nms_thresh, RGB=RGB)

    Skeleton = SmartSkel(smart_seeds, proabability_map)
    Skeleton = Skeleton > 0
    # For avoiding pixel level error
    Mask = expand_labels(Mask, distance=1)
    smart_seeds = expand_labels(smart_seeds, distance=1)

    if save_dir is not None:
        print('Saving Results and Done')
        imwrite((stardist_results + Name + '.tif'),
                star_labels.astype('uint16'))
        imwrite((vollseg_results + Name + '.tif'),
                smart_seeds.astype('uint16'))
        imwrite((probability_results + Name + '.tif'),
                proabability_map.astype('float32'))
        imwrite((marker_results + Name + '.tif'), Markers.astype('uint16'))
        imwrite((skel_results + Name + '.tif'), Skeleton.astype('uint16'))
    if noise_model == None:
        return smart_seeds, Mask, star_labels, proabability_map, Markers, Skeleton
    else:
        return smart_seeds, Mask, star_labels, proabability_map, Markers, Skeleton, image

def VollSeg_unet(image, unet_model = None, n_tiles=(2, 2), axes='YX', noise_model=None, RGB=False, iou_threshold=0, slice_merge=False, dounet = True):

    if RGB:
        if n_tiles is not None:
          n_tiles = (n_tiles[0], n_tiles[1], 1)

    if noise_model is not None:
        image = noise_model.predict(image, axes, n_tiles=n_tiles)
        
    if dounet and unet_model is not None:
        Segmented = unet_model.predict(image, axes, n_tiles=n_tiles)
    else:
        Segmented = image
    if RGB:
        Segmented = Segmented[:, :, 0]
    try:
       thresh = threshold_otsu(Segmented)
       Binary = Segmented > thresh
    except:
        Binary = Segmented > 0

    ndim = len(image.shape)
    Binary = label(Binary)
    if ndim == 3 and slice_merge:
        for i in range(image.shape[0]):
            Binary[i, :] = label(Binary[i, :])
        Binary = match_labels(Binary, iou_threshold=iou_threshold)

    Finalimage = relabel_sequential(Binary)[0]

    return Finalimage, image


def VollSeg(image,  unet_model = None, star_model = None, axes='ZYX', noise_model=None, prob_thresh=None, nms_thresh=None, min_size_mask=100, min_size=100, max_size=10000000,
n_tiles=(1, 1, 1), UseProbability=True, globalthreshold=1.0E-5, extent=0, dounet=True, seedpool=True, save_dir=None, Name='Result',  startZ=0, slice_merge=False, iou_threshold=0, RGB = False):

     if len(image.shape) == 2:
         
         #if the default tiling of the function is not changed by the user, we use the last two tuples
         if len(n_tiles) == 3:
             n_tiles = (n_tiles[1], n_tiles[2])
             
         # If stardist model is supplied we use this method    
         if star_model is not None:
             
             res = VollSeg2D(image, unet_model, star_model, noise_model=noise_model, prob_thresh=prob_thresh, nms_thresh=nms_thresh, axes=axes, min_size_mask=min_size_mask, min_size=min_size,
                       max_size=max_size, dounet=dounet, n_tiles = n_tiles, UseProbability=UseProbability, RGB=RGB, save_dir=None, Name=Name)
         
         # If there is no stardist model we use unet model or denoising model or both to get the semantic segmentation    
         if star_model is None:
             
               res = VollSeg_unet(image, unet_model = unet_model, n_tiles=n_tiles, axes=axes, noise_model=noise_model, RGB=RGB, iou_threshold=iou_threshold, slice_merge=slice_merge, dounet = dounet)
     if len(image.shape) == 3 and 'T' not in axes:
          # this is a 3D image and if stardist model is supplied we use this method
          if star_model is not None:   
                  res = VollSeg3D(image,  unet_model, star_model, axes=axes, noise_model=noise_model, prob_thresh=prob_thresh, nms_thresh=nms_thresh, min_size_mask=min_size_mask, min_size=min_size, max_size=max_size,
                  n_tiles=n_tiles, UseProbability=UseProbability, globalthreshold=globalthreshold, extent=extent, dounet=dounet, seedpool=seedpool, save_dir=None, Name=Name,  startZ=startZ, slice_merge=slice_merge, iou_threshold=iou_threshold)
          
          # If there is no stardist model we use unet model with or without denoising model
          if star_model is None:
              
               res = VollSeg_unet(image, unet_model, n_tiles=n_tiles, axes=axes, noise_model=noise_model, RGB=RGB, iou_threshold=iou_threshold, slice_merge=slice_merge, dounet = dounet)
             
                
     if len(image.shape) == 3 and 'T' in axes:
              if len(n_tiles) == 3:
                  n_tiles = (n_tiles[1], n_tiles[2])
              if star_model is not None:
                  res = tuple(
                      zip(
                          *tuple(VollSeg2D(_x, unet_model, star_model, noise_model=noise_model, prob_thresh=prob_thresh, nms_thresh=nms_thresh, axes=axes, min_size_mask=min_size_mask, min_size=min_size,
                                        max_size=max_size, dounet=dounet, n_tiles=n_tiles, UseProbability=UseProbability, RGB=RGB, save_dir=None, Name=Name) for _x in tqdm(image))))
              if star_model is None:
                  
                   res = tuple(zip(*tuple(VollSeg_unet(_x, unet_model, n_tiles=n_tiles, axes=axes, noise_model=noise_model, RGB=RGB, iou_threshold=iou_threshold, slice_merge=slice_merge, dounet = dounet)
                  for _x in tqdm(image))))
                   
                      


     if len(image.shape) == 4:
           if len(n_tiles) == 4:
               n_tiles = (n_tiles[1], n_tiles[2], n_tiles[3])
           res = tuple(
               zip(
                   *tuple(VollSeg3D(_x,  unet_model, star_model, axes=axes, noise_model=noise_model, prob_thresh=prob_thresh, nms_thresh=nms_thresh, min_size_mask=min_size_mask, min_size=min_size, max_size=max_size,
                   n_tiles=n_tiles, UseProbability=UseProbability, globalthreshold=globalthreshold, extent=extent, 
                   dounet=dounet, seedpool=seedpool, save_dir=None, Name=Name,  startZ=startZ, slice_merge=slice_merge, iou_threshold=iou_threshold) for _x in tqdm(image))))

     if noise_model is None and star_model is not None:
         Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton=res
     
     elif noise_model is not None and star_model is not None:
         Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton,  image=res
         
     elif star_model is None:
         
          SizedMask, image = res
         
     if save_dir is not None:
        print('Saving Results ...')
        Path(save_dir).mkdir(exist_ok=True)
        
        if unet_model is not None:
                unet_results=save_dir + 'BinaryMask/'
                Path(unet_results).mkdir(exist_ok=True)
                
                imwrite((unet_results + Name + '.tif'),
                        SizedMask.astype('uint16'))
        if star_model is not None:
            vollseg_results=save_dir + 'VollSeg/'
            stardist_results=save_dir + 'StarDist/'
            probability_results=save_dir + 'Probability/'
            marker_results=save_dir + 'Markers/'
            skel_results=save_dir + 'Skeleton/'
            Path(skel_results).mkdir(exist_ok=True)
            Path(vollseg_results).mkdir(exist_ok=True)
            Path(stardist_results).mkdir(exist_ok=True)
            Path(probability_results).mkdir(exist_ok=True)
            Path(marker_results).mkdir(exist_ok=True)
            imwrite((stardist_results + Name + '.tif'),
                    star_labels.astype('uint16'))
            imwrite((vollseg_results + Name + '.tif'),
                    Sizedsmart_seeds.astype('uint16'))
            imwrite((probability_results + Name + '.tif'),
                    proabability_map.astype('float32'))
            imwrite((marker_results + Name + '.tif'),
                    Markers.astype('uint16'))
            imwrite((skel_results + Name + '.tif'), Skeleton)
        if noise_model is not None:    
            denoised_results=save_dir + 'Denoised/'
            Path(denoised_results).mkdir(exist_ok=True)
            imwrite((denoised_results + Name + '.tif'),
                    image.astype('float32'))
        
        print('Done')
     #If denoising is not done but stardist and unet models are supplied we return the stardist, vollseg and semantic segmentation maps
     if noise_model is None and star_model is not None:
         
         return Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton
     
     #If denoising is done and stardist and unet models are supplied we return the stardist, vollseg, denoised image and semantic segmentation maps   
     elif noise_model is not None and star_model is not None:
         return Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton,  image
     
     #If the stardist model is not supplied but only the unet and noise model we return the denoised result and the semantic segmentation map 
     elif star_model is None:
         
          return SizedMask, image
     
def VollSeg3D(image,  unet_model, star_model, axes='ZYX', noise_model=None, prob_thresh=None, nms_thresh=None, min_size_mask=100, min_size=100, max_size=10000000,
n_tiles=(1, 2, 2), UseProbability=True, globalthreshold=1.0E-5, extent=0, dounet=True, seedpool=True, save_dir=None, Name='Result',  startZ=0, slice_merge=False, iou_threshold=0):






    if save_dir is not None:

        unet_results=save_dir + 'BinaryMask/'
        vollseg_results=save_dir + 'VollSeg/'
        stardist_results=save_dir + 'StarDist/'
        denoised_results=save_dir + 'Denoised/'
        probability_results=save_dir + 'Probability/'
        marker_results=save_dir + 'Markers/'
        skel_results=save_dir + 'Skeleton/'
        Path(save_dir).mkdir(exist_ok=True)
        Path(skel_results).mkdir(exist_ok=True)
        Path(denoised_results).mkdir(exist_ok=True)
        Path(vollseg_results).mkdir(exist_ok=True)
        Path(stardist_results).mkdir(exist_ok=True)
        Path(unet_results).mkdir(exist_ok=True)
        Path(probability_results).mkdir(exist_ok=True)
        Path(marker_results).mkdir(exist_ok=True)


    print('Generating VollSeg results')
    sizeZ=image.shape[0]
    sizeY=image.shape[1]
    sizeX=image.shape[2]

    SizedMask=np.zeros([sizeZ, sizeY, sizeX], dtype='uint16')
    Sizedsmart_seeds=np.zeros([sizeZ, sizeY, sizeX], dtype='uint16')
    Sizedproabability_map=np.zeros([sizeZ, sizeY, sizeX], dtype='float32')

    if noise_model is not None:
         print('Denoising Image')

         image=noise_model.predict(image, axes=axes, n_tiles=n_tiles)
         if save_dir is not None:
             imwrite((denoised_results + Name + '.tif'),
                     image.astype('float32'))

    if dounet:

        if unet_model is not None:
            print('UNET segmentation on Image')

            Mask=UNETPrediction3D(image, unet_model, n_tiles, 'ZYX',
                                  iou_threshold=iou_threshold, slice_merge=slice_merge)

            for i in range(0, Mask.shape[0]):
                    Mask[i, :]=remove_small_objects(
                        Mask[i, :].astype('uint16'), min_size=min_size_mask)
                    Mask[i, :]=remove_big_objects(
                        Mask[i, :].astype('uint16'), max_size=max_size)

            if slice_merge:
                Mask=match_labels(Mask, iou_threshold=iou_threshold)
            else:
                Mask=label(Mask > 0)
            SizedMask[:, :Mask.shape[1], :Mask.shape[2]]=Mask
    else:

          Mask=np.zeros(image.shape)

          for i in range(0, Mask.shape[0]):

                     thresh=threshold_otsu(image[i, :])
                     Mask[i, :]=image[i, :] > thresh
                     Mask[i, :]=label(Mask[i, :])

                     Mask[i, :]=remove_small_objects(
                         Mask[i, :].astype('uint16'), min_size=min_size_mask)
                     Mask[i, :]=remove_big_objects(
                         Mask[i, :].astype('uint16'), max_size=max_size)
          if slice_merge:
              Mask=match_labels(Mask, iou_threshold=iou_threshold)
          else:
              Mask=label(Mask > 0)
          SizedMask[:, :Mask.shape[1], :Mask.shape[2]]=Mask
    if save_dir is not None:
             imwrite((unet_results + Name + '.tif'),
                     SizedMask.astype('uint16'))
    print('Stardist segmentation on Image')

    smart_seeds, proabability_map, star_labels, Markers=STARPrediction3D(
        image, star_model,  n_tiles, unet_mask=Mask, UseProbability=UseProbability, globalthreshold=globalthreshold, extent=extent, seedpool=seedpool, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
    print('Removing small/large objects')
    for i in tqdm(range(0, smart_seeds.shape[0])):
       smart_seeds[i, :]=remove_small_objects(
           smart_seeds[i, :].astype('uint16'), min_size=min_size)
       smart_seeds[i, :]=remove_big_objects(
           smart_seeds[i, :].astype('uint16'), max_size=max_size)
    smart_seeds=fill_label_holes(smart_seeds.astype('uint16'))
    if startZ > 0:
         smart_seeds[0:startZ, :, :]=0

    Sizedsmart_seeds[:, :smart_seeds.shape[1],
        :smart_seeds.shape[2]]=smart_seeds
    Sizedproabability_map[:, :proabability_map.shape[1],
        :proabability_map.shape[2]]=proabability_map

    Skeleton=np.zeros_like(Sizedsmart_seeds)
    for i in range(0, Sizedsmart_seeds.shape[0]):
       Skeleton[i, :]=SmartSkel(Sizedsmart_seeds[i, :],
                                Sizedproabability_map[i, :])
    Skeleton=Skeleton > 0
    if save_dir is not None:
        print('Saving Results and Done')
        imwrite((stardist_results + Name + '.tif'),
                star_labels.astype('uint16'))
        imwrite((vollseg_results + Name + '.tif'),
                Sizedsmart_seeds.astype('uint16'))
        imwrite((probability_results + Name + '.tif'),
                proabability_map.astype('float32'))
        imwrite((marker_results + Name + '.tif'), Markers.astype('uint16'))
        imwrite((skel_results + Name + '.tif'), Skeleton)
    if noise_model == None:
        return Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton
    else:
        return Sizedsmart_seeds, SizedMask, star_labels, proabability_map, Markers, Skeleton,  image

def Integer_to_border(Label):

        BoundaryLabel=find_boundaries(Label, mode='outer')

        Binary=BoundaryLabel > 0

        return Binary


def DownsampleData(image, DownsampleFactor):

                if DownsampleFactor != 1:
                    print('Downsampling Image in XY by', DownsampleFactor)
                    # percent of original size
                    scale_percent=int(100/DownsampleFactor)
                    width=int(image.shape[2] * scale_percent / 100)
                    height=int(image.shape[1] * scale_percent / 100)
                    dim=(width, height)
                    smallimage=np.zeros([image.shape[0],  height, width])
                    for i in range(0, image.shape[0]):
                          # resize image
                          smallimage[i, :]=cv2.resize(
                              image[i, :].astype('float32'), dim)

                    return smallimage
                else:

                    return image






def SuperUNETPrediction(image, model, n_tiles, axis, threshold=20):


    Segmented=model.predict(image, axis, n_tiles=n_tiles)

    try:
       thresh=threshold_otsu(Segmented)
       Binary=Segmented > thresh
    except:
        Binary=Segmented > 0


    Finalimage=label(Binary)


    Finalimage=relabel_sequential(Finalimage)[0]

    return Finalimage


def merge_labels_across_volume(labelvol, relabelfunc, threshold=3):
    nz, ny, nx=labelvol.shape
    res=np.zeros_like(labelvol)
    res[0, ...]=labelvol[0, ...]
    backup=labelvol.copy()  # kapoors code modifies the input array
    for i in tqdm(range(nz-1)):

        res[i+1, ...]=relabelfunc(res[i, ...],
                                  labelvol[i+1, ...], threshold=threshold)
        labelvol=backup.copy()  # restore the input array
    res=res.astype('uint16')
    return res

def RelabelZ(previousImage, currentImage, threshold):

    currentImage=currentImage.astype('uint16')
    relabelimage=currentImage
    previousImage=previousImage.astype('uint16')
    waterproperties=measure.regionprops(previousImage)
    indices=[prop.centroid for prop in waterproperties]
    if len(indices) > 0:
       tree=spatial.cKDTree(indices)
       currentwaterproperties=measure.regionprops(currentImage)
       currentindices=[prop.centroid for prop in currentwaterproperties]
       if len(currentindices) > 0:
           for i in range(0, len(currentindices)):
               index=currentindices[i]
               currentlabel=currentImage[int(index[0]), int(index[1])]
               if currentlabel > 0:
                      previouspoint=tree.query(index)
                      previouslabel=previousImage[int(indices[previouspoint[1]][0]), int(
                          indices[previouspoint[1]][1])]
                      if previouspoint[0] > threshold:
                             relabelimage[np.where(
                                 currentImage == currentlabel)]=currentlabel
                      else:
                             relabelimage[np.where(
                                 currentImage == currentlabel)]=previouslabel
    return relabelimage

def SuperSTARPrediction(image, model, n_tiles, unet_mask, OverAllunet_mask=None, UseProbability=True, prob_thresh=None, nms_thresh=None, RGB=False):


    image=normalize(image, 1, 99.8, axis=(0, 1))

    shape=[image.shape[0], image.shape[1]]



    if prob_thresh is not None and nms_thresh is not None:

        MidImage,  SmallProbability, SmallDistance=model.predict_vollseg(
            image, n_tiles=n_tiles, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
    else:
        MidImage,  SmallProbability, SmallDistance=model.predict_vollseg(
            image, n_tiles=n_tiles)

    star_labels=MidImage[:shape[0], :shape[1]]


    grid=model.config.grid
    Probability=cv2.resize(SmallProbability, dsize=(
        SmallProbability.shape[1] * grid[1], SmallProbability.shape[0] * grid[0]))
    Distance=MaxProjectDist(SmallDistance, axis=-1)
    Distance=cv2.resize(Distance, dsize=(
        Distance.shape[1] * grid[1], Distance.shape[0] * grid[0]))
    if UseProbability:

        MaxProjectDistance=Probability[:shape[0], :shape[1]]

    else:

        MaxProjectDistance=Distance[:shape[0], :shape[1]]




    if OverAllunet_mask is None:
        OverAllunet_mask=unet_mask
    OverAllunet_mask=CleanMask(star_labels, OverAllunet_mask)


    Watershed, Markers=SuperWatershedwithMask(
        MaxProjectDistance, star_labels.astype('uint16'), unet_mask.astype('uint16'), grid)
    Watershed=fill_label_holes(Watershed.astype('uint16'))


    return Watershed, Markers, star_labels, MaxProjectDistance

def CleanMask(star_labels, OverAllunet_mask):
    OverAllunet_mask=np.logical_or(OverAllunet_mask > 0, star_labels > 0)
    OverAllunet_mask=binary_erosion(OverAllunet_mask)
    OverAllunet_mask=label(OverAllunet_mask)
    OverAllunet_mask=fill_label_holes(OverAllunet_mask.astype('uint16'))

    return OverAllunet_mask

def UNETPrediction3D(image, model, n_tiles, axis, iou_threshold=0, slice_merge=False):


    Segmented=model.predict(image, axis, n_tiles=n_tiles)



    try:
       thresh=threshold_mean(Segmented)
       Binary=Segmented > thresh
    except:
        Binary=Segmented > 0
    Binary=label(Binary)
    ndim=len(image.shape)
    if ndim == 3 and slice_merge:
        for i in range(image.shape[0]):
            Binary[i, :]=label(Binary[i, :])
        Binary=match_labels(Binary, iou_threshold=iou_threshold)

    # Postprocessing steps
    Finalimage=fill_label_holes(Binary)
    Finalimage=relabel_sequential(Finalimage)[0]


    return Finalimage

def RemoveLabels(LabelImage, minZ=2):

    properties=measure.regionprops(LabelImage, LabelImage)
    for prop in properties:
                regionlabel=prop.label
                sizeZ=abs(prop.bbox[0] - prop.bbox[3])
                if sizeZ <= minZ:
                    LabelImage[LabelImage == regionlabel]=0
    return LabelImage

def STARPrediction3D(image, model, n_tiles, unet_mask=None, smartcorrection=None, UseProbability=True, globalthreshold=1.0E-5, extent=0, seedpool=True, prob_thresh=None, nms_thresh=None):

    copymodel=model
    image=normalize(image, 1, 99.8, axis=(0, 1, 2))
    shape=[image.shape[1], image.shape[2]]
    image=zero_pad_time(image, 64, 64)
    grid=copymodel.config.grid

    print('Predicting Instances')
    if prob_thresh is not None and nms_thresh is not None:
       res=model.predict_vollseg(
           image, n_tiles=n_tiles, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
    else:
       res=model.predict_vollseg(image, n_tiles=n_tiles)
    MidImage, SmallProbability, SmallDistance=res
    print('Predictions Done')
    star_labels=MidImage[:image.shape[0], :shape[0], :shape[1]]

    star_labels=RemoveLabels(star_labels)
    if UseProbability == False:

        SmallDistance=MaxProjectDist(SmallDistance, axis=-1)
        Distance=np.zeros([SmallDistance.shape[0] * grid[0],
                          SmallDistance.shape[1] * grid[1], SmallDistance.shape[2] * grid[2]])

    Probability=np.zeros([SmallProbability.shape[0] * grid[0],
                         SmallProbability.shape[1] * grid[1], SmallProbability.shape[2] * grid[2]])

    # We only allow for the grid parameter to be 1 along the Z axis
    for i in range(0, SmallProbability.shape[0]):
        Probability[i, :]=cv2.resize(SmallProbability[i, :], dsize=(
            SmallProbability.shape[2] * grid[2], SmallProbability.shape[1] * grid[1]))
        if UseProbability == False:
            Distance[i, :]=cv2.resize(SmallDistance[i, :], dsize=(
                SmallDistance.shape[2] * grid[2], SmallDistance.shape[1] * grid[1]))

    if UseProbability:

        print('Using Probability maps')
        Probability[Probability < globalthreshold]=0

        MaxProjectDistance=Probability[:image.shape[0], :shape[0], :shape[1]]

    else:

        print('Using Distance maps')
        MaxProjectDistance=Distance[:image.shape[0], :shape[0], :shape[1]]


    print('Doing Watershedding')
    Watershed, Markers=WatershedwithMask3D(MaxProjectDistance.astype(
        'uint16'), star_labels.astype('uint16'), unet_mask.astype('uint16'), grid, extent, seedpool)
    Watershed=fill_label_holes(Watershed.astype('uint16'))




    return Watershed, MaxProjectDistance, star_labels, Markers


def VetoRegions(Image, Zratio=3):

    Image=Image.astype('uint16')

    properties=measure.regionprops(Image, Image)

    for prop in properties:

        LabelImage=prop.image
        if LabelImage.shape[0] < Image.shape[0]/Zratio:
            indices=zip(*np.where(LabelImage > 0))
            for z, y, x in indices:

                 Image[z, y, x]=0

    return Image


# Default method that works well with cells which are below a certain shape and do not have weak edges

def iou3D(boxA, centroid, extent=0):

    ndim=len(centroid)
    inside=False

    Condition=[Conditioncheck(centroid, boxA, p, ndim, extent)
                              for p in range(0, ndim)]

    inside=all(Condition)

    return inside

def Conditioncheck(centroid, boxA, p, ndim, extent):

      condition=False

      vol=extent * (boxA[p + ndim] - boxA[p]) / 2

      if centroid[p] >= boxA[p] - vol and centroid[p] <= boxA[p + ndim] + vol:

           condition=True

      return condition


def WatershedwithMask3D(Image, Label, mask, grid, extent=0, seedpool=True):
    properties=measure.regionprops(Label, Image)
    binaryproperties=measure.regionprops(label(mask), Image)

    Coordinates=[prop.centroid for prop in properties]
    BinaryCoordinates=[prop.centroid for prop in binaryproperties]

    Binarybbox=[prop.bbox for prop in binaryproperties]
    Coordinates=sorted(Coordinates, key=lambda k: [k[0], k[1], k[2]])

    if seedpool:
      if len(Binarybbox) > 0:
            for i in range(0, len(Binarybbox)):

                box=Binarybbox[i]
                inside=[iou3D(box, star, extent) for star in Coordinates]

                if not any(inside):
                         Coordinates.append(BinaryCoordinates[i])


    Coordinates.append((0, 0, 0))


    Coordinates=np.asarray(Coordinates)
    coordinates_int=np.round(Coordinates).astype(int)

    markers_raw=np.zeros_like(Image)
    markers_raw[tuple(coordinates_int.T)]=1 + np.arange(len(Coordinates))
    markers=morphology.dilation(
        markers_raw.astype('uint16'), morphology.ball(2))
    watershedImage=watershed(-Image, markers, mask=mask.copy())


    return watershedImage, markers





def zero_pad(image, PadX, PadY):

          sizeY=image.shape[1]
          sizeX=image.shape[0]

          sizeXextend=sizeX
          sizeYextend=sizeY


          while sizeXextend % PadX != 0:
              sizeXextend=sizeXextend + 1

          while sizeYextend % PadY != 0:
              sizeYextend=sizeYextend + 1

          extendimage=np.zeros([sizeXextend, sizeYextend])

          extendimage[0:sizeX, 0:sizeY]=image


          return extendimage


def zero_pad_color(image, PadX, PadY):

          sizeY=image.shape[1]
          sizeX=image.shape[0]
          color=image.shape[2]

          sizeXextend=sizeX
          sizeYextend=sizeY


          while sizeXextend % PadX != 0:
              sizeXextend=sizeXextend + 1

          while sizeYextend % PadY != 0:
              sizeYextend=sizeYextend + 1

          extendimage=np.zeros([sizeXextend, sizeYextend, color])

          extendimage[0:sizeX, 0:sizeY, 0:color]=image


          return extendimage

def zero_pad_time(image, PadX, PadY):

          sizeY=image.shape[2]
          sizeX=image.shape[1]

          sizeXextend=sizeX
          sizeYextend=sizeY


          while sizeXextend % PadX != 0:
              sizeXextend=sizeXextend + 1

          while sizeYextend % PadY != 0:
              sizeYextend=sizeYextend + 1

          extendimage=np.zeros([image.shape[0], sizeXextend, sizeYextend])

          extendimage[:, 0:sizeX, 0:sizeY]=image


          return extendimage

def BackGroundCorrection2D(Image, sigma):


     Blur=gaussian(Image.astype(float), sigma)


     Corrected=Image - Blur

     return Corrected



def MaxProjectDist(Image, axis=-1):

    MaxProject=np.amax(Image, axis=axis)

    return MaxProject

def MidProjectDist(Image, axis=-1, slices=1):

    assert len(Image.shape) >= 3
    SmallImage=Image.take(indices=range(
        Image.shape[axis]//2 - slices, Image.shape[axis]//2 + slices), axis=axis)

    MaxProject=np.amax(SmallImage, axis=axis)
    return MaxProject


def multiplot(imageA, imageB, imageC, titleA, titleB, titleC, targetdir=None, File=None, plotTitle=None):
    fig, axes=plt.subplots(1, 3, figsize=(15, 6))
    ax=axes.ravel()
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

def doubleplot(imageA, imageB, titleA, titleB, targetdir=None, File=None, plotTitle=None):
    fig, axes=plt.subplots(1, 2, figsize=(15, 6))
    ax=axes.ravel()
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
    mi=np.percentile(x, pmin, axis=axis, keepdims=True)
    ma=np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalizer(x, mi, ma, eps=eps, dtype=dtype)

# https://docs.python.org/3/library/itertools.html#itertools-recipes
def move_image_axes(x, fr, to, adjust_singletons=False):
    """
    x: ndarray
    fr,to: axes string (see `axes_dict`)
    """
    fr=axes_check_and_normalize(fr, length=x.ndim)
    to=axes_check_and_normalize(to)

    fr_initial=fr
    x_shape_initial=x.shape
    adjust_singletons=bool(adjust_singletons)
    if adjust_singletons:
        # remove axes not present in 'to'
        slices=[slice(None) for _ in x.shape]
        for i, a in enumerate(fr):
            if (a not in to) and (x.shape[i] == 1):
                # remove singleton axis
                slices[i]=0
                fr=fr.replace(a, '')
        x=x[slices]
        # add dummy axes present in 'to'
        for i, a in enumerate(to):
            if (a not in fr):
                # add singleton axis
                x=np.expand_dims(x, -1)
                fr += a

    if set(fr) != set(to):
        _adjusted='(adjusted to %s and %s) ' % (
            x.shape, fr) if adjust_singletons else ''
        raise ValueError(
            'image with shape %s and axes %s %snot compatible with target axes %s.'
            % (x_shape_initial, fr_initial, _adjusted, to)
        )

    ax_from, ax_to=axes_dict(fr), axes_dict(to)
    if fr == to:
        return x
    return np.moveaxis(x, [ax_from[a] for a in fr], [ax_to[a] for a in fr])
def consume(iterator):
    collections.deque(iterator, maxlen=0)

def _raise(e):
    raise e
def compose(*funcs):
    return lambda x: reduce(lambda f, g: g(f), funcs, x)

def normalizeZeroOne(x):

     x=x.astype('float32')

     minVal=np.min(x)
     maxVal=np.max(x)

     x=((x-minVal) / (maxVal - minVal + 1.0e-20))

     return x

def normalizeZero255(x):

     x=x.astype('float32')

     minVal=np.min(x)
     maxVal=np.max(x)

     x=((x-minVal) / (maxVal - minVal + 1.0e-20))

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
        x=x.astype(dtype, copy=False)
        mi=dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma=dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps=dtype(eps)

    try:
        import numexpr
        x=numexpr.evaluate("(x - mi ) / (ma - mi + eps)")
    except ImportError:
        x=(x - mi) / (ma - mi + eps)

        x=normalizeZeroOne(x)
    return x


def LocalThreshold2D(Image, boxsize, offset=0, size=10):

    if boxsize % 2 == 0:
        boxsize=boxsize + 1
    adaptive_thresh=threshold_local(Image, boxsize, offset=offset)
    Binary=Image > adaptive_thresh
    # Clean =  remove_small_objects(Binary, min_size=size, connectivity=4, in_place=False)

    return Binary



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
    mi=np.percentile(x, pmin, axis=axis, keepdims=True)
    ma=np.percentile(x, pmax, axis=axis, keepdims=True)
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
        x=x.astype(dtype, copy=False)
        mi=dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma=dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps=dtype(eps)

    try:
        import numexpr
        x=numexpr.evaluate("(x - mi ) / (ma - mi + eps)")
    except ImportError:
        x=(x - mi) / (ma - mi + eps)


    return x







def backend_channels_last():
    import keras.backend as K
    assert K.image_data_format() in ('channels_first', 'channels_last')
    return K.image_data_format() == 'channels_last'


def move_channel_for_backend(X, channel):
    if backend_channels_last():
        return np.moveaxis(X, channel, -1)
    else:
        return np.moveaxis(X, channel,  1)


def axes_check_and_normalize(axes, length=None, disallowed=None, return_allowed=False):
    """
    S(ample), T(ime), C(hannel), Z, Y, X
    """
    allowed='STCZYX'
    axes=str(axes).upper()
    consume(a in allowed or _raise(ValueError(
        "invalid axis '%s', must be one of %s." % (a, list(allowed)))) for a in axes)
    disallowed is None or consume(a not in disallowed or _raise(
        ValueError("disallowed axis '%s'." % a)) for a in axes)
    consume(axes.count(a) == 1 or _raise(ValueError(
        "axis '%s' occurs more than once." % a)) for a in axes)
    length is None or len(axes) == length or _raise(
        ValueError('axes (%s) must be of length %d.' % (axes, length)))
    return (axes, allowed) if return_allowed else axes
def axes_dict(axes):
    """
    from axes string to dict
    """
    axes, allowed=axes_check_and_normalize(axes, return_allowed=True)
    return {a: None if axes.find(a) == -1 else axes.find(a) for a in allowed}
    # return collections.namedt
