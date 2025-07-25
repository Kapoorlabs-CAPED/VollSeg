#!/usr/bin/env python3
"""
Created on Fri Sep 27 13:08:41 2019
@author: vkapoor
"""


import math
import os
from pathlib import Path
import torch
import napari
import gc
from skimage.transform import resize
import numpy as np
from numba import jit
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import convolve, mean
import cv2
from skimage.segmentation import clear_border
from scipy.ndimage import gaussian_filter

# import matplotlib.pyplot as plt
import pandas as pd
from cellpose import models
from csbdeep.utils import normalize
from concurrent.futures import ThreadPoolExecutor
from scipy import spatial
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
)
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.morphology import binary_fill_holes
from skimage import measure, morphology
from skimage.filters import threshold_multiotsu
from skimage.measure import label, regionprops
from skimage.morphology import (
    dilation,
    remove_small_objects,
    skeletonize,
    square,
)
from skimage.segmentation import (
    find_boundaries,
    relabel_sequential,
    watershed,
    join_segmentations,
)
from skimage.util import invert as invertimage
from tifffile import imread, imwrite
from tqdm import tqdm
from typing import Union
from .StarDist3D import StarDist3D
from .UNET import UNET
from .CARE import CARE
from .MASKUNET import MASKUNET
from vollseg.matching import matching
from vollseg.nmslabel import NMSLabel
from vollseg.seedpool import SeedPool
from vollseg.unetstarmask import UnetStarMask
from numba import njit
from csbdeep.models import ProjectionCARE
from skimage.filters import threshold_otsu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Boxname = "ImageIDBox"
GLOBAL_THRESH = 1.0e-2
GLOBAL_ERODE = 8


class SegCorrect:
    def __init__(self, imagedir, segmentationdir):

        self.imagedir = imagedir
        self.segmentationdir = segmentationdir
        self.acceptable_formats = [".tif", ".TIFF", ".TIF", ".png"]

    def showNapari(self):
        from qtpy.QtWidgets import QComboBox, QPushButton

        self.viewer = napari.Viewer()
        X = os.listdir(self.imagedir)
        Imageids = []
        Y = os.listdir(self.segmentationdir)
        SegImageids = []
        for imagename in X:
            if any(imagename.endswith(f) for f in self.acceptable_formats):
                Imageids.append(imagename)
        for imagename in Y:
            if any(imagename.endswith(f) for f in self.acceptable_formats):
                SegImageids.append(imagename)

        imageidbox = QComboBox()
        imageidbox.addItem(Boxname)
        savebutton = QPushButton(" Save Corrections")

        for i in range(0, len(SegImageids)):

            imageidbox.addItem(str(SegImageids[i]))

        imageidbox.currentIndexChanged.connect(
            lambda trackid=imageidbox: self.image_add(
                imageidbox.currentText(),
                Path(Path(self.imagedir, imageidbox.currentText())),
                Path(Path(self.segmentationdir, imageidbox.currentText())),
                False,
            )
        )

        savebutton.clicked.connect(
            lambda trackid=imageidbox: self.image_add(
                imageidbox.currentText(),
                Path(Path(self.imagedir, imageidbox.currentText())),
                Path(Path(self.segmentationdir, imageidbox.currentText())),
                True,
            )
        )

        self.viewer.window.add_dock_widget(imageidbox, name="Image", area="bottom")
        self.viewer.window.add_dock_widget(
            savebutton, name="Save Segmentations", area="bottom"
        )

        napari.run()

    def image_add(
        self,
        imagename: str,
        image_toread: Path,
        seg_image_toread: Path,
        save=False,
    ):
        if seg_image_toread.exists():
            if not save:
                for layer in list(self.viewer.layers):

                    if "Image" in layer.name or layer.name in "Image":

                        self.viewer.layers.remove(layer)

                self.image = imread(image_toread)
                self.segimage = imread(seg_image_toread)

                self.viewer.add_image(self.image, name="Image" + imagename)
                self.viewer.add_labels(
                    self.segimage, name="Image" + "Integer_Labels" + imagename
                )

            if save:

                ModifiedArraySeg = self.viewer.layers[
                    "Image" + "Integer_Labels" + imagename
                ].data
                ModifiedArraySeg = ModifiedArraySeg.astype("uint16")
                imwrite(
                    (os.path.join(self.segmentationdir, imagename)),
                    ModifiedArraySeg,
                )


class ProjSegCreate:
    def __init__(self, imagedir, segmentationdir):

        self.imagedir = imagedir
        self.segmentationdir = segmentationdir
        self.acceptable_formats = [".tif", ".TIFF", ".TIF", ".png"]

    def showNapari(self):
        from qtpy.QtWidgets import QComboBox, QPushButton

        self.viewer = napari.Viewer()
        X = os.listdir(self.imagedir)
        Imageids = []
        for imagename in X:
            if any(imagename.endswith(f) for f in self.acceptable_formats):
                Imageids.append(imagename)

        imageidbox = QComboBox()
        imageidbox.addItem(Boxname)
        savebutton = QPushButton(" Save Segmentation")

        for i in range(0, len(Imageids)):

            imageidbox.addItem(str(Imageids[i]))

        imageidbox.currentIndexChanged.connect(
            lambda trackid=imageidbox: self.image_add(
                imageidbox.currentText(),
                Path(Path(self.imagedir, imageidbox.currentText())),
                False,
            )
        )

        savebutton.clicked.connect(
            lambda trackid=imageidbox: self.image_add(
                imageidbox.currentText(),
                Path(Path(self.imagedir, imageidbox.currentText())),
                True,
            )
        )

        self.viewer.window.add_dock_widget(imageidbox, name="Image", area="bottom")
        self.viewer.window.add_dock_widget(
            savebutton, name="Save Segmentations", area="bottom"
        )

        napari.run()

    def image_add(
        self,
        imagename: str,
        image_toread: Path,
        save=False,
    ):
        if not save:
            for layer in list(self.viewer.layers):

                if "Image" in layer.name or layer.name in "Image":

                    self.viewer.layers.remove(layer)

            self.image = imread(image_toread)
            self.segimage = np.zeros(
                [self.image.shape[1], self.image.shape[2]], dtype=np.uint16
            )

            self.viewer.add_image(self.image, name="Image" + imagename)
            self.viewer.add_labels(
                self.segimage, name="Image" + "Integer_Labels" + imagename
            )

        if save:

            ModifiedArraySeg = self.viewer.layers[
                "Image" + "Integer_Labels" + imagename
            ].data
            ModifiedArraySeg = ModifiedArraySeg.astype("uint16")
            imwrite(
                (os.path.join(self.segmentationdir, imagename)),
                ModifiedArraySeg,
            )


def BinaryLabel(BinaryImageOriginal, max_size=15000):

    BinaryImageOriginal = BinaryImageOriginal.astype("uint16")
    image = normalizeFloatZeroOne(BinaryImageOriginal)
    image = invertimage(image)
    IntegerImage = watershed(-image)
    AugmentedLabel = remove_big_objects(IntegerImage, max_size=max_size)

    return AugmentedLabel


def expand_labels(label_image, distance=1):

    distances, nearest_label_coords = distance_transform_edt(
        label_image == 0, return_indices=True
    )
    labels_out = np.zeros_like(label_image)
    dilate_mask = distances <= distance

    masked_nearest_label_coords = [
        dimension_indices[dilate_mask] for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels
    return labels_out


def SimplePrediction(
    x,
    UnetModel,
    StarModel,
    n_tiles=(2, 2),
    UseProbability=True,
    axes="ZYX",
    ExpandLabels=True,
):

    Mask = UNETPrediction3D(x, UnetModel, n_tiles, axes, ExpandLabels)

    smart_seeds, _, _, _ = STARPrediction3D(
        x,
        axes,
        StarModel,
        n_tiles,
        unet_mask=Mask,
        smartcorrection=None,
        UseProbability=UseProbability,
    )

    smart_seeds = smart_seeds.astype("uint16")

    return smart_seeds


def fill_label_holes(lbl_img, **kwargs):
    """Fill small holes in label image."""
    # TODO: refactor 'fill_label_holes' and 'edt_prob' to share code
    def grow(sl, interior):
        return tuple(
            slice(s.start - int(w[0]), s.stop + int(w[1])) for s, w in zip(sl, interior)
        )

    def shrink(interior):
        return tuple(slice(int(w[0]), (-1 if w[1] else None)) for w in interior)

    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)
    for i, sl in enumerate(objects, 1):
        if sl is None:
            continue
        interior = [(s.start > 0, s.stop < sz) for s, sz in zip(sl, lbl_img.shape)]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl, interior)] == i
        mask_filled = binary_fill_holes(grown_mask, **kwargs)[shrink_slice]
        lbl_img_filled[sl][mask_filled] = i
    return lbl_img_filled


def dilate_label_holes(lbl_img, iterations):
    lbl_img_filled = np.zeros_like(lbl_img)
    for lb in range(np.min(lbl_img), np.max(lbl_img) + 1):
        mask = lbl_img == lb
        mask_filled = binary_dilation(mask, iterations=iterations)
        lbl_img_filled[mask_filled] = lb
    return lbl_img_filled


def erode_labels(lbl_img, iterations=1):
    lbl_img_filled = np.zeros_like(lbl_img)
    for lb in range(np.min(lbl_img), np.max(lbl_img) + 1):
        mask = lbl_img == lb
        mask_filled = binary_erosion(mask, iterations=iterations)
        lbl_img_filled[mask_filled] = lb
    return lbl_img_filled


def erode_label_regions(segmentation, erosion_iterations=1):
    regions = regionprops(segmentation)
    erode = np.zeros(segmentation.shape)

    def erode_mask(segmentation_labels, label_id, erosion_iterations):
        only_current_label_id = np.where(segmentation_labels == label_id, 1, 0)
        eroded = binary_erosion(only_current_label_id, iterations=erosion_iterations)
        relabeled_eroded = np.where(eroded == 1, label_id, 0)
        return relabeled_eroded

    def process_region_2d(label_id):
        return erode_mask(segmentation, label_id, erosion_iterations)

    def process_region_3d(label_id, z):
        return erode_mask(segmentation[z, :, :], label_id, erosion_iterations)

    # For 3D segmentation, we parallelize over both regions and slices (z-axis)
    if segmentation.ndim == 3:
        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(len(regions)):
                label_id = regions[i].label
                for z in range(segmentation.shape[0]):
                    futures.append(executor.submit(process_region_3d, label_id, z))

            # Aggregate results
            for future in futures:
                result, z = (
                    future.result(),
                    futures.index(future) % segmentation.shape[0],
                )
                erode[z, :, :] += result

    # For 2D segmentation, we parallelize only over regions
    else:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_region_2d, regions[i].label)
                for i in range(len(regions))
            ]

            # Aggregate results
            for future in futures:
                erode += future.result()

    return erode


def dilate_label_regions(segmentation, dilation_iterations=1):
    regions = regionprops(segmentation)
    erode = np.zeros(segmentation.shape)

    def dilate_mask(segmentation_labels, label_id, dilation_iterations):
        only_current_label_id = np.where(segmentation_labels == label_id, 1, 0)
        dilated = binary_dilation(only_current_label_id, iterations=dilation_iterations)
        relabeled_dilated = np.where(dilated == 1, label_id, 0)
        return relabeled_dilated

    def process_region_2d(label_id):
        return dilate_mask(segmentation, label_id, dilation_iterations)

    def process_region_3d(label_id, z):
        return dilate_mask(segmentation[z, :, :], label_id, dilation_iterations)

    # For 3D segmentation, parallelize over both regions and slices (z-axis)
    if segmentation.ndim == 3:
        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(len(regions)):
                label_id = regions[i].label
                for z in range(segmentation.shape[0]):
                    futures.append(executor.submit(process_region_3d, label_id, z))

            # Aggregate results
            for future in futures:
                result, z = (
                    future.result(),
                    futures.index(future) % segmentation.shape[0],
                )
                erode[z, :, :] += result

    # For 2D segmentation, parallelize only over regions
    else:
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_region_2d, regions[i].label)
                for i in range(len(regions))
            ]

            # Aggregate results
            for future in futures:
                erode += future.result()

    return erode


def match_labels(ys: np.ndarray, nms_thresh=0.5):

    if nms_thresh is None:
        nms_thresh = 0.3
    ys_grouped = np.empty_like(ys, dtype=np.uint16)

    def _match_single(y_prev, y, next_id):
        y = y.astype(np.uint16, copy=False)
        res = matching(
            y_prev,
            y,
            report_matches=True,
            thresh=nms_thresh,
            criterion="iou",
        )
        # relabel dict (for matching labels) that maps label ids from y -> y_prev
        relabel = dict(reversed(res.matched_pairs[i]) for i in res.matched_tps)
        y_grouped = np.zeros_like(y)
        for r in regionprops(y):
            m = y[r.slice] == r.label
            if r.label in relabel:
                y_grouped[r.slice][m] = relabel[r.label]
            else:
                y_grouped[r.slice][m] = next_id
                next_id += 1
        return y_grouped, next_id

    ys_grouped[0] = ys[0]
    next_id = ys_grouped[0].max() + 1
    for i in range(len(ys) - 1):
        ys_grouped[i + 1], next_id = _match_single(ys_grouped[i], ys[i + 1], next_id)
    return ys_grouped


def remove_big_objects(ar: np.ndarray, max_size):

    out = ar.copy()
    ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.model_dimage.label` or "
            "`skimage.morphology.label`."
        )

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
    AugmentedLabel = dilation(labelclean, footprint=square(3))
    AugmentedLabel = np.multiply(AugmentedLabel, Orig)

    return AugmentedLabel


def CreateTrackMate_CSV(Label, Name, savedir):

    TimeList = []

    XList = []
    YList = []
    TrackIDList = []
    QualityList = []

    CurrentSegimage = Label.astype("uint16")
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
        TimeList.append(int(T))
        XList.append(int(X))
        YList.append(int(Y))
        TrackIDList.append(regionlabel)
        QualityList.append(radius)

    df = pd.DataFrame(
        list(zip(XList, YList, TimeList, TrackIDList, QualityList)),
        index=None,
        columns=["POSITION_X", "POSITION_Y", "FRAME", "TRACK_ID", "QUALITY"],
    )

    df.to_csv(savedir + "/" + "TrackMate_csv" + Name + ".csv", index=False)


def SmartSkel(smart_seedsLabels, ProbImage, RGB=False):

    if RGB:
        return smart_seedsLabels > 0
    SegimageB = find_boundaries(smart_seedsLabels)
    invertProbimage = 1 - ProbImage
    image_max = np.add(invertProbimage, SegimageB)

    pixel_condition = image_max < 1.2
    pixel_replace_condition = 0
    image_max = image_conditionals(image_max, pixel_condition, pixel_replace_condition)

    skeleton = skeletonize(image_max.astype("uint16") > 0)

    return skeleton


def Skel(smart_seedsLabels, RGB=False):

    if RGB:
        return smart_seedsLabels > 0
    image_max = find_boundaries(smart_seedsLabels)

    skeleton = skeletonize(image_max.astype("uint16") > 0)

    return skeleton


# If there are neighbouring seeds we do not put more seeds


def Region_embedding(image, region, sourceimage, dtype=np.float32, RGB=False):
    returnimage = np.zeros(image.shape, dtype=dtype)
    if len(region) == 4 and len(image.shape) == 2:
        rowstart = region[0]
        colstart = region[1]
        endrow = region[2]
        endcol = region[3]
        returnimage[rowstart:endrow, colstart:endcol] = sourceimage
    if len(image.shape) == 3 and len(region) == 6 and RGB is False:
        zstart = region[0]
        rowstart = region[1]
        colstart = region[2]
        zend = region[3]
        endrow = region[4]
        endcol = region[5]
        returnimage[zstart:zend, rowstart:endrow, colstart:endcol] = sourceimage

    if len(image.shape) == 3 and len(region) == 4 and RGB is False:
        rowstart = region[0]
        colstart = region[1]
        endrow = region[2]
        endcol = region[3]
        returnimage[0 : image.shape[0], rowstart:endrow, colstart:endcol] = sourceimage

    if len(image.shape) == 3 and len(region) == 4 and RGB:
        returnimage = returnimage[:, :, 0]
        rowstart = region[0]
        colstart = region[1]
        endrow = region[2]
        endcol = region[3]
        returnimage[rowstart:endrow, colstart:endcol] = sourceimage

    return returnimage


def VollSeg2D(
    image,
    unet_model,
    star_model,
    noise_model=None,
    roi_model=None,
    prob_thresh=None,
    nms_thresh=None,
    axes="YX",
    min_size_mask=5,
    min_size=5,
    max_size=10000000,
    dounet=True,
    n_tiles=(2, 2),
    donormalize=True,
    lower_perc=1,
    upper_perc=99.8,
    UseProbability=True,
    RGB=False,
    seedpool=False,
):

    if star_model is not None:
        nms_thresh = star_model.thresholds[1]
    elif nms_thresh is not None:
        nms_thresh = nms_thresh
    else:
        nms_thresh = 0

    if RGB:
        axes = "YXC"
    if "T" in axes:
        axes = "YX"
        if RGB:
            axes = "YXC"
    if noise_model is not None:

        image = noise_model.predict(image.astype("float32"), axes=axes, n_tiles=n_tiles)
        pixel_condition = image < 0
        pixel_replace_condition = 0
        image = image_conditionals(image, pixel_condition, pixel_replace_condition)

    Mask = None
    Mask_patch = None
    roi_image = None
    if roi_model is not None:
        model_dim = roi_model.config.n_dim
        assert model_dim == len(
            image.shape
        ), f"For 2D images the region of interest model has to be 2D, model provided had {model_dim} instead"
        Segmented = roi_model.predict(image.astype("float32"), "YX", n_tiles=n_tiles)
        try:
            thresholds = threshold_multiotsu(Segmented, classes=2)

            # Using the threshold values, we generate the three regions.
            regions = np.digitize(Segmented, bins=thresholds)
        except ValueError:

            regions = Segmented

        roi_image = regions > 0
        roi_image = label(roi_image)
        roi_bbox = Bbox_region(roi_image)
        if roi_bbox is not None:
            rowstart = roi_bbox[0]
            colstart = roi_bbox[1]
            endrow = roi_bbox[2]
            endcol = roi_bbox[3]
            region = (slice(rowstart, endrow), slice(colstart, endcol))
            # The actual pixels in that region.
            patch = image[region]
        else:

            patch = image
            region = (slice(0, image.shape[0]), slice(0, image.shape[1]))
            rowstart = 0
            colstart = 0
            endrow = image.shape[1]
            endcol = image.shape[0]
            roi_bbox = [colstart, rowstart, endcol, endrow]

    else:

        patch = image

        region = (slice(0, image.shape[0]), slice(0, image.shape[1]))
        rowstart = 0
        colstart = 0
        endrow = image.shape[1]
        endcol = image.shape[0]
        roi_bbox = [colstart, rowstart, endcol, endrow]
    if dounet:

        if unet_model is not None:

            Segmented = unet_model.predict(
                image.astype("float32"), axes, n_tiles=n_tiles
            )
        else:
            Segmented = image
        if RGB:
            Segmented = Segmented[:, :, 0]

        try:
            thresholds = threshold_multiotsu(Segmented, classes=2)

            # Using the threshold values, we generate the three regions.
            regions = np.digitize(Segmented, bins=thresholds)
        except ValueError:

            regions = Segmented
        Binary = regions > 0
        Mask = Binary.copy()

        Mask = Region_embedding(image, roi_bbox, Mask, dtype=np.uint8, RGB=RGB)
        Mask_patch = Mask.copy()
    elif noise_model is not None and dounet is False:

        Mask = np.zeros(patch.shape)
        try:
            thresholds = threshold_multiotsu(patch, classes=2)

            # Using the threshold values, we generate the three regions.
            regions = np.digitize(patch, bins=thresholds)
        except ValueError:

            regions = patch
        Mask = regions > 0

        Mask = label(Mask)
        Mask = remove_small_objects(Mask.astype("uint16"), min_size=min_size_mask)
        Mask = remove_big_objects(Mask.astype("uint16"), max_size=max_size)

        if RGB:
            Mask = Mask[:, :, 0]
            Mask_patch = Mask_patch[:, :, 0]
        Mask = Region_embedding(image, roi_bbox, Mask, dtype=np.uint8, RGB=RGB)
        Mask_patch = Mask.copy()
    # Smart Seed prediction
    if RGB:
        axis = (0, 1, 2)
    else:
        axis = (0, 1)
    if donormalize:
        patch_star = normalize(
            patch.astype("float32"), lower_perc, upper_perc, axis=axis
        )
    else:
        patch_star = patch
    smart_seeds, markers, star_labels, probability_map = SuperSTARPrediction(
        patch_star,
        star_model,
        n_tiles,
        unet_mask=Mask_patch,
        UseProbability=UseProbability,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
        seedpool=seedpool,
    )
    smart_seeds = remove_small_objects(smart_seeds.astype("uint16"), min_size=min_size)
    smart_seeds = remove_big_objects(smart_seeds.astype("uint16"), max_size=max_size)
    star_labels = remove_small_objects(star_labels.astype("uint16"), min_size=min_size)
    star_labels = remove_big_objects(star_labels.astype("uint16"), max_size=max_size)
    
    skeleton = SmartSkel(smart_seeds, probability_map, RGB)
    skeleton = skeleton > 0
    # For avoiding pixel level error
    if Mask is not None:
        Mask = expand_labels(Mask, distance=1)

    smart_seeds = expand_labels(smart_seeds, distance=1)

    smart_seeds = Region_embedding(
        image, roi_bbox, smart_seeds, dtype=np.uint16, RGB=RGB
    )
    markers = Region_embedding(image, roi_bbox, markers, dtype=np.uint16, RGB=RGB)
    star_labels = Region_embedding(
        image, roi_bbox, star_labels, dtype=np.uint16, RGB=RGB
    )
    probability_map = Region_embedding(image, roi_bbox, probability_map, RGB=RGB)
    skeleton = Region_embedding(image, roi_bbox, skeleton, dtype=np.uint8, RGB=RGB)
    if Mask is None:
        Mask = smart_seeds > 0

    if noise_model is None and roi_image is not None:
        return (
            smart_seeds.astype("uint16"),
            Mask.astype("uint16"),
            star_labels.astype("uint16"),
            probability_map,
            markers.astype("uint16"),
            skeleton.astype("uint16"),
            roi_image.astype("uint16"),
        )

    if noise_model is None and roi_image is None:
        return (
            smart_seeds.astype("uint16"),
            Mask.astype("uint16"),
            star_labels.astype("uint16"),
            probability_map,
            markers.astype("uint16"),
            skeleton.astype("uint16"),
        )

    if noise_model is not None and roi_image is not None:
        return (
            smart_seeds.astype("uint16"),
            Mask.astype("uint16"),
            star_labels.astype("uint16"),
            probability_map,
            markers.astype("uint16"),
            skeleton.astype("uint16"),
            image,
            roi_image.astype("uint16"),
        )

    if noise_model is not None and roi_image is None:
        return (
            smart_seeds.astype("uint16"),
            Mask.astype("uint16"),
            star_labels.astype("uint16"),
            probability_map,
            markers.astype("uint16"),
            skeleton.astype("uint16"),
            image,
        )


def VollSeg_nolabel_precondition(image, Finalimage):

    model_dim = len(image.shape)
    if model_dim == 3:
        for i in range(image.shape[0]):
            Finalimage[i] = expand_labels(Finalimage[i], distance=GLOBAL_ERODE)

    return Finalimage


def VollSeg_label_precondition(image, overall_mask, Finalimage):

    model_dim = len(image.shape)
    if model_dim == 3:
        for i in range(image.shape[0]):
            Finalimage[i] = expand_labels(Finalimage[i], distance=50)
        pixel_condition = overall_mask == 0
        pixel_replace_condition = 0
        Finalimage = image_conditionals(
            Finalimage, pixel_condition, pixel_replace_condition
        )

    return Finalimage


def VollSeg_label_expansion(image, overall_mask, Finalimage, skeleton, RGB):

    for i in range(image.shape[0]):
        Finalimage[i, :] = expand_labels(Finalimage[i, :], distance=50)
        skeleton[i, :] = Skel(Finalimage[i, :], RGB)
        skeleton[i, :] = skeleton[i, :] > 0
    pixel_condition = overall_mask == 0
    pixel_replace_condition = 0
    Finalimage = image_conditionals(
        Finalimage, pixel_condition, pixel_replace_condition
    )
    skeleton = image_conditionals(skeleton, pixel_condition, pixel_replace_condition)

    return Finalimage, skeleton


def VollSeg_nolabel_expansion(image, Finalimage, skeleton, RGB):

    for i in range(image.shape[0]):
        Finalimage[i, :] = expand_labels(Finalimage[i, :], distance=GLOBAL_ERODE)
        skeleton[i, :] = Skel(Finalimage[i, :], RGB)
        skeleton[i, :] = skeleton[i, :] > 0

    return Finalimage, skeleton


def VollSeg_unet(
    image,
    unet_model=None,
    roi_model=None,
    n_tiles=(2, 2),
    axes="YX",
    ExpandLabels=True,
    noise_model=None,
    min_size_mask=100,
    max_size=10000000,
    RGB=False,
    nms_thresh=0.3,
    slice_merge=False,
    dounet=True,
    erosion_iterations=2,
):
    Finalimage = np.zeros(image.shape, dtype=np.uint16)
    skeleton = np.zeros(image.shape, dtype=np.uint8)
    model_dim = len(image.shape)
    if len(n_tiles) != model_dim:
        if model_dim == 3:
            n_tiles = (n_tiles[-3], n_tiles[-2], n_tiles[-1])
        if model_dim == 2:
            n_tiles = (n_tiles[-2], n_tiles[-1])
    if roi_model is not None:

        if noise_model is not None:
            if isinstance(noise_model, ProjectionCARE):
                n_tiles = (1, n_tiles[-2], n_tiles[-1])
            image = noise_model.predict(image.astype("float32"), axes, n_tiles=n_tiles)

            pixel_condition = image < 0
            pixel_replace_condition = 0
            image = image_conditionals(image, pixel_condition, pixel_replace_condition)

        model_dim = roi_model.config.n_dim
        if model_dim < len(image.shape):
            if len(n_tiles) == len(image.shape):
                tiles = (n_tiles[1], n_tiles[2])
            else:
                tiles = n_tiles
            maximage = np.amax(image, axis=0)
            if isinstance(roi_model, ProjectionCARE):
                n_tiles = (1, n_tiles[-2], n_tiles[-1])
            roi_Segmented = roi_model.predict(
                maximage.astype("float32"), "YX", n_tiles=tiles
            )
            try:
                thresholds = threshold_multiotsu(roi_Segmented, classes=2)

                # Using the threshold values, we generate the three regions.
                roi_regions = np.digitize(roi_Segmented, bins=thresholds)
            except ValueError:

                roi_regions = roi_Segmented

            s_Binary = roi_regions > 0

            s_Binary = label(s_Binary)
            s_Binary = remove_small_objects(
                s_Binary.astype("uint16"), min_size=min_size_mask
            )
            s_Binary = remove_big_objects(s_Binary.astype("uint16"), max_size=max_size)
            s_Binary = fill_label_holes(s_Binary)

        elif model_dim == len(image.shape):
            if isinstance(roi_model, ProjectionCARE):
                n_tiles = (1, n_tiles[-2], n_tiles[-1])
            roi_Segmented = roi_model.predict(
                image.astype("float32"), axes, n_tiles=n_tiles
            )
            try:
                thresholds = threshold_multiotsu(roi_Segmented, classes=2)

                # Using the threshold values, we generate the three regions.
                roi_regions = np.digitize(roi_Segmented, bins=thresholds)
            except ValueError:

                roi_regions = roi_Segmented

            s_Binary = roi_regions > 0

            s_Binary = label(s_Binary)
            s_Binary = fill_label_holes(s_Binary)
            s_Binary = binary_dilation(s_Binary, iterations=erosion_iterations)
            if len(s_Binary.shape) == 3 and slice_merge:
                for i in range(image.shape[0]):
                    s_Binary[i] = label(s_Binary[i])

                s_Binary = match_labels(s_Binary, nms_thresh=nms_thresh)
                s_Binary = fill_label_holes(s_Binary)
                for i in range(image.shape[0]):
                    s_Binary[i] = remove_small_objects(
                        s_Binary[i].astype("uint16"), min_size=min_size_mask
                    )
                    s_Binary[i] = remove_big_objects(
                        s_Binary[i].astype("uint16"), max_size=max_size
                    )

    if unet_model is not None:
        if RGB:
            if n_tiles is not None:
                n_tiles = (n_tiles[0], n_tiles[1], 1)

        if noise_model is not None:
            if isinstance(noise_model, ProjectionCARE):
                n_tiles = (1, n_tiles[-2], n_tiles[-1])
            image = noise_model.predict(image.astype("float32"), axes, n_tiles=n_tiles)
            if roi_model is not None:
                pixel_condition = s_Binary > 0
                pixel_replace_condition = 1
                s_Binary = image_conditionals(
                    s_Binary, pixel_condition, pixel_replace_condition
                )

        if dounet:
            Segmented = unet_model.predict(
                image.astype("float32"), axes, n_tiles=n_tiles
            )
        else:
            Segmented = image
        if RGB:
            Segmented = Segmented[:, :, 0]

        try:
            thresholds = threshold_multiotsu(Segmented, classes=2)

            # Using the threshold values, we generate the three regions.
            regions = np.digitize(Segmented, bins=thresholds)
        except ValueError:

            regions = Segmented
        Binary = regions > 0
        overall_mask = Binary.copy()

        if model_dim == 3:
            for i in range(image.shape[0]):
                overall_mask[i] = binary_dilation(
                    overall_mask[i], iterations=erosion_iterations
                )
                overall_mask[i] = binary_erosion(
                    overall_mask[i], iterations=erosion_iterations
                )
                overall_mask[i] = fill_label_holes(overall_mask[i])

        Binary = label(Binary)

        if model_dim == 2:
            Binary = remove_small_objects(
                Binary.astype("uint16"), min_size=min_size_mask
            )
            Binary = remove_big_objects(Binary.astype("uint16"), max_size=max_size)
            Binary = fill_label_holes(Binary)
            Finalimage = relabel_sequential(Binary)[0]

            skeleton = Skel(Finalimage, RGB)
            skeleton = skeleton > 0

        if model_dim == 3:
            for i in range(image.shape[0]):
                Binary[i] = binary_erosion(Binary[i], iterations=2)
                Binary[i] = label(Binary[i])
                Binary[i] = dilate_label_holes(Binary[i], iterations=2)
                Binary[i] = remove_small_objects(
                    Binary[i].astype("uint16"), min_size=min_size_mask
                )
                Binary[i] = remove_big_objects(
                    Binary[i].astype("uint16"), max_size=max_size
                )
            Finalimage = relabel_sequential(Binary)[0]
            skeleton = Skel(Finalimage)
            Finalimage = clear_border(Finalimage)
            if ExpandLabels:

                Finalimage, skeleton = VollSeg_label_expansion(
                    image, overall_mask, Finalimage, skeleton, RGB
                )
            if slice_merge:
                Finalimage = match_labels(Finalimage, nms_thresh=nms_thresh)
                Finalimage = fill_label_holes(Finalimage)

    if roi_model is None:
        return Finalimage.astype("uint16"), skeleton, image
    else:
        return Finalimage.astype("uint16"), skeleton, image, s_Binary





def collate_fn(data):

    slices = []
    input_tensor = []
    for x, y in data:

        if len(input_tensor) == 0:
            input_tensor = torch.stack([x])
        else:
            input_tensor = torch.stack([input_tensor, x])

        slices.append(y)

    return input_tensor, slices



def CellPoseSeg(
    image: np.ndarray,
    diameter_cellpose: float = 34.6,
    stitch_threshold: float = 0.5,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    anisotropy=None,
    cellpose_model_path: str = None,
    cellpose_model_type: str = None,
    gpu: bool = False,
    axes: str = "ZYX",
    save_dir: str = None,
    Name: str = "Result",
    do_3D: bool = False,
    channels=[0, 0],
    bsize=224
):

    if len(image.shape) == 3 and "T" not in axes:
        if cellpose_model_path is not None:
            cellpose_model = models.CellposeModel(
                gpu=gpu, pretrained_model=cellpose_model_path
            )
        if cellpose_model_type is not None:
            cellpose_model = models.CellposeModel(
                gpu=gpu, model_type=cellpose_model_type
            )

        if anisotropy is not None:
            cellres = cellpose_model.eval(
                image,
                diameter=diameter_cellpose,
                channels=channels,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                stitch_threshold=stitch_threshold,
                anisotropy=anisotropy,
                bsize=bsize,
                do_3D=do_3D,
            )
        else:
            cellres = cellpose_model.eval(
                image,
                diameter=diameter_cellpose,
                channels=channels,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                stitch_threshold=stitch_threshold,
                bsize=bsize,
                do_3D=do_3D,
            )

    if len(image.shape) == 4 and "T" in axes:

        cellpose_model = models.CellposeModel(
            gpu=gpu, pretrained_model=cellpose_model_path
        )
      
        if anisotropy is not None:

            
            cellres = tuple(
                zip(
                    *tuple(
                        cellpose_model.eval(
                            _x,
                            diameter=diameter_cellpose,
                            channels=channels,
                            flow_threshold=flow_threshold,
                            cellprob_threshold=cellprob_threshold,
                            stitch_threshold=stitch_threshold,
                            anisotropy=anisotropy,
                            bsize = bsize,
                            do_3D=do_3D,
                        )
                        for _x in tqdm(image)
                    )
                )
            )
        else:
            cellres = tuple(
                zip(
                    *tuple(
                        cellpose_model.eval(
                            _x,
                            diameter=diameter_cellpose,
                            channels=channels,
                            flow_threshold=flow_threshold,
                            cellprob_threshold=cellprob_threshold,
                            stitch_threshold=stitch_threshold,
                            bsize = bsize,
                            do_3D=do_3D,
                        )
                        for _x in tqdm(image)
                    )
                )
            )


    if len(image.shape) == 3 and "T" in axes:

        cellpose_model = models.CellposeModel(
            gpu=gpu, pretrained_model=cellpose_model_path
        )
      
        if anisotropy is not None:

            
            cellres = cellpose_model.eval(
                            image,
                            diameter=diameter_cellpose,
                            channels=channels,
                            flow_threshold=flow_threshold,
                            cellprob_threshold=cellprob_threshold,
                            stitch_threshold=stitch_threshold,
                            anisotropy=anisotropy,
                            bsize = bsize,
                            do_3D=do_3D,
                        )
        else:
            cellres =  cellpose_model.eval(
                            image,
                            diameter=diameter_cellpose,
                            channels=channels,
                            flow_threshold=flow_threshold,
                            cellprob_threshold=cellprob_threshold,
                            stitch_threshold=stitch_threshold,
                            bsize = bsize,
                            do_3D=do_3D,
                        )
                       
                  

    if cellpose_model_path is not None or cellpose_model_type is not None:
        cellpose_labels = cellres[0]
        cellpose_labels = np.asarray(cellpose_labels)

    if save_dir is not None:
        Path(save_dir).mkdir(exist_ok=True)

        if cellpose_model_path is not None or cellpose_model_type is not None:
            cellpose_results = Path(save_dir) / "CellPose"
            Path(cellpose_results).mkdir(exist_ok=True)
            imwrite(
                (os.path.join(cellpose_results.as_posix(), Name + ".tif")),
                np.asarray(cellpose_labels).astype("uint16"),
            )

    return cellpose_labels


def VollCellSeg(
    image: np.ndarray,
    nuclei_seg_image: np.ndarray,
    mask: np.ndarray = None,
    channel_membrane: int = 0,
    axes: str = "ZYX",
    n_tiles: tuple = (1, 1, 1),
    save_dir: str = None,
    Name: str = "Result",
):

    if len(image.shape) == 3 and "T" not in axes:
        image_membrane = image

    if len(image.shape) == 4 and "T" not in axes:
        image_membrane = image[:, channel_membrane, :, :]

    if len(image.shape) > 4 and "T" in axes:

        if len(n_tiles) == 4:
            n_tiles = (n_tiles[1], n_tiles[2], n_tiles[3])
        image_membrane = image[:, :, channel_membrane, :, :]

    if nuclei_seg_image is not None:

        voll_cell_seg = _cellwater_block(axes, image_membrane, nuclei_seg_image, mask)

        if save_dir is not None:
            Path(save_dir).mkdir(exist_ok=True)

            vollcellpose_results = Path(save_dir) / "VollCellPose"
            Path(vollcellpose_results).mkdir(exist_ok=True)
            imwrite(
                (os.path.join(vollcellpose_results.as_posix(), Name + ".tif")),
                np.asarray(voll_cell_seg).astype("uint16"),
            )


def _cellwater_block(axes, membrane_image, sized_smart_seeds, mask):

    if "T" not in axes:

        voll_cell_seg = CellPoseWater(membrane_image, sized_smart_seeds, mask)
    if "T" in axes:

        voll_cell_seg = []
        for time in range(sized_smart_seeds.shape[0]):
            sized_smart_seeds_time = sized_smart_seeds[time]
            membrane_image_time = membrane_image[time]
            voll_cell_seg_time = CellPoseWater(
                membrane_image_time, sized_smart_seeds_time, mask
            )
            voll_cell_seg.append(voll_cell_seg_time)
        voll_cell_seg = np.asarray(voll_cell_seg_time)

    return voll_cell_seg


def check_and_update_mask(mask, image):

    if len(mask.shape) < len(image.shape):
        update_mask = np.zeros(
            [
                image.shape[0],
                image.shape[1],
                image.shape[2],
            ],
            dtype="uint8",
        )
        for i in range(0, update_mask.shape[0]):
            update_mask[i, :, :] = mask
    else:
        update_mask = mask

    return update_mask


def VollOne(
    image: np.ndarray,
    channel_membrane: int = 0,
    channel_nuclei: int = 1,
    star_model_nuclei: Union[StarDist3D, None] = None,
    unet_model_nuclei: Union[UNET, None] = None,
    unet_model_membrane: Union[UNET, None] = None,
    noise_model_membrane: Union[CARE, None] = None,
    roi_model: Union[MASKUNET, ProjectionCARE, None] = None,
    prob_thresh: float = None,
    ExpandLabels: bool = False,
    nms_thresh: float = None,
    slice_merge_nuclei: bool = False,
    slice_merge_membrane: bool = True,
    min_size_mask: int = 10,
    min_size: int = 10,
    max_size: int = 10000,
    n_tiles: tuple = (1, 1, 1),
    UseProbability: bool = True,
    donormalize: bool = True,
    lower_perc: float = 1.0,
    upper_perc: float = 99.8,
    dounet: bool = True,
    seedpool: bool = True,
    save_dir: str = None,
    Name: str = "Result",
    axes: str = "CZYX",
):

    channel_index = axes.index("C")
    if prob_thresh is None and nms_thresh is None:
        prob_thresh = star_model_nuclei.thresholds.prob
        nms_thresh = star_model_nuclei.thresholds.nms

    if len(image.shape) > 4 and "T" in axes:

        if len(n_tiles) == 4:
            n_tiles = (n_tiles[1], n_tiles[2], n_tiles[3])

        image_membrane = np.take(image, channel_membrane, axis=channel_index)
        image_nuclei = np.take(image, channel_nuclei, axis=channel_index)
        nuclei_res = tuple(
            zip(
                *tuple(
                    VollSeg(
                        image_nuclei[i],
                        unet_model=unet_model_nuclei,
                        star_model=star_model_nuclei,
                        axes=(axes.replace("C", "")).replace("T", ""),
                        prob_thresh=prob_thresh,
                        nms_thresh=nms_thresh,
                        min_size_mask=min_size_mask,
                        min_size=min_size,
                        max_size=max_size,
                        n_tiles=n_tiles,
                        UseProbability=UseProbability,
                        donormalize=donormalize,
                        lower_perc=lower_perc,
                        upper_perc=upper_perc,
                        dounet=dounet,
                        seedpool=seedpool,
                        slice_merge=slice_merge_nuclei,
                    )
                    for i in tqdm(range(image_nuclei.shape[0]))
                )
            )
        )
        (
            nuclei_sized_smart_seeds,
            muclei_instance_labels,
            nuclei_star_labels,
            nuclei_probability_map,
            nuclei_markers,
            nuclei_skeleton,
        ) = nuclei_res

        nuclei_markers = np.asarray(nuclei_markers)

        nuclei_star_labels = np.asarray(nuclei_star_labels)

        for i in range(nuclei_markers.shape[0]):
            nuclei_markers[i] = clear_border(nuclei_markers[i])

        membrane_res = tuple(
            zip(
                *tuple(
                    VollSeg_unet(
                        image_membrane[i],
                        unet_model=unet_model_membrane,
                        noise_model=noise_model_membrane,
                        roi_model=roi_model,
                        axes=(axes.replace("C", "")).replace("T", ""),
                        min_size_mask=min_size_mask,
                        max_size=max_size,
                        n_tiles=n_tiles,
                        ExpandLabels=ExpandLabels,
                        slice_merge=slice_merge_membrane,
                    )
                    for i in tqdm(range(image_membrane.shape[0]))
                )
            )
        )

        if roi_model is not None:

            (
                membrane_seg,
                membrane_skeleton,
                membrane_denoised,
                membrane_mask,
            ) = membrane_res

        if roi_model is None and noise_model_membrane is not None:

            membrane_seg, membrane_skeleton, membrane_denoised = membrane_res

        membrane_denoised = np.asarray(membrane_denoised)
        membrane_mask = np.asarray(membrane_mask)
        membrane_seg = np.asarray(membrane_seg)
        membrane_z_mask = np.zeros_like(membrane_denoised)
        nuclei_membrane_seg = np.zeros_like(membrane_denoised)
        for i in range(nuclei_markers.shape[0]):

            if roi_model is not None:
                membrane_mask[i] = np.asarray(membrane_mask[i])
                membrane_prop = measure.regionprops(membrane_mask[i].astype(np.uint16))
                membrane_area = np.sum([prop.area for prop in membrane_prop])
                membrane_mask[i] = binary_dilation(membrane_mask[i], iterations=8)
                membrane_z_mask[i] = check_and_update_mask(
                    membrane_mask[i], image_membrane[i]
                )
            properties = measure.regionprops(nuclei_star_labels[i])
            Coordinates = [prop.centroid for prop in properties]

            Coordinates = np.asarray(Coordinates)
            coordinates_int = np.round(Coordinates).astype(int)

            markers_raw = np.zeros_like(membrane_denoised[i])
            markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))
            markers = morphology.dilation(
                markers_raw.astype("uint16"), morphology.ball(2)
            )
            if roi_model is not None:
                membrane_denoised[i] = membrane_denoised[i] * membrane_mask[i]

                nuclei_membrane_seg[i] = (
                    watershed(membrane_denoised[i], markers) * membrane_mask[i]
                )
            else:
                nuclei_membrane_seg[i] = watershed(membrane_denoised[i], markers)
            if roi_model is not None:
                remove_labels = []
                current_nuclei_membrane_seg = nuclei_membrane_seg[i]
                for k in range(current_nuclei_membrane_seg.shape[0]):

                    nuclei_membrane_props = measure.regionprops(
                        current_nuclei_membrane_seg[k].astype(np.uint16)
                    )
                    for prop in nuclei_membrane_props:
                        if prop.area > 0.5 * membrane_area:
                            remove_labels.append(prop.label)
                for remove_label in remove_labels:
                    current_nuclei_membrane_seg[
                        current_nuclei_membrane_seg == remove_label
                    ] = 0
                nuclei_membrane_seg[i] = current_nuclei_membrane_seg
    if len(image.shape) == 4 and "T" not in axes:
        image_membrane = np.take(image, channel_membrane, axis=channel_index)
        image_nuclei = np.take(image, channel_nuclei, axis=channel_index)
        nuclei_res = VollSeg(
            image_nuclei,
            unet_model=unet_model_nuclei,
            star_model=star_model_nuclei,
            axes=axes.replace("C", ""),
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
            min_size_mask=min_size_mask,
            min_size=min_size,
            max_size=max_size,
            n_tiles=n_tiles,
            UseProbability=UseProbability,
            donormalize=donormalize,
            lower_perc=lower_perc,
            upper_perc=upper_perc,
            dounet=dounet,
            seedpool=seedpool,
            slice_merge=slice_merge_nuclei,
        )
        (
            nuclei_sized_smart_seeds,
            muclei_instance_labels,
            nuclei_star_labels,
            nuclei_probability_map,
            nuclei_markers,
            nuclei_skeleton,
        ) = nuclei_res

        nuclei_markers = np.asarray(nuclei_markers)
        nuclei_markers = clear_border(nuclei_markers)
        nuclei_star_labels = np.asarray(nuclei_star_labels)

        membrane_res = VollSeg_unet(
            image_membrane,
            unet_model=unet_model_membrane,
            noise_model=noise_model_membrane,
            roi_model=roi_model,
            axes=axes.replace("C", ""),
            min_size_mask=min_size_mask,
            max_size=max_size,
            n_tiles=n_tiles,
            ExpandLabels=ExpandLabels,
            slice_merge=slice_merge_membrane,
        )
        if roi_model is not None and noise_model_membrane is not None:

            (
                membrane_seg,
                membrane_skeleton,
                membrane_denoised,
                membrane_mask,
            ) = membrane_res

        if roi_model is None and noise_model_membrane is not None:

            membrane_seg, membrane_skeleton, membrane_denoised = membrane_res

        membrane_denoised = np.asarray(membrane_denoised)
        membrane_seg = np.asarray(membrane_seg)
        if roi_model is not None:
            membrane_prop = measure.regionprops(membrane_mask.astype(np.uint16))
            membrane_area = np.sum([prop.area for prop in membrane_prop])

            membrane_mask = np.asarray(membrane_mask)
            membrane_mask = binary_dilation(membrane_mask, iterations=8)
            membrane_mask = check_and_update_mask(membrane_mask, image_membrane)
        properties = measure.regionprops(nuclei_star_labels)
        Coordinates = [prop.centroid for prop in properties]

        Coordinates = np.asarray(Coordinates)
        coordinates_int = np.round(Coordinates).astype(int)

        markers_raw = np.zeros_like(membrane_denoised)
        markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))
        markers = morphology.dilation(markers_raw.astype("uint16"), morphology.ball(2))
        if roi_model is not None:
            membrane_denoised = membrane_denoised * membrane_mask

            nuclei_membrane_seg = watershed(membrane_denoised, markers) * membrane_mask
        else:

            nuclei_membrane_seg = watershed(membrane_denoised, markers)

        if roi_model is not None:
            remove_labels = []
            for i in range(nuclei_membrane_seg.shape[0]):

                nuclei_membrane_props = measure.regionprops(nuclei_membrane_seg[i])
                for prop in nuclei_membrane_props:
                    if prop.area > 0.5 * membrane_area:
                        remove_labels.append(prop.label)
            for remove_label in remove_labels:
                nuclei_membrane_seg[nuclei_membrane_seg == remove_label] = 0

    if len(image.shape) > 5:
        raise NotImplementedError('Please provide a 4D/5D image with axes "TCZYX"')
    if save_dir is not None:
        Path(save_dir).mkdir(exist_ok=True)
        if star_model_nuclei is not None and noise_model_membrane is not None:
            nuclei_labels = Path(save_dir) / "nuclei_labels"
            nuclei_markers_labels = Path(save_dir) / "nuclei_markers"
            membrane_labels = Path(save_dir) / "membrane_labels"
            membrane_denoised_labels = Path(save_dir) / "membrane_denoised"
            nuclei_labels.mkdir(exist_ok=True)
            nuclei_markers_labels.mkdir(exist_ok=True)
            membrane_labels.mkdir(exist_ok=True)
            membrane_denoised_labels.mkdir(exist_ok=True)

            imwrite(
                os.path.join(nuclei_labels.as_posix(), Name + ".tif"),
                nuclei_star_labels.astype("uint16"),
            )

            imwrite(
                os.path.join(nuclei_markers_labels.as_posix(), Name + ".tif"),
                nuclei_markers.astype("uint16"),
            )

            imwrite(
                os.path.join(membrane_denoised_labels.as_posix(), Name + ".tif"),
                membrane_denoised.astype("float32"),
            )

            imwrite(
                os.path.join(membrane_labels.as_posix(), Name + ".tif"),
                nuclei_membrane_seg.astype("uint16"),
            )

            if unet_model_membrane is not None:
                membrane_seg_labels = Path(save_dir) / "pure_membrane_labels"
                membrane_seg_labels.mkdir(exist_ok=True)
                imwrite(
                    os.path.join(membrane_seg_labels.as_posix(), Name + ".tif"),
                    membrane_seg.astype("float32"),
                )

            if roi_model is not None:
                membrane_mask_labels = Path(save_dir) / "region_of_interest"
                membrane_mask_labels.mkdir(exist_ok=True)
                imwrite(
                    os.path.join(membrane_mask_labels.as_posix(), Name + ".tif"),
                    membrane_mask.astype("float32"),
                )

    return nuclei_res, membrane_res


def VollSeg(
    image: np.ndarray,
    unet_model: Union[UNET, None] = None,
    star_model: Union[StarDist3D, None] = None,
    roi_model=None,
    axes="ZYX",
    noise_model=None,
    prob_thresh=None,
    ExpandLabels=False,
    nms_thresh=None,
    min_size_mask=100,
    min_size=100,
    max_size=10000000,
    n_tiles=(1, 1, 1),
    UseProbability=True,
    donormalize=True,
    lower_perc=1,
    upper_perc=99.8,
    dounet=True,
    seedpool=True,
    save_dir=None,
    Name="Result",
    slice_merge=False,
    RGB=False,
):

    if len(image.shape) == 2:

        # if the default tiling of the function is not changed by the user, we use the last two tuples
        if len(n_tiles) == 3:
            n_tiles = (n_tiles[1], n_tiles[2])

        # If stardist model is supplied we use this method
        if star_model is not None:

            res = VollSeg2D(
                image,
                unet_model,
                star_model,
                roi_model=roi_model,
                noise_model=noise_model,
                prob_thresh=prob_thresh,
                nms_thresh=nms_thresh,
                donormalize=donormalize,
                lower_perc=lower_perc,
                upper_perc=upper_perc,
                axes=axes,
                min_size_mask=min_size_mask,
                min_size=min_size,
                max_size=max_size,
                dounet=dounet,
                n_tiles=n_tiles,
                UseProbability=UseProbability,
                RGB=RGB,
                seedpool=seedpool
            )

        # If there is no stardist model we use unet model or denoising model or both to get the semantic segmentation
        if star_model is None:

            res = VollSeg_unet(
                image,
                unet_model=unet_model,
                roi_model=roi_model,
                ExpandLabels=ExpandLabels,
                n_tiles=n_tiles,
                axes=axes,
                min_size_mask=min_size_mask,
                max_size=max_size,
                noise_model=noise_model,
                RGB=RGB,
                nms_thresh=nms_thresh,
                slice_merge=slice_merge,
                dounet=dounet,
            )
    if len(image.shape) == 3 and "T" not in axes and RGB is False:
        # this is a 3D image and if stardist model is supplied we use this method
        if star_model is not None:
            res = VollSeg3D(
                image,
                unet_model,
                star_model,
                roi_model=roi_model,
                ExpandLabels=ExpandLabels,
                axes=axes,
                noise_model=noise_model,
                prob_thresh=prob_thresh,
                nms_thresh=nms_thresh,
                donormalize=donormalize,
                lower_perc=lower_perc,
                upper_perc=upper_perc,
                min_size_mask=min_size_mask,
                min_size=min_size,
                max_size=max_size,
                n_tiles=n_tiles,
                UseProbability=UseProbability,
                dounet=dounet,
                seedpool=seedpool,
                slice_merge=slice_merge,
            )

        # If there is no stardist model we use unet model with or without denoising model
        if star_model is None:

            res = VollSeg_unet(
                image,
                unet_model=unet_model,
                roi_model=roi_model,
                ExpandLabels=ExpandLabels,
                n_tiles=n_tiles,
                axes=axes,
                min_size_mask=min_size_mask,
                max_size=max_size,
                noise_model=noise_model,
                RGB=RGB,
                slice_merge=slice_merge,
                nms_thresh=nms_thresh,
                dounet=dounet,
            )
    if len(image.shape) == 3 and "T" not in axes and RGB:
        # this is a 3D image and if stardist model is supplied we use this method
        if star_model is not None:
            res = VollSeg2D(
                image,
                unet_model,
                star_model,
                roi_model=roi_model,
                noise_model=noise_model,
                prob_thresh=prob_thresh,
                nms_thresh=nms_thresh,
                donormalize=donormalize,
                lower_perc=lower_perc,
                upper_perc=upper_perc,
                axes=axes,
                min_size_mask=min_size_mask,
                min_size=min_size,
                max_size=max_size,
                dounet=dounet,
                n_tiles=n_tiles,
                UseProbability=UseProbability,
                RGB=RGB,
                seedpool=seedpool
            )
        # If there is no stardist model we use unet model with or without denoising model
        if star_model is None:

            res = VollSeg_unet(
                image,
                unet_model=unet_model,
                roi_model=roi_model,
                ExpandLabels=ExpandLabels,
                n_tiles=n_tiles,
                axes=axes,
                min_size_mask=min_size_mask,
                max_size=max_size,
                noise_model=noise_model,
                RGB=RGB,
                slice_merge=slice_merge,
                nms_thresh=nms_thresh,
                dounet=dounet,
            )

    if len(image.shape) == 3 and "T" in axes:
        if len(n_tiles) == 3:
            n_tiles = (n_tiles[1], n_tiles[2])
        if star_model is not None:
            res = tuple(
                zip(
                    *tuple(
                        VollSeg2D(
                            _x,
                            unet_model,
                            star_model,
                            noise_model=noise_model,
                            roi_model=roi_model,
                            prob_thresh=prob_thresh,
                            nms_thresh=nms_thresh,
                            donormalize=donormalize,
                            lower_perc=lower_perc,
                            upper_perc=upper_perc,
                            axes=axes,
                            min_size_mask=min_size_mask,
                            min_size=min_size,
                            max_size=max_size,
                            dounet=dounet,
                            n_tiles=n_tiles,
                            UseProbability=UseProbability,
                            RGB=RGB,
                            seedpool=seedpool
                        )
                        for _x in tqdm(image)
                    )
                )
            )
        if star_model is None:

            res = tuple(
                zip(
                    *tuple(
                        VollSeg_unet(
                            _x,
                            unet_model=unet_model,
                            roi_model=roi_model,
                            ExpandLabels=ExpandLabels,
                            n_tiles=n_tiles,
                            axes=axes,
                            noise_model=noise_model,
                            RGB=RGB,
                            slice_merge=slice_merge,
                            nms_thresh=nms_thresh,
                            dounet=dounet,
                        )
                        for _x in tqdm(image)
                    )
                )
            )

    if len(image.shape) == 4:
        if len(n_tiles) == 4:
            n_tiles = (n_tiles[1], n_tiles[2], n_tiles[3])
        res = tuple(
            zip(
                *tuple(
                    VollSeg3D(
                        _x,
                        unet_model,
                        star_model,
                        axes=axes,
                        noise_model=noise_model,
                        roi_model=roi_model,
                        ExpandLabels=ExpandLabels,
                        prob_thresh=prob_thresh,
                        nms_thresh=nms_thresh,
                        donormalize=donormalize,
                        lower_perc=lower_perc,
                        upper_perc=upper_perc,
                        min_size_mask=min_size_mask,
                        min_size=min_size,
                        max_size=max_size,
                        n_tiles=n_tiles,
                        UseProbability=UseProbability,
                        dounet=dounet,
                        seedpool=seedpool,
                        slice_merge=slice_merge,
                    )
                    for _x in tqdm(image)
                )
            )
        )

    if noise_model is None and star_model is not None and roi_model is not None:
        (
            sized_smart_seeds,
            instance_labels,
            star_labels,
            probability_map,
            markers,
            skeleton,
            roi_image,
        ) = res

    if noise_model is None and star_model is not None and roi_model is None:
        (
            sized_smart_seeds,
            instance_labels,
            star_labels,
            probability_map,
            markers,
            skeleton,
        ) = res

    if noise_model is not None and star_model is not None and roi_model is not None:
        (
            sized_smart_seeds,
            instance_labels,
            star_labels,
            probability_map,
            markers,
            skeleton,
            image,
            roi_image,
        ) = res

    if noise_model is not None and star_model is not None and roi_model is None:
        (
            sized_smart_seeds,
            instance_labels,
            star_labels,
            probability_map,
            markers,
            skeleton,
            image,
        ) = res

    if (
        noise_model is not None
        and star_model is None
        and roi_model is None
        and unet_model is None
    ):

        instance_labels, skeleton, image = res

    if (
        star_model is None
        and roi_model is not None
        and unet_model is not None
        and noise_model is not None
    ):

        instance_labels, skeleton, image, roi_image = res

    if (
        star_model is None
        and roi_model is None
        and unet_model is not None
        and noise_model is not None
    ):

        instance_labels, skeleton, image = res

    if (
        star_model is None
        and roi_model is None
        and unet_model is not None
        and noise_model is None
    ):

        instance_labels, skeleton, image = res

    if (
        star_model is None
        and roi_model is not None
        and unet_model is None
        and noise_model is None
    ):

        instance_labels, roi_image = res
        skeleton = Skel(instance_labels)

    if (
        star_model is None
        and roi_model is not None
        and unet_model is None
        and noise_model is not None
    ):

        instance_labels, skeleton, image, roi_image = res

    if (
        star_model is None
        and roi_model is not None
        and unet_model is not None
        and noise_model is None
    ):

        instance_labels, skeleton, image, roi_image = res

    if save_dir is not None:
        Path(save_dir).mkdir(exist_ok=True)

        if roi_model is not None:
            roi_results = Path(save_dir) / "Roi"
            Path(roi_results).mkdir(exist_ok=True)
            imwrite(
                os.path.join(roi_results.as_posix(), Name + ".tif"),
                np.asarray(roi_image).astype("uint16"),
            )

        if unet_model is not None:
            unet_results = Path(save_dir) / "BinaryMask"
            skel_unet_results = Path(save_dir) / "Skeleton"
            Path(unet_results).mkdir(exist_ok=True)
            Path(skel_unet_results).mkdir(exist_ok=True)

            imwrite(
                os.path.join(unet_results.as_posix(), Name + ".tif"),
                np.asarray(instance_labels).astype("uint16"),
            )
            imwrite(
                os.path.join(skel_unet_results.as_posix(), Name + ".tif"),
                np.asarray(skeleton).astype("uint16"),
            )
        if star_model is not None:
            vollseg_results = Path(save_dir) / "VollSeg"
            stardist_results = Path(save_dir) / "StarDist"
            probability_results = Path(save_dir) / "Probability"
            marker_results = Path(save_dir) / "Markers"
            skel_results = Path(save_dir) / "Skeleton"
            Path(skel_results).mkdir(exist_ok=True)
            Path(vollseg_results).mkdir(exist_ok=True)
            Path(stardist_results).mkdir(exist_ok=True)
            Path(probability_results).mkdir(exist_ok=True)
            Path(marker_results).mkdir(exist_ok=True)
            imwrite(
                os.path.join(stardist_results.as_posix(), Name + ".tif"),
                np.asarray(star_labels).astype("uint16"),
            )
            imwrite(
                os.path.join(vollseg_results.as_posix(), Name + ".tif"),
                np.asarray(sized_smart_seeds).astype("uint16"),
            )
            imwrite(
                os.path.join(probability_results.as_posix(), Name + ".tif"),
                np.asarray(probability_map).astype("float32"),
            )
            imwrite(
                os.path.join(marker_results.as_posix(), Name + ".tif"),
                np.asarray(markers).astype("uint16"),
            )
            imwrite(
                os.path.join(skel_results.as_posix(), Name + ".tif"),
                np.asarray(skeleton),
            )
        if noise_model is not None:
            denoised_results = Path(save_dir) / "Denoised"
            Path(denoised_results).mkdir(exist_ok=True)
            imwrite(
                os.path.join(denoised_results.as_posix(), Name + ".tif"),
                np.asarray(image).astype("float32"),
            )

    # If denoising is not done but stardist and unet models are supplied we return the stardist, vollseg and semantic segmentation maps
    if noise_model is None and star_model is not None and roi_model is not None:

        return (
            sized_smart_seeds,
            instance_labels,
            star_labels,
            probability_map,
            markers,
            skeleton,
            roi_image,
        )

    if noise_model is None and star_model is not None and roi_model is None:

        return (
            sized_smart_seeds,
            instance_labels,
            star_labels,
            probability_map,
            markers,
            skeleton,
        )

    # If denoising is done and stardist and unet models are supplied we return the stardist, vollseg, denoised image and semantic segmentation maps
    if noise_model is not None and star_model is not None and roi_model is not None:

        return (
            sized_smart_seeds,
            instance_labels,
            star_labels,
            probability_map,
            markers,
            skeleton,
            image,
            roi_image,
        )

    if noise_model is not None and star_model is not None and roi_model is None:

        return (
            sized_smart_seeds,
            instance_labels,
            star_labels,
            probability_map,
            markers,
            skeleton,
            image,
        )

    # If the stardist model is not supplied but only the unet and noise model we return the denoised result and the semantic segmentation map
    if star_model is None and noise_model is not None:

        return instance_labels, skeleton, image

    if star_model is None and roi_model is not None and noise_model is None:

        return instance_labels, roi_image

    if star_model is None and roi_model is not None and noise_model is not None:

        return instance_labels, roi_image, image

    if (
        noise_model is not None
        and star_model is None
        and roi_model is None
        and unet_model is None
    ):

        return instance_labels, skeleton, image

    if (
        star_model is None
        and roi_model is None
        and noise_model is None
        and unet_model is not None
    ):

        return instance_labels, skeleton


def VollSeg3D(
    image,
    unet_model,
    star_model,
    axes="ZYX",
    noise_model=None,
    roi_model=None,
    prob_thresh=None,
    nms_thresh=None,
    min_size_mask=100,
    min_size=100,
    max_size=10000000,
    n_tiles=(1, 2, 2),
    UseProbability=True,
    ExpandLabels=True,
    dounet=True,
    seedpool=True,
    donormalize=True,
    lower_perc=1,
    upper_perc=99.8,
    slice_merge=False,
):

    sizeZ = image.shape[0]
    sizeY = image.shape[1]
    sizeX = image.shape[2]
    if len(n_tiles) >= len(image.shape):
        n_tiles = (n_tiles[-3], n_tiles[-2], n_tiles[-1])
    else:
        tiles = n_tiles
    instance_labels = np.zeros([sizeZ, sizeY, sizeX], dtype="uint16")

    sized_smart_seeds = np.zeros([sizeZ, sizeY, sizeX], dtype="uint16")
    sized_probability_map = np.zeros([sizeZ, sizeY, sizeX], dtype="float32")
    sized_markers = np.zeros([sizeZ, sizeY, sizeX], dtype="uint16")
    sized_stardist = np.zeros([sizeZ, sizeY, sizeX], dtype="uint16")
    Mask = None
    Mask_patch = None
    roi_image = None
    if noise_model is not None:

        image = noise_model.predict(image.astype("float32"), axes=axes, n_tiles=n_tiles)
        pixel_condition = image < 0
        pixel_replace_condition = 0
        image = image_conditionals(image, pixel_condition, pixel_replace_condition)

    if roi_model is not None:

        model_dim = roi_model.config.n_dim
        if model_dim < len(image.shape):
            if len(n_tiles) >= len(image.shape):
                tiles = (n_tiles[-2], n_tiles[-1])
            else:
                tiles = n_tiles
            maximage = np.amax(image, axis=0)
            Segmented = roi_model.predict(
                maximage.astype("float32"), "YX", n_tiles=tiles
            )
            try:
                thresholds = threshold_multiotsu(Segmented, classes=2)

                # Using the threshold values, we generate the three regions.
                regions = np.digitize(Segmented, bins=thresholds)
            except ValueError:

                regions = Segmented

            roi_image = regions > 0
            roi_image = label(roi_image)
            roi_bbox = Bbox_region(roi_image)
            if roi_bbox is not None:
                rowstart = roi_bbox[0]
                colstart = roi_bbox[1]
                endrow = roi_bbox[2]
                endcol = roi_bbox[3]
                region = (
                    slice(0, image.shape[0]),
                    slice(rowstart, endrow),
                    slice(colstart, endcol),
                )
            else:
                region = (
                    slice(0, image.shape[0]),
                    slice(0, image.shape[1]),
                    slice(0, image.shape[2]),
                )
                rowstart = 0
                colstart = 0
                endrow = image.shape[2]
                endcol = image.shape[1]
                roi_bbox = [colstart, rowstart, endcol, endrow]
        elif model_dim == len(image.shape):
            Segmented = roi_model.predict(
                image.astype("float32"), "ZYX", n_tiles=n_tiles
            )
            try:
                thresholds = threshold_multiotsu(Segmented, classes=2)

                # Using the threshold values, we generate the three regions.
                regions = np.digitize(Segmented, bins=thresholds)
            except ValueError:

                regions = Segmented

            roi_image = regions > 0
            roi_image = label(roi_image)
            roi_bbox = Bbox_region(roi_image)
            if roi_bbox is not None:
                zstart = roi_bbox[0]
                rowstart = roi_bbox[1]
                colstart = roi_bbox[2]
                zend = roi_bbox[3]
                endrow = roi_bbox[4]
                endcol = roi_bbox[5]
                region = (
                    slice(zstart, zend),
                    slice(rowstart, endrow),
                    slice(colstart, endcol),
                )
            else:

                region = (
                    slice(0, image.shape[0]),
                    slice(0, image.shape[1]),
                    slice(0, image.shape[2]),
                )
                rowstart = 0
                colstart = 0
                endrow = image.shape[2]
                endcol = image.shape[1]
                roi_bbox = [colstart, rowstart, endcol, endrow]

        # The actual pixels in that region.
        if roi_bbox is not None:
            patch = image[region]

        else:
            patch = image

    else:

        patch = image

        region = (
            slice(0, image.shape[0]),
            slice(0, image.shape[1]),
            slice(0, image.shape[2]),
        )
        rowstart = 0
        colstart = 0
        endrow = image.shape[2]
        endcol = image.shape[1]
        roi_bbox = [colstart, rowstart, endcol, endrow]

    if dounet:

        gc.collect()
        if unet_model is not None:

            Mask = UNETPrediction3D(
                patch,
                unet_model,
                n_tiles,
                axes,
                nms_thresh=nms_thresh,
                slice_merge=slice_merge,
                ExpandLabels=ExpandLabels,
            )
            for i in range(0, Mask.shape[0]):
                Mask[i] = remove_small_objects(
                    Mask[i].astype("uint16"), min_size=min_size_mask
                )
                Mask[i] = remove_big_objects(
                    Mask[i].astype("uint16"), max_size=max_size
                )
            Mask_patch = Mask.copy()
            Mask = Region_embedding(image, roi_bbox, Mask, dtype=np.uint8)
            if slice_merge:
                Mask = match_labels(Mask.astype("uint16"), nms_thresh=nms_thresh)
            else:
                Mask = label(Mask > 0)
            instance_labels[:, : Mask.shape[1], : Mask.shape[2]] = Mask

    elif noise_model is not None and dounet is False:

        Mask = np.zeros(patch.shape)

        for i in range(0, Mask.shape[0]):

            try:
                thresholds = threshold_multiotsu(patch[i, :], classes=2)

                # Using the threshold values, we generate the three regions.
                regions = np.digitize(patch[i], bins=thresholds)

            except ValueError:

                regions = patch[i]
            Mask[i] = regions > 0
            Mask[i] = label(Mask[i, :])

            Mask[i] = remove_small_objects(
                Mask[i].astype("uint16"), min_size=min_size_mask
            )
            Mask[i] = remove_big_objects(Mask[i].astype("uint16"), max_size=max_size)
        if slice_merge:
            Mask = match_labels(Mask, nms_thresh=nms_thresh)
        else:
            Mask = label(Mask > 0)
        Mask_patch = Mask.copy()
        Mask = Region_embedding(image, roi_bbox, Mask, dtype=np.uint8)
        instance_labels[:, : Mask.shape[1], : Mask.shape[2]] = Mask

    if star_model is not None:
        if donormalize:

            patch_star = normalize(patch, lower_perc, upper_perc, axis=(0, 1, 2))
        else:
            patch_star = patch

        smart_seeds, probability_map, star_labels, markers = STARPrediction3D(
            patch_star,
            axes,
            star_model,
            n_tiles,
            unet_mask=Mask_patch,
            UseProbability=UseProbability,
            seedpool=seedpool,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
        )

        smart_seeds = Region_embedding(image, roi_bbox, smart_seeds, dtype=np.uint16)
        sized_smart_seeds[
            :, : smart_seeds.shape[1], : smart_seeds.shape[2]
        ] = smart_seeds
        markers = Region_embedding(image, roi_bbox, markers, dtype=np.uint16)
        sized_markers[:, : smart_seeds.shape[1], : smart_seeds.shape[2]] = markers
        probability_map = Region_embedding(image, roi_bbox, probability_map)
        sized_probability_map[
            :, : probability_map.shape[1], : probability_map.shape[2]
        ] = probability_map
        star_labels = Region_embedding(image, roi_bbox, star_labels, dtype=np.uint16)
        sized_stardist[:, : star_labels.shape[1], : star_labels.shape[2]] = star_labels
        skeleton = np.zeros_like(sized_smart_seeds)
        for i in range(0, sized_smart_seeds.shape[0]):
            skeleton[i] = SmartSkel(sized_smart_seeds[i], sized_probability_map[i])
        skeleton = skeleton > 0

    if noise_model is None and roi_image is not None and star_model is not None:
        return (
            sized_smart_seeds.astype("uint16"),
            instance_labels.astype("uint16"),
            star_labels.astype("uint16"),
            sized_probability_map,
            markers.astype("uint16"),
            skeleton.astype("uint16"),
            roi_image.astype("uint16"),
        )
    if noise_model is None and roi_image is None and star_model is not None:
        return (
            sized_smart_seeds.astype("uint16"),
            instance_labels.astype("uint16"),
            star_labels.astype("uint16"),
            sized_probability_map,
            markers.astype("uint16"),
            skeleton.astype("uint16"),
        )
    if noise_model is not None and roi_image is None and star_model is not None:
        return (
            sized_smart_seeds.astype("uint16"),
            instance_labels.astype("uint16"),
            star_labels.astype("uint16"),
            sized_probability_map,
            markers.astype("uint16"),
            skeleton.astype("uint16"),
            image,
        )
    if noise_model is not None and roi_image is not None and star_model is not None:
        return (
            sized_smart_seeds.astype("uint16"),
            instance_labels.astype("uint16"),
            star_labels.astype("uint16"),
            sized_probability_map,
            markers.astype("uint16"),
            skeleton.astype("uint16"),
            image,
            roi_image.astype("uint16"),
        )

    if noise_model is not None and roi_image is not None and star_model is None:
        return instance_labels.astype("uint16"), skeleton, image

    if (
        noise_model is not None
        and roi_image is None
        and star_model is None
        and unet_model is None
    ):
        return image

    if (
        noise_model is None
        and roi_image is None
        and star_model is None
        and unet_model is not None
    ):
        return instance_labels.astype("uint16"), skeleton


def return_masks(instance_labels, image, min_size, max_size):

    if len(instance_labels) > 0:
        segmentation_image = instance_labels[0]["segmentation"].astype("uint16")
        segmentation_image[np.where(segmentation_image > 0)] = 2
        for i in range(1, len(instance_labels)):
            m = instance_labels[i]["segmentation"].astype("uint16")
            m[np.where(m > 0)] = 2 + i

            segmentation_image = join_segmentations(segmentation_image, m)
            segmentation_image, _, _ = relabel_sequential(
                segmentation_image, offset=segmentation_image.max()
            )
        segmentation_image = remove_small_objects(
            segmentation_image.astype("uint16"), min_size=min_size
        )
        segmentation_image = remove_big_objects(
            segmentation_image.astype("uint16"), max_size=max_size
        )

        return segmentation_image
    else:

        return np.zeros_like(image[..., 0])


def SuperVollSeg(
    image_nuclei: np.ndarray,
    unet_model_nuclei: UNET,
    star_model_nuclei: StarDist3D,
    axes="ZYX",
    noise_model: CARE = None,
    roi_model_nuclei: MASKUNET = None,
    prob_thresh_nuclei: float = None,
    nms_thresh_nuclei: float = None,
    min_size_mask: int = 100,
    min_size: int = 100,
    max_size: int = 10000000,
    n_tiles: tuple = (1, 2, 2),
    UseProbability: bool = True,
    ExpandLabels: bool = True,
    dounet: bool = True,
    seedpool: bool = True,
    donormalize: bool = True,
    lower_perc: float = 1,
    upper_perc: float = 99.8,
    slice_merge: bool = False,
):

    sizeZ = image_nuclei.shape[0]
    sizeY = image_nuclei.shape[1]
    sizeX = image_nuclei.shape[2]
    if len(n_tiles) >= len(image_nuclei.shape):
        n_tiles = (n_tiles[-3], n_tiles[-2], n_tiles[-1])
    else:
        tiles = n_tiles
    instance_labels_nuclei = np.zeros([sizeZ, sizeY, sizeX], dtype="uint16")
    sized_smart_seeds_nuclei = np.zeros([sizeZ, sizeY, sizeX], dtype="uint16")
    sized_probability_map_nuclei = np.zeros([sizeZ, sizeY, sizeX], dtype="float32")
    sized_markers_nuclei = np.zeros([sizeZ, sizeY, sizeX], dtype="uint16")
    sized_stardist_nuclei = np.zeros([sizeZ, sizeY, sizeX], dtype="uint16")

    Mask_nuclei_patch = None
    roi_image = None
    if noise_model is not None:

        image_nuclei = noise_model.predict(
            image_nuclei.astype("float32"), axes=axes, n_tiles=n_tiles
        )
        pixel_condition = image_nuclei < 0
        pixel_replace_condition = 0
        image_nuclei = image_conditionals(
            image_nuclei, pixel_condition, pixel_replace_condition
        )

    if roi_model_nuclei is not None:

        model_dim = roi_model_nuclei.config.n_dim
        if model_dim < len(image_nuclei.shape):
            if len(n_tiles) >= len(image_nuclei.shape):
                tiles = (n_tiles[-2], n_tiles[-1])
            else:
                tiles = n_tiles
            maximage = np.amax(image_nuclei, axis=0)
            Segmented = roi_model_nuclei.predict(
                maximage.astype("float32"), "YX", n_tiles=tiles
            )
            try:
                thresholds = threshold_multiotsu(Segmented, classes=2)

                # Using the threshold values, we generate the three regions.
                regions = np.digitize(Segmented, bins=thresholds)
            except ValueError:

                regions = Segmented

            roi_image = regions > 0
            roi_image = label(roi_image)
            roi_bbox = Bbox_region(roi_image)
            if roi_bbox is not None:
                rowstart = roi_bbox[0]
                colstart = roi_bbox[1]
                endrow = roi_bbox[2]
                endcol = roi_bbox[3]
                region = (
                    slice(0, image_nuclei.shape[0]),
                    slice(rowstart, endrow),
                    slice(colstart, endcol),
                )
            else:
                region = (
                    slice(0, image_nuclei.shape[0]),
                    slice(0, image_nuclei.shape[1]),
                    slice(0, image_nuclei.shape[2]),
                )
                rowstart = 0
                colstart = 0
                endrow = image_nuclei.shape[2]
                endcol = image_nuclei.shape[1]
                roi_bbox = [colstart, rowstart, endcol, endrow]
        elif model_dim == len(image_nuclei.shape):
            Segmented = roi_model_nuclei.predict(
                maximage.astype("float32"), "YX", n_tiles=n_tiles
            )
            try:
                thresholds = threshold_multiotsu(Segmented, classes=2)

                # Using the threshold values, we generate the three regions.
                regions = np.digitize(Segmented, bins=thresholds)
            except ValueError:

                regions = Segmented

            roi_image = regions > 0
            roi_image = label(roi_image)
            roi_bbox = Bbox_region(roi_image)
            if roi_bbox is not None:
                zstart = roi_bbox[0]
                rowstart = roi_bbox[1]
                colstart = roi_bbox[2]
                zend = roi_bbox[3]
                endrow = roi_bbox[4]
                endcol = roi_bbox[5]
                region = (
                    slice(zstart, zend),
                    slice(rowstart, endrow),
                    slice(colstart, endcol),
                )
            else:

                region = (
                    slice(0, image_nuclei.shape[0]),
                    slice(0, image_nuclei.shape[1]),
                    slice(0, image_nuclei.shape[2]),
                )
                rowstart = 0
                colstart = 0
                endrow = image_nuclei.shape[2]
                endcol = image_nuclei.shape[1]
                roi_bbox = [colstart, rowstart, endcol, endrow]

        # The actual pixels in that region.
        if roi_bbox is not None:
            patch_nuclei = image_nuclei[region]

        else:
            patch_nuclei = image_nuclei

    else:

        patch_nuclei = image_nuclei

        region = (
            slice(0, image_nuclei.shape[0]),
            slice(0, image_nuclei.shape[1]),
            slice(0, image_nuclei.shape[2]),
        )
        rowstart = 0
        colstart = 0
        endrow = image_nuclei.shape[2]
        endcol = image_nuclei.shape[1]
        roi_bbox = [colstart, rowstart, endcol, endrow]

    if dounet:

        gc.collect()

        if unet_model_nuclei is not None:
            Mask_nuclei = UNETPrediction3D(
                patch_nuclei,
                unet_model_nuclei,
                n_tiles,
                axes,
                nms_thresh=nms_thresh_nuclei,
                slice_merge=slice_merge,
                ExpandLabels=ExpandLabels,
            )
            for i in range(0, Mask_nuclei.shape[0]):
                Mask_nuclei[i] = remove_small_objects(
                    Mask_nuclei[i].astype("uint16"), min_size=min_size_mask
                )
                Mask_nuclei[i] = remove_big_objects(
                    Mask_nuclei[i].astype("uint16"), max_size=max_size
                )
            Mask_nuclei_patch = Mask_nuclei.copy()
            Mask_nuclei = Region_embedding(
                image_nuclei, roi_bbox, Mask_nuclei, dtype=np.uint8
            )
            if slice_merge:
                Mask_nuclei = match_labels(
                    Mask_nuclei.astype("uint16"),
                    nms_thresh=nms_thresh_nuclei,
                )
            else:
                Mask_nuclei = label(Mask_nuclei > 0)
            instance_labels_nuclei[
                :, : Mask_nuclei.shape[1], : Mask_nuclei.shape[2]
            ] = Mask_nuclei

    elif noise_model is not None and dounet is False:

        Mask_nuclei = np.zeros(patch_nuclei.shape)

        for i in range(0, Mask_nuclei.shape[0]):

            try:
                thresholds = threshold_multiotsu(patch_nuclei[i, :], classes=2)

                # Using the threshold values, we generate the three regions.
                regions = np.digitize(patch_nuclei[i], bins=thresholds)

            except ValueError:

                regions = patch_nuclei[i]
            Mask_nuclei[i] = regions > 0
            Mask_nuclei[i] = label(Mask_nuclei[i, :])

            Mask_nuclei[i] = remove_small_objects(
                Mask_nuclei[i].astype("uint16"), min_size=min_size_mask
            )
            Mask_nuclei[i] = remove_big_objects(
                Mask_nuclei[i].astype("uint16"), max_size=max_size
            )
        if slice_merge:
            Mask_nuclei = match_labels(Mask_nuclei, nms_thresh=nms_thresh_nuclei)
        else:
            Mask_nuclei = label(Mask_nuclei > 0)
        Mask_nuclei_patch = Mask_nuclei.copy()
        Mask_nuclei = Region_embedding(
            image_nuclei, roi_bbox, Mask_nuclei, dtype=np.uint8
        )
        instance_labels_nuclei[
            :, : Mask_nuclei.shape[1], : Mask_nuclei.shape[2]
        ] = Mask_nuclei

    if star_model_nuclei is not None:
        if donormalize:

            patch_star_nuclei = normalize(
                patch_nuclei, lower_perc, upper_perc, axis=(0, 1, 2)
            )
        else:
            patch_star_nuclei = patch_nuclei

        (
            smart_seeds_nuclei,
            probability_map_nuclei,
            star_labels_nuclei,
            markers_nuclei,
        ) = STARPrediction3D(
            patch_star_nuclei,
            axes,
            star_model_nuclei,
            n_tiles,
            unet_mask=Mask_nuclei_patch,
            UseProbability=UseProbability,
            seedpool=seedpool,
            prob_thresh=prob_thresh_nuclei,
            nms_thresh=nms_thresh_nuclei,
        )

        smart_seeds_nuclei = Region_embedding(
            image_nuclei, roi_bbox, smart_seeds_nuclei, dtype=np.uint16
        )
        sized_smart_seeds_nuclei[
            :, : smart_seeds_nuclei.shape[1], : smart_seeds_nuclei.shape[2]
        ] = smart_seeds_nuclei
        markers_nuclei = Region_embedding(
            image_nuclei, roi_bbox, markers_nuclei, dtype=np.uint16
        )
        sized_markers_nuclei[
            :, : smart_seeds_nuclei.shape[1], : smart_seeds_nuclei.shape[2]
        ] = markers_nuclei
        probability_map_nuclei = Region_embedding(
            image_nuclei, roi_bbox, probability_map_nuclei
        )
        sized_probability_map_nuclei[
            :,
            : probability_map_nuclei.shape[1],
            : probability_map_nuclei.shape[2],
        ] = probability_map_nuclei
        star_labels_nuclei = Region_embedding(
            image_nuclei, roi_bbox, star_labels_nuclei, dtype=np.uint16
        )
        sized_stardist_nuclei[
            :, : star_labels_nuclei.shape[1], : star_labels_nuclei.shape[2]
        ] = star_labels_nuclei

    if noise_model is None and roi_image is not None and star_model_nuclei is not None:
        return (
            sized_smart_seeds_nuclei.astype("uint16"),
            instance_labels_nuclei.astype("uint16"),
            star_labels_nuclei.astype("uint16"),
            sized_probability_map_nuclei,
            markers_nuclei.astype("uint16"),
            roi_image.astype("uint16"),
        )
    if noise_model is None and roi_image is None and star_model_nuclei is not None:
        return (
            sized_smart_seeds_nuclei.astype("uint16"),
            instance_labels_nuclei.astype("uint16"),
            star_labels_nuclei.astype("uint16"),
            sized_probability_map_nuclei,
            markers_nuclei.astype("uint16"),
        )
    if noise_model is not None and roi_image is None and star_model_nuclei is not None:
        return (
            sized_smart_seeds_nuclei.astype("uint16"),
            instance_labels_nuclei.astype("uint16"),
            star_labels_nuclei.astype("uint16"),
            sized_probability_map_nuclei,
            markers_nuclei.astype("uint16"),
            image_nuclei,
        )
    if (
        noise_model is not None
        and roi_image is not None
        and star_model_nuclei is not None
    ):
        return (
            sized_smart_seeds_nuclei.astype("uint16"),
            instance_labels_nuclei.astype("uint16"),
            star_labels_nuclei.astype("uint16"),
            sized_probability_map_nuclei,
            markers_nuclei.astype("uint16"),
            image_nuclei,
            roi_image.astype("uint16"),
        )

    if noise_model is not None and roi_image is not None and star_model_nuclei is None:
        return (
            instance_labels_nuclei.astype("uint16"),
            image_nuclei,
            roi_image.astype("uint16"),
        )

    if (
        noise_model is not None
        and roi_image is None
        and star_model_nuclei is None
        and unet_model_nuclei is None
    ):
        return (
            instance_labels_nuclei.astype("uint16"),
            image_nuclei,
        )

    if (
        noise_model is None
        and roi_image is None
        and star_model_nuclei is None
        and unet_model_nuclei is not None
    ):
        return instance_labels_nuclei.astype("uint16")


def image_pixel_duplicator(image, size):

    assert len(image.shape) == len(
        size
    ), f"The provided size {len(size)} should match the image dimensions {len(image.shape)}"

    model_dim = len(size)

    if model_dim == 3:
        size_y = size[0]
        size_x = size[1]
        size_z = size[2]
        if size_y <= image.shape[0]:
            size_y = image.shape[0]
        if size_x <= image.shape[1]:
            size_x = image.shape[1]
        if size_z <= image.shape[2]:
            size_z = image.shape[2]

        size = (size_y, size_x, size_z)
        ResizeImage = np.zeros(size)
        j = 0
        for i in range(0, ResizeImage.shape[1]):

            if j < image.shape[1]:
                ResizeImage[: image.shape[0], i, : image.shape[2]] = image[
                    : image.shape[0], j, : image.shape[2]
                ]
                j = j + 1
            else:
                j = 0

        j = 0
        for i in range(0, ResizeImage.shape[2]):

            if j < image.shape[2]:
                ResizeImage[:, :, i] = ResizeImage[:, :, j]
                j = j + 1
            else:
                j = 0

        j = 0
        for i in range(0, ResizeImage.shape[0]):

            if j < image.shape[0]:
                ResizeImage[i, :, :] = ResizeImage[j, :, :]
                j = j + 1
            else:
                j = 0

    if model_dim == 2:

        size_y = size[0]
        size_x = size[1]
        if size_y <= image.shape[0]:
            size_y = image.shape[0]
        if size_x <= image.shape[1]:
            size_x = image.shape[1]

        size = (size_y, size_x)

        ResizeImage = np.zeros(size)
        j = 0
        for i in range(0, ResizeImage.shape[1]):

            if j < image.shape[1]:
                ResizeImage[: image.shape[0], i] = image[: image.shape[0], j]
                j = j + 1
            else:
                j = 0

        j = 0
        for i in range(0, ResizeImage.shape[0]):

            if j < image.shape[0]:
                ResizeImage[i, :] = ResizeImage[j, :]
                j = j + 1
            else:
                j = 0

    return ResizeImage


def image_conditionals(image, pixel_condition, pixel_replace_condition):

    indices = zip(*np.where(pixel_condition))
    for index in indices:

        image[index] = pixel_replace_condition

    return image


def image_addition_conditionals(image, pixel_condition, pixel_replace_condition):

    indices = zip(*np.where(pixel_condition))
    for index in indices:

        image[index] = image[index] + pixel_replace_condition

    return image


def image_embedding(image, size):

    model_dim = len(image.shape)
    if model_dim == 2:
        assert len(image.shape) == len(
            size
        ), f"The provided size {len(size)} should match the image dimensions {len(image.shape)}"
        for i in range(len(size)):
            assert (
                image.shape[i] <= size[i]
            ), f"The image size should be smaller \
            than the volume it is to be embedded in but found image of size {image.shape[i]} for dimension{i}"
            width = []
            for i in range(len(size)):
                width.append(size[i] - image.shape[i])
            width = np.asarray(width)

            ResizeImage = np.pad(image, width, "constant", constant_values=0)
    if model_dim == 3:
        ResizeImage = []
        width = []
        for i in range(len(size)):
            width.append(size[i] - image.shape[i + 1])
        width = np.asarray(width)
        for i in range(image.shape[0]):

            ResizeImage.append(
                np.pad(image[i, :], width, "constant", constant_values=0)
            )
        ResizeImage = np.asarray(ResizeImage)
    return ResizeImage


def Integer_to_border(Label):

    BoundaryLabel = find_boundaries(Label, mode="outer")

    Binary = BoundaryLabel > 0

    return Binary


def SuperUNETPrediction(image, model, n_tiles, axis):

    Segmented = model.predict(image.astype("float32"), axis, n_tiles=n_tiles)

    try:
        thresholds = threshold_multiotsu(Segmented, classes=2)

        # Using the threshold values, we generate the three regions.
        regions = np.digitize(Segmented, bins=thresholds)
    except ValueError:

        regions = Segmented

    Binary = regions > 0
    Finalimage = label(Binary)

    Finalimage = relabel_sequential(Finalimage)[0]

    return Finalimage


def stitch3D(masks, stitch_threshold=0.25):
    """stitch 2D masks into 3D volume with stitch_threshold on IOU"""
    mmax = masks[0].max()
    empty = 0

    for i in range(len(masks) - 1):
        if masks[i].max() > 0 and masks[i + 1].max() > 0:
            iou = _intersection_over_union(masks[i + 1], masks[i])[1:, 1:]
            if not iou.size and empty == 0:
                masks[i + 1] = masks[i + 1]
                mmax = masks[i + 1].max()
            elif not iou.size and not empty == 0:
                icount = masks[i + 1].max()
                istitch = np.arange(mmax + 1, mmax + icount + 1, 1, int)
                mmax += icount
                istitch = np.append(np.array(0), istitch)
                masks[i + 1] = istitch[masks[i + 1]]
            else:
                iou[iou < stitch_threshold] = 0.0
                iou[iou < iou.max(axis=0)] = 0.0
                istitch = iou.argmax(axis=1) + 1
                ino = np.nonzero(iou.max(axis=1) == 0.0)[0]
                istitch[ino] = np.arange(mmax + 1, mmax + len(ino) + 1, 1, int)
                mmax += len(ino)
                istitch = np.append(np.array(0), istitch)
                masks[i + 1] = istitch[masks[i + 1]]
                empty = 1

    return masks


def mask_ious(masks_true, masks_pred):
    """return best-matched masks"""
    iou = _intersection_over_union(masks_true, masks_pred)[1:, 1:]
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= 0.5).astype(float) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    iout = np.zeros(masks_true.max())
    iout[true_ind] = iou[true_ind, pred_ind]
    preds = np.zeros(masks_true.max(), "int")
    preds[true_ind] = pred_ind + 1
    return iout, preds


def boundary_scores(masks_true, masks_pred, scales):
    """boundary precision / recall / Fscore"""
    diams = [diameters(lbl)[0] for lbl in masks_true]
    precision = np.zeros((len(scales), len(masks_true)))
    recall = np.zeros((len(scales), len(masks_true)))
    fscore = np.zeros((len(scales), len(masks_true)))
    for j, scale in enumerate(scales):
        for n in range(len(masks_true)):
            diam = max(1, scale * diams[n])
            rs, ys, xs = circleMask([int(np.ceil(diam)), int(np.ceil(diam))])
            filt = (rs <= diam).astype(np.float32)
            otrue = masks_to_outlines(masks_true[n])
            otrue = convolve(otrue, filt)
            opred = masks_to_outlines(masks_pred[n])
            opred = convolve(opred, filt)
            tp = np.logical_and(otrue == 1, opred == 1).sum()
            fp = np.logical_and(otrue == 0, opred == 1).sum()
            fn = np.logical_and(otrue == 1, opred == 0).sum()
            precision[j, n] = tp / (tp + fp)
            recall[j, n] = tp / (tp + fn)
        fscore[j] = 2 * precision[j] * recall[j] / (precision[j] + recall[j])
    return precision, recall, fscore


def masks_to_outlines(masks):
    """get outlines of masks as a 0-1 array

    Parameters
    ----------------

    masks: int, 2D or 3D array
        size [Ly x Lx] or [Lz x Ly x Lx], 0=NO masks; 1,2,...=mask labels

    Returns
    ----------------

    outlines: 2D or 3D array
        size [Ly x Lx] or [Lz x Ly x Lx], True pixels are outlines

    """
    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError(
            "masks_to_outlines takes 2D or 3D array, not %dD array" % masks.ndim
        )
    outlines = np.zeros(masks.shape, bool)

    if masks.ndim == 3:
        for i in range(masks.shape[0]):
            outlines[i] = masks_to_outlines(masks[i])
        return outlines
    else:
        slices = find_objects(masks.astype(int))
        for i, si in enumerate(slices):
            if si is not None:
                sr, sc = si
                mask = (masks[sr, sc] == (i + 1)).astype(np.uint8)
                contours = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                pvc, pvr = np.concatenate(contours[-2], axis=0).squeeze().T
                vr, vc = pvr + sr.start, pvc + sc.start
                outlines[vr, vc] = 1
        return outlines


def diameters(masks):
    _, counts = np.unique(np.int32(masks), return_counts=True)
    counts = counts[1:]
    md = np.median(counts**0.5)
    if np.isnan(md):
        md = 0
    md /= (np.pi**0.5) / 2
    return md, counts**0.5


def circleMask(d0):
    """creates array with indices which are the radius of that x,y point
    inputs:
        d0 (patch of (-d0,d0+1) over which radius computed
    outputs:
        rs: array (2*d0+1,2*d0+1) of radii
        dx,dy: indices of patch
    """
    dx = np.tile(np.arange(-d0[1], d0[1] + 1), (2 * d0[0] + 1, 1))
    dy = np.tile(np.arange(-d0[0], d0[0] + 1), (2 * d0[1] + 1, 1))
    dy = dy.transpose()

    rs = (dy**2 + dx**2) ** 0.5
    return rs, dx, dy


def aggregated_jaccard_index(masks_true, masks_pred):
    """AJI = intersection of all matched masks / union of all masks

    Parameters
    ------------

    masks_true: list of ND-arrays (int) or ND-array (int)
        where 0=NO masks; 1,2... are mask labels
    masks_pred: list of ND-arrays (int) or ND-array (int)
        ND-array (int) where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    aji : aggregated jaccard index for each set of masks

    """

    aji = np.zeros(len(masks_true))
    for n in range(len(masks_true)):
        iout, preds = mask_ious(masks_true[n], masks_pred[n])
        inds = np.arange(0, masks_true[n].max(), 1, int)
        overlap = _label_overlap(masks_true[n], masks_pred[n])
        union = np.logical_or(masks_true[n] > 0, masks_pred[n] > 0).sum()
        overlap = overlap[inds[preds > 0] + 1, preds[preds > 0].astype(int)]
        aji[n] = overlap.sum() / union
    return aji


def average_precision(masks_true, masks_pred, threshold=[0.5, 0.75, 0.9]):
    """average precision estimation: AP = TP / (TP + FP + FN)

    This function is based heavily on the *fast* stardist matching functions
    (https://github.com/mpicbg-csbd/stardist/blob/master/stardist/matching.py)

    Parameters
    ------------

    masks_true: list of ND-arrays (int) or ND-array (int)
        where 0=NO masks; 1,2... are mask labels
    masks_pred: list of ND-arrays (int) or ND-array (int)
        ND-array (int) where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    ap: array [len(masks_true) x len(threshold)]
        average precision at thresholds
    tp: array [len(masks_true) x len(threshold)]
        number of true positives at thresholds
    fp: array [len(masks_true) x len(threshold)]
        number of false positives at thresholds
    fn: array [len(masks_true) x len(threshold)]
        number of false negatives at thresholds

    """
    not_list = False
    if not isinstance(masks_true, list):
        masks_true = [masks_true]
        masks_pred = [masks_pred]
        not_list = True
    if not isinstance(threshold, list) and not isinstance(threshold, np.ndarray):
        threshold = [threshold]

    if len(masks_true) != len(masks_pred):
        raise ValueError(
            "metrics.average_precision requires len(masks_true)==len(masks_pred)"
        )

    ap = np.zeros((len(masks_true), len(threshold)), np.float32)
    tp = np.zeros((len(masks_true), len(threshold)), np.float32)
    fp = np.zeros((len(masks_true), len(threshold)), np.float32)
    fn = np.zeros((len(masks_true), len(threshold)), np.float32)
    n_true = np.array(list(map(np.max, masks_true)))
    n_pred = np.array(list(map(np.max, masks_pred)))

    for n in range(len(masks_true)):
        # _,mt = np.reshape(np.unique(masks_true[n], return_index=True), masks_pred[n].shape)
        if n_pred[n] > 0:
            iou = _intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
            for k, th in enumerate(threshold):
                tp[n, k] = _true_positive(iou, th)
        fp[n] = n_pred[n] - tp[n]
        fn[n] = n_true[n] - tp[n]
        ap[n] = tp[n] / (tp[n] + fp[n] + fn[n])

    if not_list:
        ap, tp, fp, fn = ap[0], tp[0], fp[0], fn[0]
    return ap, tp, fp, fn


@jit(nopython=True)
def _label_overlap(x, y):
    """fast function to get pixel overlaps between masks in x and y

    Parameters
    ------------

    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]

    """
    # put label arrays into standard form then flatten them
    #     x = (utils.format_labels(x)).ravel()
    #     y = (utils.format_labels(y)).ravel()
    x = x.ravel()
    y = y.ravel()

    # preallocate a 'contact map' matrix
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)

    # loop over the labels in x and add to the corresponding
    # overlap entry. If label A in x and label B in y share P
    # pixels, then the resulting overlap is P
    # len(x)=len(y), the number of pixels in the whole image
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap


def _intersection_over_union(masks_true, masks_pred):
    """intersection over union of all mask pairs

    Parameters
    ------------

    masks_true: ND-array, int
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    iou: ND-array, float
        matrix of IOU pairs of size [x.max()+1, y.max()+1]

    ------------
    How it works:
        The overlap matrix is a lookup table of the area of intersection
        between each set of labels (true and predicted). The true labels
        are taken to be along axis 0, and the predicted labels are taken
        to be along axis 1. The sum of the overlaps along axis 0 is thus
        an array giving the total overlap of the true labels with each of
        the predicted labels, and likewise the sum over axis 1 is the
        total overlap of the predicted labels with each of the true labels.
        Because the label 0 (background) is included, this sum is guaranteed
        to reconstruct the total area of each label. Adding this row and
        column vectors gives a 2D array with the areas of every label pair
        added together. This is equivalent to the union of the label areas
        except for the duplicated overlap area, so the overlap matrix is
        subtracted to find the union matrix.

    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


def _true_positive(iou, th):
    """true positive at threshold th

    Parameters
    ------------

    iou: float, ND-array
        array of IOU pairs
    th: float
        threshold on IOU for positive label

    Returns
    ------------

    tp: float
        number of true positives at threshold

    ------------
    How it works:
        (1) Find minimum number of masks
        (2) Define cost matrix; for a given threshold, each element is negative
            the higher the IoU is (perfect IoU is 1, worst is 0). The second term
            gets more negative with higher IoU, but less negative with greater
            n_min (but that's a constant...)
        (3) Solve the linear sum assignment problem. The costs array defines the cost
            of matching a true label with a predicted label, so the problem is to
            find the set of pairings that minimizes this cost. The scipy.optimize
            function gives the ordered lists of corresponding true and predicted labels.
        (4) Extract the IoUs fro these parings and then threshold to get a boolean array
            whose sum is the number of true positives that is returned.

    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2 * n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    return tp


def flow_error(maski, dP_net, use_gpu=False):
    """error in flows from predicted masks vs flows predicted by network run on image

    This function serves to benchmark the quality of masks, it works as follows
    1. The predicted masks are used to create a flow diagram
    2. The mask-flows are compared to the flows that the network predicted

    If there is a discrepancy between the flows, it suggests that the mask is incorrect.
    Masks with flow_errors greater than 0.4 are discarded by default. Setting can be
    changed in Cellpose.eval or CellposeModel.eval.

    Parameters
    ------------

    maski: ND-array (int)
        masks produced from running dynamics on dP_net,
        where 0=NO masks; 1,2... are mask labels
    dP_net: ND-array (float)
        ND flows where dP_net.shape[1:] = maski.shape

    Returns
    ------------

    flow_errors: float array with length maski.max()
        mean squared error between predicted flows and flows from masks
    dP_masks: ND-array (float)
        ND flows produced from the predicted masks

    """
    if dP_net.shape[1:] != maski.shape:
        print("ERROR: net flow is not same size as predicted masks")
        return

    # flows predicted from estimated masks
    dP_masks = masks_to_flows(maski, use_gpu=use_gpu)
    # difference between predicted flows vs mask flows
    flow_errors = np.zeros(maski.max())
    for i in range(dP_masks.shape[0]):
        flow_errors += mean(
            (dP_masks[i] - dP_net[i] / 5.0) ** 2,
            maski,
            index=np.arange(1, maski.max() + 1),
        )

    return flow_errors, dP_masks


def masks_to_flows_cpu(masks):
    """convert masks to flows using diffusion from center pixel
    Center of masks where diffusion starts is defined to be the
    closest pixel to the median of all pixels that is inside the
    mask. Result of diffusion is converted into flows by computing
    the gradients of the diffusion density map.
    Parameters
    -------------
    masks: int, 2D array
        labelled masks 0=NO masks; 1,2,...=mask labels
    Returns
    -------------
    mu: float, 3D array
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z = mu[0].
    mu_c: float, 2D array
        for each pixel, the distance to the center of the mask
        in which it resides
    """

    Ly, Lx = masks.shape
    mu = np.zeros((2, Ly, Lx), np.float64)
    mu_c = np.zeros((Ly, Lx), np.float64)

    slices = find_objects(masks)
    dia = diameters(masks)[0]
    s2 = (0.15 * dia) ** 2
    for i, si in enumerate(slices):
        if si is not None:
            sr, sc = si
            ly, lx = sr.stop - sr.start + 1, sc.stop - sc.start + 1
            y, x = np.nonzero(masks[sr, sc] == (i + 1))
            y = y.astype(np.int32) + 1
            x = x.astype(np.int32) + 1
            ymed = np.median(y)
            xmed = np.median(x)
            imin = np.argmin((x - xmed) ** 2 + (y - ymed) ** 2)
            xmed = x[imin]
            ymed = y[imin]

            d2 = (x - xmed) ** 2 + (y - ymed) ** 2
            mu_c[sr.start + y - 1, sc.start + x - 1] = np.exp(-d2 / s2)

            niter = 2 * np.int32(np.ptp(x) + np.ptp(y))
            T = np.zeros((ly + 2) * (lx + 2), np.float64)
            T = _extend_centers(T, y, x, ymed, xmed, np.int32(lx), np.int32(niter))
            T[(y + 1) * lx + x + 1] = np.log(1.0 + T[(y + 1) * lx + x + 1])

            dy = T[(y + 1) * lx + x] - T[(y - 1) * lx + x]
            dx = T[y * lx + x + 1] - T[y * lx + x - 1]
            mu[:, sr.start + y - 1, sc.start + x - 1] = np.stack((dy, dx))

    mu /= 1e-20 + (mu**2).sum(axis=0) ** 0.5

    return mu, mu_c


@njit("(float64[:], int32[:], int32[:], int32, int32, int32, int32)", nogil=True)
def _extend_centers(T, y, x, ymed, xmed, Lx, niter):
    """run diffusion from center of mask (ymed, xmed) on mask pixels (y, x)
    Parameters
    --------------
    T: float64, array
        _ x Lx array that diffusion is run in
    y: int32, array
        pixels in y inside mask
    x: int32, array
        pixels in x inside mask
    ymed: int32
        center of mask in y
    xmed: int32
        center of mask in x
    Lx: int32
        size of x-dimension of masks
    niter: int32
        number of iterations to run diffusion
    Returns
    ---------------
    T: float64, array
        amount of diffused particles at each pixel
    """

    for t in range(niter):
        T[ymed * Lx + xmed] += 1
        T[y * Lx + x] = (
            1
            / 9.0
            * (
                T[y * Lx + x]
                + T[(y - 1) * Lx + x]
                + T[(y + 1) * Lx + x]
                + T[y * Lx + x - 1]
                + T[y * Lx + x + 1]
                + T[(y - 1) * Lx + x - 1]
                + T[(y - 1) * Lx + x + 1]
                + T[(y + 1) * Lx + x - 1]
                + T[(y + 1) * Lx + x + 1]
            )
        )
    return T


def masks_to_flows(masks):
    """convert masks to flows using diffusion from center pixel

    Center of masks where diffusion starts is defined to be the
    closest pixel to the median of all pixels that is inside the
    mask. Result of diffusion is converted into flows by computing
    the gradients of the diffusion density map.

    Parameters
    -------------

    masks: int, 2D or 3D array
        labelled masks 0=NO masks; 1,2,...=mask labels

    Returns
    -------------

    mu: float, 3D or 4D array
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z = mu[0].

    mu_c: float, 2D or 3D array
        for each pixel, the distance to the center of the mask
        in which it resides

    """
    if masks.max() == 0:
        return np.zeros((2, *masks.shape), "float32")

    masks_to_flows_device = masks_to_flows_cpu

    if masks.ndim == 3:
        Lz, Ly, Lx = masks.shape
        mu = np.zeros((3, Lz, Ly, Lx), np.float32)
        for z in range(Lz):
            mu0 = masks_to_flows_device(masks[z], device=device)[0]
            mu[[1, 2], z] += mu0
        for y in range(Ly):
            mu0 = masks_to_flows_device(masks[:, y], device=device)[0]
            mu[[0, 2], :, y] += mu0
        for x in range(Lx):
            mu0 = masks_to_flows_device(masks[:, :, x], device=device)[0]
            mu[[0, 1], :, :, x] += mu0
        return mu
    elif masks.ndim == 2:
        mu, mu_c = masks_to_flows_device(masks, device=device)
        return mu

    else:
        raise ValueError("masks_to_flows only takes 2D or 3D arrays")


def merge_labels_across_volume(labelvol, relabelfunc, threshold=3):
    nz, ny, nx = labelvol.shape
    res = np.zeros_like(labelvol)
    res[0, ...] = labelvol[0, ...]
    backup = labelvol.copy()  # kapoors code modifies the input array
    for i in tqdm(range(nz - 1)):

        res[i + 1, ...] = relabelfunc(
            res[i, ...], labelvol[i + 1, ...], threshold=threshold
        )
        labelvol = backup.copy()  # restore the input array
    res = res.astype("uint16")
    return res


def RelabelZ(previousImage, currentImage, threshold):

    currentImage = currentImage.astype("uint16")
    relabelimage = currentImage
    previousImage = previousImage.astype("uint16")
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
                    previouslabel = previousImage[
                        int(indices[previouspoint[1]][0]),
                        int(indices[previouspoint[1]][1]),
                    ]
                    if previouspoint[0] > threshold:

                        pixel_condition = currentImage == currentlabel
                        pixel_replace_condition = currentlabel
                        relabelimage = image_conditionals(
                            relabelimage,
                            pixel_condition,
                            pixel_replace_condition,
                        )

                    else:
                        pixel_condition = currentImage == currentlabel
                        pixel_replace_condition = previouslabel
                        relabelimage = image_conditionals(
                            relabelimage,
                            pixel_condition,
                            pixel_replace_condition,
                        )

    return relabelimage


def CleanMask(star_labels, OverAllunet_mask):
    OverAllunet_mask = np.logical_or(OverAllunet_mask > 0, star_labels > 0)
    OverAllunet_mask = binary_erosion(OverAllunet_mask)
    OverAllunet_mask = label(OverAllunet_mask)
    OverAllunet_mask = fill_label_holes(OverAllunet_mask.astype("uint16"))

    return OverAllunet_mask


def UNETPrediction3D(
    image,
    model,
    n_tiles,
    axis,
    nms_thresh=0.3,
    min_size_mask=10,
    max_size=100000,
    slice_merge=False,
    erosion_iterations=15,
    ExpandLabels=True,
):

    model_dim = model.config.n_dim
    if nms_thresh is None:
        nms_thresh = 0.3
    if model_dim < len(image.shape):
        Segmented = np.zeros_like(image)

        for i in range(image.shape[0]):
            Segmented[i] = model.predict(
                image[i].astype("float32"),
                axis.replace("Z", ""),
                n_tiles=(n_tiles[-2], n_tiles[-1]),
            )

    else:

        Segmented = model.predict(image.astype("float32"), axis, n_tiles=n_tiles)

    try:
        thresholds = threshold_multiotsu(Segmented, classes=2)

        # Using the threshold values, we generate the three regions.
        regions = np.digitize(Segmented, bins=thresholds)
    except ValueError:

        regions = Segmented

    Binary = regions > 0
    overall_mask = Binary.copy()

    if model_dim == 3:
        for i in range(image.shape[0]):
            overall_mask[i] = binary_dilation(
                overall_mask[i], iterations=erosion_iterations
            )
            overall_mask[i] = binary_erosion(
                overall_mask[i], iterations=erosion_iterations
            )
            overall_mask[i] = fill_label_holes(overall_mask[i])

    Binary = label(Binary)

    if model_dim == 2:
        Binary = remove_small_objects(Binary.astype("uint16"), min_size=min_size_mask)
        Binary = remove_big_objects(Binary.astype("uint16"), max_size=max_size)
        Binary = fill_label_holes(Binary)
        Finalimage = relabel_sequential(Binary)[0]
        skeleton = Skel(Finalimage)
        skeleton = skeleton > 0
    if model_dim == 3 and slice_merge:
        for i in range(image.shape[0]):
            Binary[i] = label(Binary[i])

        Binary = match_labels(Binary, nms_thresh=nms_thresh)
        Binary = fill_label_holes(Binary)

    if model_dim == 3:
        for i in range(image.shape[0]):
            Binary[i] = remove_small_objects(
                Binary[i].astype("uint16"), min_size=min_size_mask
            )
            Binary[i] = remove_big_objects(
                Binary[i].astype("uint16"), max_size=max_size
            )
        Finalimage = relabel_sequential(Binary)[0]
        skeleton = Skel(Finalimage)

        if ExpandLabels:

            Finalimage, skeleton = VollSeg_label_expansion(
                image, overall_mask, Finalimage, skeleton
            )

    return Finalimage


def Bbox_region(image):

    props = measure.regionprops(image)
    area = [prop.area for prop in props]
    if len(area) > 0:
        largest_blob_ind = np.argmax(area)
        largest_bbox = props[largest_blob_ind].bbox
        return largest_bbox

def SuperSTARPrediction(
    image,
    model,
    n_tiles,
    unet_mask=None,
    OverAllunet_mask=None,
    UseProbability=True,
    prob_thresh=None,
    nms_thresh=None,
    seedpool=False,
):
    skip_watershed = True
    if seedpool:  
       skip_watershed = False

    if prob_thresh is None and nms_thresh is None:
        prob_thresh = model.thresholds.prob
        nms_thresh = model.thresholds.nms

    star_labels, SmallProbability, SmallDistance = model.predict_vollseg(
        image.astype("float32"),
        n_tiles=n_tiles,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
    )

    grid = model.config.grid
    Probability = resize(
        SmallProbability,
        output_shape=(
            SmallProbability.shape[0] * grid[0],
            SmallProbability.shape[1] * grid[1],
        ),
    )
    Distance = MaxProjectDist(SmallDistance, axis=-1)
    Distance = resize(
        Distance,
        output_shape=(
            Distance.shape[0] * grid[0],
            Distance.shape[1] * grid[1],
        ),
    )

    pixel_condition = Probability < GLOBAL_THRESH
    pixel_replace_condition = 0
    Probability = image_conditionals(
        Probability, pixel_condition, pixel_replace_condition
    )

    if UseProbability:

        MaxProjectDistance = Probability[: star_labels.shape[0], : star_labels.shape[1]]

    else:

        MaxProjectDistance = Distance[: star_labels.shape[0], : star_labels.shape[1]]

    if OverAllunet_mask is None:
        OverAllunet_mask = unet_mask
    if OverAllunet_mask is not None:
        OverAllunet_mask = CleanMask(star_labels, OverAllunet_mask)

    if unet_mask is None:
        
        unet_mask = star_labels > 0
        
    if skip_watershed:
        Watershed = star_labels 
        markers = star_labels
    else:        
        Watershed, markers = SuperWatershedwithMask(
            MaxProjectDistance,
            star_labels.astype("uint16"),
            unet_mask.astype("uint16"),
            nms_thresh=nms_thresh,
            seedpool=seedpool,
        )
        Watershed = fill_label_holes(Watershed.astype("uint16"))
        
    return Watershed, markers, star_labels, MaxProjectDistance

def STARPrediction3D(
    image,
    axes,
    model,
    n_tiles,
    unet_mask=None,
    UseProbability=True,
    seedpool=True,
    prob_thresh=None,
    nms_thresh=None,
):

    copymodel = model
    grid = copymodel.config.grid
    if prob_thresh is None and nms_thresh is None:
        prob_thresh = model.thresholds.prob
        nms_thresh = model.thresholds.nms

    (star_labels, SmallProbability, SmallDistance,) = model.predict_vollseg(
        image.astype("float32"),
        axes=axes,
        n_tiles=n_tiles,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
    )

    if unet_mask is not None:
        if UseProbability is False:

            SmallDistance = MaxProjectDist(SmallDistance, axis=-1)
            Distance = np.zeros(
                [
                    SmallDistance.shape[0] * grid[0],
                    SmallDistance.shape[1] * grid[1],
                    SmallDistance.shape[2] * grid[2],
                ]
            )

        Probability = np.zeros(
            [
                SmallProbability.shape[0] * grid[0],
                SmallProbability.shape[1] * grid[1],
                SmallProbability.shape[2] * grid[2],
            ]
        )

        # We only allow for the grid parameter to be 1 along the Z axis
        for i in range(0, SmallProbability.shape[0]):
            Probability[i, :] = resize(
                SmallProbability[i, :],
                output_shape=(Probability.shape[1], Probability.shape[2]),
            )

            if UseProbability is False:
                Distance[i, :] = resize(
                    SmallDistance[i, :],
                    output_shape=(Distance.shape[1], Distance.shape[2]),
                )

        if UseProbability:

            MaxProjectDistance = Probability[
                : star_labels.shape[0],
                : star_labels.shape[1],
                : star_labels.shape[2],
            ]

        else:

            MaxProjectDistance = Distance[
                : star_labels.shape[0],
                : star_labels.shape[1],
                : star_labels.shape[2],
            ]

        if unet_mask is None:
            unet_mask = star_labels > 0

        Watershed, markers = WatershedwithMask3D(
            MaxProjectDistance,
            star_labels.astype("uint16"),
            unet_mask.astype("uint16"),
            nms_thresh=nms_thresh,
            seedpool=seedpool,
        )
    else:

        Watershed = star_labels
        properties = measure.regionprops(star_labels)

        Coordinates = [prop.centroid for prop in properties]
        Coordinates.append((0, 0, 0))

        Coordinates = np.asarray(Coordinates)
        coordinates_int = np.round(Coordinates).astype(int)

        markers_raw = np.zeros_like(star_labels)
        markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))
        markers = morphology.dilation(markers_raw.astype("uint16"), morphology.ball(2))
        if UseProbability is False:

            SmallDistance = MaxProjectDist(SmallDistance, axis=-1)
            Distance = np.zeros(
                [
                    SmallDistance.shape[0] * grid[0],
                    SmallDistance.shape[1] * grid[1],
                    SmallDistance.shape[2] * grid[2],
                ]
            )

        Probability = np.zeros(
            [
                SmallProbability.shape[0] * grid[0],
                SmallProbability.shape[1] * grid[1],
                SmallProbability.shape[2] * grid[2],
            ]
        )

        # We only allow for the grid parameter to be 1 along the Z axis
        for i in range(0, SmallProbability.shape[0]):
            Probability[i, :] = resize(
                SmallProbability[i, :],
                output_shape=(Probability.shape[1], Probability.shape[2]),
            )

            if UseProbability is False:
                Distance[i, :] = resize(
                    SmallDistance[i, :],
                    output_shape=(Distance.shape[1], Distance.shape[2]),
                )

        if UseProbability:

            MaxProjectDistance = Probability[
                : star_labels.shape[0],
                : star_labels.shape[1],
                : star_labels.shape[2],
            ]

        else:

            MaxProjectDistance = Distance[
                : star_labels.shape[0],
                : star_labels.shape[1],
                : star_labels.shape[2],
            ]
    return Watershed, MaxProjectDistance, star_labels, markers


def SuperWatershedwithMask(Image, Label, mask, nms_thresh, seedpool, z_thresh=1):

    CopyImage = Image.copy()
    properties = measure.regionprops(Label)
    Coordinates = [prop.centroid for prop in properties]
    binaryproperties = measure.regionprops(label(mask), CopyImage)
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
    markers_raw = np.zeros_like(CopyImage)
    markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))

    markers = morphology.dilation(markers_raw, morphology.disk(2))
    watershedImage = watershed(-CopyImage, markers, mask=mask.copy())

    watershedImage = NMSLabel(
        watershedImage, nms_thresh, z_thresh=z_thresh
    ).supressregions()

    return watershedImage, markers


def simple_dist(label_image):

    # Create an empty output image
    binary_image = np.zeros_like(label_image, dtype=np.float32)
    binary_image = find_boundaries(label_image, mode="outer") * 255
    binary_image = gaussian_filter(binary_image, sigma=2)
    output_image = binary_image / np.max(binary_image)
    return output_image


def exponential_decay(z, center_z, decay_rate=1.0):
    """
    Exponentially decaying function centered at center_z.
    The farther from the center_z, the smaller the value.
    """
    distance = np.abs(z - center_z)
    return np.exp(-decay_rate * distance)


def generate_decay_map(center_z, z_dim, decay_rate):
    z = np.arange(z_dim)
    return exponential_decay(z, center_z, decay_rate)


def CellPoseWater(membrane_image, sized_smart_seeds, mask):

    if mask.ndim == 2:
        mask = np.repeat(mask[np.newaxis, :, :], membrane_image.shape[0], axis=0)

    mask = binary_erosion(mask, iterations=1)
    mask = binary_dilation(mask, iterations=1)
    properties = measure.regionprops(sized_smart_seeds)
    Coordinates = [prop.centroid for prop in properties]
    Coordinates.append((0, 0, 0))
    Coordinates = np.asarray(Coordinates)
    coordinates_int = np.round(Coordinates).astype(int)

    markers_raw = np.zeros_like(sized_smart_seeds)
    markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))
    markers = morphology.dilation(markers_raw.astype("uint16"), morphology.ball(2))

    thresh = threshold_otsu(membrane_image)
    binary_image = membrane_image > thresh
    thick_binary_image = binary_image.copy()
    thinner_binary_image = binary_erosion(thick_binary_image, iterations=2)
    boundary_binary_image = (
        find_boundaries(thick_binary_image.copy(), mode="outer") * 255
    )

    watershed_result = watershed(boundary_binary_image, markers, mask=mask)
    watershed_result, _, _ = relabel_sequential(watershed_result.astype(np.uint16))

    labels_to_remove = np.unique(watershed_result[thinner_binary_image > 0])

    for rem_label in labels_to_remove:
        if rem_label != 0:
            watershed_result[watershed_result == rem_label] = 0

    return watershed_result


def WatershedwithMask3D(Image, Label, mask, nms_thresh, seedpool=True, z_thresh=1):

    CopyImage = Image.copy()
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

    markers_raw = np.zeros_like(CopyImage)
    markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))
    markers = morphology.dilation(markers_raw.astype("uint16"), morphology.ball(2))
    watershedImage = watershed(-CopyImage, markers, mask=mask.copy())

    watershedImage = NMSLabel(
        watershedImage, nms_thresh, z_thresh=z_thresh
    ).supressregions()

    return watershedImage, markers


def MaxProjectDist(Image, axis=-1):

    MaxProject = np.amax(Image, axis=axis)

    return MaxProject


def MidProjectDist(Image, axis=-1, slices=1):

    assert len(Image.shape) >= 3
    SmallImage = Image.take(
        indices=range(Image.shape[axis] // 2 - slices, Image.shape[axis] // 2 + slices),
        axis=axis,
    )

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

    x = x.astype("float32")

    minVal = np.min(x)
    maxVal = np.max(x)

    x = (x - minVal) / (maxVal - minVal + 1.0e-20)

    return x


def normalizeZero255(x):

    x = x.astype("float32")

    minVal = np.min(x)
    maxVal = np.max(x)

    x = (x - minVal) / (maxVal - minVal + 1.0e-20)

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


def plot_train_history(history, savedir, modelname, *keys, **kwargs):
    """Plot (Keras) training history returned by :func:`CARE.train`."""
    import matplotlib.pyplot as plt

    logy = kwargs.pop("logy", False)

    if all(isinstance(k, str) for k in keys):
        w, keys = 1, [keys]
    else:
        w = len(keys)

    plt.gcf()
    for i, group in enumerate(keys):
        plt.subplot(1, w, i + 1)
        for k in [group] if isinstance(group, str) else group:
            plt.plot(history.epoch, history.history[k], ".-", label=k, **kwargs)
            if logy:
                plt.gca().set_yscale("log", nonposy="clip")
        plt.xlabel("epoch")
        plt.legend(loc="best")
    plt.savefig(savedir + "/" + modelname + "train_accuracy" + ".png", dpi=600)
