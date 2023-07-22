import os
import numpy as np
from pathlib import Path
from tifffile import imread, imwrite
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_erosion
from tqdm import tqdm
from datetime import datetime
from uuid import uuid4


class SimplePatches:
    def __init__(
        self,
        base_dir,
        real_mask_dir,
        real_mask_patch_dir,
        patch_size,
        pattern=".tif",
        lower_ratio_fore_to_back=0.5,
        upper_ratio_fore_to_back=0.9,
    ):

       
        self.base_dir = base_dir
        self.real_mask_dir = os.path.join(base_dir, real_mask_dir)
        self.real_mask_patch_dir = os.path.join(
            self.base_dir, real_mask_patch_dir
        )
        self.patch_size = patch_size
        self.pattern = pattern
        self.lower_ratio_fore_to_back = lower_ratio_fore_to_back
        self.upper_ratio_fore_to_back = upper_ratio_fore_to_back
        self.acceptable_formats = [".tif", ".TIFF", ".TIF", ".png"]

        self._create_smart_patches()

   

    def _create_smart_patches(self):

        Path(self.real_mask_patch_dir).mkdir(exist_ok=True)
        files = os.listdir(self.real_mask_dir)
        for fname in files:
            if any(fname.endswith(f) for f in self.acceptable_formats):
                    label_image_membrane = imread(
                        os.path.join(
                            self.real_mask_dir, fname
                        )
                    ).astype(np.uint16)
                    self.ndim = len(label_image_membrane.shape)
                    properties_membrane = regionprops(label_image_membrane)
                    for count, prop in tqdm(enumerate(properties_membrane)):
                        self._label_maker(
                            fname,
                            label_image_membrane,
                            count,
                            prop,
                            self.real_mask_patch_dir,
                        )

    

    def _label_maker(
        self,
        fname: str,
        labelimage: np.ndarray,
        count: int,
        prop: regionprops,
        real_mask_patch_dir: str,
    ):

        name = os.path.splitext(fname)[0]

        if self.ndim == 2:

            self.valid = False
            centroid = prop.centroid
            x = centroid[1]
            y = centroid[0]

            crop_Xminus = x - int(self.patch_size[1] / 2)
            crop_Xplus = x + int(self.patch_size[1] / 2)
            crop_Yminus = y - int(self.patch_size[0] / 2)
            crop_Yplus = y + int(self.patch_size[0] / 2)
            crop_minus = [crop_Yminus, crop_Xminus]
            region = (
                slice(int(crop_Yminus), int(crop_Yplus)),
                slice(int(crop_Xminus), int(crop_Xplus)),
            )
        if self.ndim == 3:

            self.valid = False
            centroid = prop.centroid
            z = centroid[0]
            x = centroid[2]
            y = centroid[1]

            crop_Xminus = x - int(self.patch_size[2] / 2)
            crop_Xplus = x + int(self.patch_size[2] / 2)
            crop_Yminus = y - int(self.patch_size[1] / 2)
            crop_Yplus = y + int(self.patch_size[1] / 2)
            crop_Zminus = z - int(self.patch_size[0] / 2)
            crop_Zplus = z + int(self.patch_size[0] / 2)
            crop_minus = [crop_Zminus, crop_Yminus, crop_Xminus]
            region = (
                slice(int(crop_Zminus), int(crop_Zplus)),
                slice(int(crop_Yminus), int(crop_Yplus)),
                slice(int(crop_Xminus), int(crop_Xplus)),
            )
        if all(crop for crop in crop_minus) > 0:
            self.crop_labelimage = labelimage[region]
            self.crop_labelimage = remove_small_objects(
                self.crop_labelimage.astype("uint16"), min_size=10
            )
            if (
                self.crop_labelimage.shape[0] == self.patch_size[0]
                and self.crop_labelimage.shape[1] == self.patch_size[1]
                and self.ndim == 2
            ):
                self._crop_maker(
                    name,
                    count,
                    real_mask_patch_dir,
                )
            if (
                self.crop_labelimage.shape[0] == self.patch_size[0]
                and self.crop_labelimage.shape[1] == self.patch_size[1]
                and self.crop_labelimage.shape[2] == self.patch_size[2]
                and self.ndim == 3
            ):
                self._crop_maker(
                    name,
                    count,
                    real_mask_patch_dir,
                )

    def _crop_maker(
        self,
        name,
        count,
        real_mask_patch_dir,
    ):

        self._region_selector()
        if (
            self.valid
            
        ):
           
            
            self.eroded_crop_labelimage = self.crop_labelimage
            eventid = datetime.now().strftime("%Y%m-%d%H-%M%S-") + str(uuid4())
           
            imwrite(
                os.path.join(
                    real_mask_patch_dir,
                    name + eventid + str(count) + self.pattern,
                ),
                self.eroded_crop_labelimage.astype("uint16"),
            )

    def _region_selector(self):

        non_zero_indices = list(zip(*np.where(self.crop_labelimage > 0)))

        total_indices = list(zip(*np.where(self.crop_labelimage >= 0)))
        if len(total_indices) > 0:
            norm_foreground = len(non_zero_indices) / len(total_indices)
            index_ratio = float(norm_foreground)
            if (
                index_ratio >= self.lower_ratio_fore_to_back
                and index_ratio <= self.upper_ratio_fore_to_back
            ):

                self.valid = True


def erode_label_holes(lbl_img, iterations):
    lbl_img_filled = np.zeros_like(lbl_img)
    for lb in range(np.min(lbl_img), np.max(lbl_img) + 1):
        mask = lbl_img == lb
        mask_filled = binary_erosion(mask, iterations=iterations)
        lbl_img_filled[mask_filled] = lb
    return lbl_img_filled


def erode_labels(segmentation, erosion_iterations=2):
    # create empty list where the eroded masks can be saved to
    regions = regionprops(segmentation)
    erode = np.zeros(segmentation.shape)

    def erode_mask(segmentation_labels, label_id, erosion_iterations):

        only_current_label_id = np.where(segmentation_labels == label_id, 1, 0)
        eroded = binary_erosion(
            only_current_label_id, iterations=erosion_iterations
        )
        relabeled_eroded = np.where(eroded == 1, label_id, 0)
        return relabeled_eroded

    for i in range(len(regions)):
        label_id = regions[i].label
        erode = erode + erode_mask(segmentation, label_id, erosion_iterations)

    # convert list of numpy arrays to stacked numpy array
    return erode
