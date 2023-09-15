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
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation
from skimage.filters import gaussian


class SmartNucleiPatches:
    def __init__(
        self,
        base_signal_dir,
        raw_signal_dir,
        signal_channel_results_directory,
        signal_raw_save_dir,
        signal_real_mask_patch_dir,
        signal_binary_mask_patch_dir,
        patch_size,
        erosion_iterations=0,
        pattern=".tif",
        lower_ratio_fore_to_back=0.5,
        upper_ratio_fore_to_back=0.9,
        max_foreground_patches_per_image=np.inf,
        max_background_patches_per_image=np.inf,
        create_background_only=False,
        create_foreground_only=False,
    ):

        self.max_foreground_patches_per_image = (
            max_foreground_patches_per_image
        )
        self.max_background_patches_per_image = (
            max_background_patches_per_image
        )
        self.base_signal_dir = base_signal_dir
        self.create_background_only = create_background_only
        self.create_foreground_only = create_foreground_only
        
        self.raw_signal_dir = os.path.join(
            self.base_signal_dir, raw_signal_dir
        )
        
        self.signal_channel_results_directory = os.path.join(
            self.base_signal_dir, signal_channel_results_directory
        )
        self.signal_raw_save_dir = os.path.join(
            base_signal_dir, signal_raw_save_dir
        )
        
        self.signal_binary_mask_patch_dir = os.path.join(
            self.base_signal_dir, signal_binary_mask_patch_dir
        )


        self.signal_real_mask_patch_dir = os.path.join(
            self.base_signal_dir, signal_real_mask_patch_dir
        )
        
        self.patch_size = patch_size
        self.erosion_iterations = erosion_iterations
        self.pattern = pattern
        self.lower_ratio_fore_to_back = lower_ratio_fore_to_back
        self.upper_ratio_fore_to_back = upper_ratio_fore_to_back
        self.acceptable_formats = [".tif", ".TIFF", ".TIF", ".png"]

        if self.create_background_only:
            self._create_background_patches()
        elif self.create_foreground_only:
            self._create_smart_patches()
        else:
            self._create_smart_patches()
            self._create_background_patches()

    def _create_background_patches(self):

        Path(self.signal_raw_save_dir).mkdir(exist_ok=True)
        Path(self.signal_binary_mask_patch_dir).mkdir(exist_ok=True)
        Path(self.signal_real_mask_patch_dir).mkdir(exist_ok=True)
        files = os.listdir(self.raw_signal_dir)
        
        for fname in files:
            if any(fname.endswith(f) for f in self.acceptable_formats):

                    self.main_count = 0

                    raw_signal_image = imread(
                        os.path.join(self.raw_signal_dir, fname)
                    )
                    self.ndim = len(raw_signal_image.shape)
                    label_image_signal = imread(
                        os.path.join(
                            self.signal_channel_results_directory, fname
                        )
                    ).astype(np.uint16)

                    self._background_label_maker(
                        fname,
                        raw_signal_image,
                        label_image_signal,
                        self.signal_raw_save_dir,
                        self.signal_binary_mask_patch_dir,
                        self.signal_real_mask_patch_dir,
                    )

            

    def _create_smart_patches(self):

        Path(self.signal_raw_save_dir).mkdir(exist_ok=True)
        Path(self.signal_binary_mask_patch_dir).mkdir(exist_ok=True)
        Path(self.signal_real_mask_patch_dir).mkdir(exist_ok=True)
        files = os.listdir(self.raw_signal_dir)
       
        for fname in files:
            if any(fname.endswith(f) for f in self.acceptable_formats):

                    self.main_count = 0

                    raw_signal_image = imread(
                        os.path.join(self.raw_signal_dir, fname)
                    )
                    self.ndim = len(raw_signal_image.shape)
                    label_image_signal = imread(
                        os.path.join(
                            self.signal_channel_results_directory, fname
                        )
                    ).astype(np.uint16)
                    properties_signal = regionprops(label_image_signal)
                    for count, prop in tqdm(enumerate(properties_signal)):
                        self._label_maker(
                            fname,
                            raw_signal_image,
                            label_image_signal,
                            count,
                            prop,
                            self.signal_raw_save_dir,
                            self.signal_binary_mask_patch_dir,
                            self.signal_real_mask_patch_dir,
                        )
                

    def _background_label_maker(
        self,
        fname,
        rawimage,
        labelimage,
        raw_save_dir: str,
        binary_mask_patch_dir: str,
        real_mask_patch_dir: str,
    ):

        zero_indices = list(zip(*np.where(labelimage == 0)))
        self.main_count = 0
        for index in zero_indices:

            if self.main_count < self.max_background_patches_per_image:
                name = os.path.splitext(fname)[0]
                if self.ndim == 2:
                    x = index[1]
                    y = index[0]
                    crop_Xminus = x - int(self.patch_size[1] / 2)
                    crop_Xplus = x + int(self.patch_size[1] / 2)
                    crop_Yminus = y - int(self.patch_size[0] / 2)
                    crop_Yplus = y + int(self.patch_size[0] / 2)
                    if (
                        crop_Xminus > 0
                        and crop_Xplus < rawimage.shape[1]
                        and crop_Yminus > 0
                        and crop_Yplus < rawimage.shape[0]
                    ):

                        properties = regionprops(labelimage)
                        for prop in properties:
                            centroid = prop.centroid
                            xc = centroid[1]
                            yc = centroid[0]
                            crop_Xminusc = xc - int(self.patch_size[1] / 2)
                            crop_Xplusc = xc + int(self.patch_size[1] / 2)
                            crop_Yminusc = yc - int(self.patch_size[0] / 2)
                            crop_Yplusc = yc + int(self.patch_size[0] / 2)
                            regionc = (
                                slice(int(crop_Yminusc), int(crop_Yplusc)),
                                slice(int(crop_Xminusc), int(crop_Xplusc)),
                            )

                            if (
                                crop_Xminusc > 0
                                and crop_Xplusc < rawimage.shape[1]
                                and crop_Yminusc > 0
                                and crop_Yplusc < rawimage.shape[0]
                            ):
                                raw_patch = rawimage[
                                    crop_Yminus:crop_Yplus,
                                    crop_Xminus:crop_Xplus,
                                ]
                                mask_patch_zero = labelimage[
                                    crop_Yminus:crop_Yplus,
                                    crop_Xminus:crop_Xplus,
                                ]

                                raw_patchc = rawimage[regionc]
                                mask_patchc = labelimage[regionc]
                                raw_patch = np.add(raw_patch, raw_patchc)
                                mask_patch = np.add(
                                    mask_patch_zero, mask_patchc
                                )

                                if (
                                    np.sum(raw_patch) > 0
                                    and np.sum(mask_patch_zero) == 0
                                ):
                                    self.main_count += 1
                                    eventid = datetime.now().strftime(
                                        "%Y%m-%d%H-%M%S-"
                                    ) + str(uuid4())

                                    imwrite(
                                        os.path.join(
                                            raw_save_dir,
                                            name
                                            + "back"
                                            + eventid
                                            + str(self.main_count)
                                            + ".tif",
                                        ),
                                        raw_patch.astype("float32"),
                                    )
                                    if self.erosion_iterations > 0:
                                        binary_mask_patch = erode_labels(
                                            mask_patch.astype("uint16"),
                                            self.erosion_iterations,
                                        )
                                    else:
                                        binary_mask_patch = mask_patch
                                    binary_mask_patch = binary_mask_patch > 0
                                    imwrite(
                                        os.path.join(
                                            binary_mask_patch_dir,
                                            name
                                            + "back"
                                            + eventid
                                            + str(self.main_count)
                                            + ".tif",
                                        ),
                                        binary_mask_patch.astype("uint16"),
                                    )

                                    imwrite(
                                        os.path.join(
                                            real_mask_patch_dir,
                                            name
                                            + "back"
                                            + eventid
                                            + str(self.main_count)
                                            + ".tif",
                                        ),
                                        mask_patch.astype("uint16"),
                                    )
                if self.ndim == 3:
                    x = index[2]
                    y = index[1]
                    z = index[0]
                    crop_Xminus = x - int(self.patch_size[2] / 2)
                    crop_Xplus = x + int(self.patch_size[2] / 2)
                    crop_Yminus = y - int(self.patch_size[1] / 2)
                    crop_Yplus = y + int(self.patch_size[1] / 2)
                    crop_Zminus = z - int(self.patch_size[0] / 2)
                    crop_Zplus = z + int(self.patch_size[0] / 2)
                    if (
                        crop_Xminus > 0
                        and crop_Xplus < rawimage.shape[2]
                        and crop_Yminus > 0
                        and crop_Yplus < rawimage.shape[1]
                        and crop_Zminus > 0
                        and crop_Zplus < rawimage.shape[0]
                    ):
                        properties = regionprops(labelimage)
                        for prop in properties:
                            centroid = prop.centroid
                            xc = centroid[2]
                            yc = centroid[1]
                            zc = centroid[0]

                            crop_Xminusc = xc - int(self.patch_size[2] / 2)
                            crop_Xplusc = xc + int(self.patch_size[2] / 2)
                            crop_Yminusc = yc - int(self.patch_size[1] / 2)
                            crop_Yplusc = yc + int(self.patch_size[1] / 2)
                            crop_Zminusc = zc - int(self.patch_size[0] / 2)
                            crop_Zplusc = zc + int(self.patch_size[0] / 2)
                            regionc = (
                                slice(int(crop_Zminusc), int(crop_Zplusc)),
                                slice(int(crop_Yminusc), int(crop_Yplusc)),
                                slice(int(crop_Xminusc), int(crop_Xplusc)),
                            )

                            if (
                                crop_Xminusc > 0
                                and crop_Xplusc < rawimage.shape[2]
                                and crop_Yminusc > 0
                                and crop_Yplusc < rawimage.shape[1]
                                and crop_Zminusc > 0
                                and crop_Zplusc < rawimage.shape[0]
                            ):
                                raw_patch = rawimage[
                                    crop_Zminus:crop_Zplus,
                                    crop_Yminus:crop_Yplus,
                                    crop_Xminus:crop_Xplus,
                                ]
                                mask_patch_zero = labelimage[
                                    crop_Zminus:crop_Zplus,
                                    crop_Yminus:crop_Yplus,
                                    crop_Xminus:crop_Xplus,
                                ]

                                raw_patchc = rawimage[regionc]
                                mask_patchc = labelimage[regionc]
                                raw_patch = np.add(raw_patch, raw_patchc)
                                mask_patch = np.add(
                                    mask_patch_zero, mask_patchc
                                )

                                if (
                                    np.sum(raw_patch) > 0
                                    and np.sum(mask_patch_zero) == 0
                                ):
                                    self.main_count += 1
                                    eventid = datetime.now().strftime(
                                        "%Y%m-%d%H-%M%S-"
                                    ) + str(uuid4())

                                    imwrite(
                                        os.path.join(
                                            raw_save_dir,
                                            name
                                            + "back"
                                            + eventid
                                            + str(self.main_count)
                                            + ".tif",
                                        ),
                                        raw_patch.astype("float32"),
                                    )
                                    if self.erosion_iterations > 0:
                                        binary_mask_patch = erode_labels(
                                            mask_patch.astype("uint16"),
                                            self.erosion_iterations,
                                        )
                                    else:
                                        binary_mask_patch = mask_patch
                                    binary_mask_patch = binary_mask_patch > 0
                                    imwrite(
                                        os.path.join(
                                            binary_mask_patch_dir,
                                            name
                                            + "back"
                                            + eventid
                                            + str(self.main_count)
                                            + ".tif",
                                        ),
                                        binary_mask_patch.astype("uint16"),
                                    )

                                    imwrite(
                                        os.path.join(
                                            real_mask_patch_dir,
                                            name
                                            + "back"
                                            + eventid
                                            + str(self.main_count)
                                            + ".tif",
                                        ),
                                        mask_patch.astype("uint16"),
                                    )

    def _label_maker(
        self,
        fname: str,
        rawimage: np.ndarray,
        labelimage: np.ndarray,
        count: int,
        prop: regionprops,
        raw_save_dir: str,
        binary_mask_patch_dir: str,
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
            self.crop_image = rawimage[region]
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
                    raw_save_dir,
                    binary_mask_patch_dir,
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
                    raw_save_dir,
                    binary_mask_patch_dir,
                    real_mask_patch_dir,
                )

    def _crop_maker(
        self,
        name,
        count,
        raw_save_dir,
        binary_mask_patch_dir,
        real_mask_patch_dir,
    ):

        self._region_selector()
        if (
            self.valid
            and self.main_count < self.max_foreground_patches_per_image
        ):
            self.main_count += 1
            if self.erosion_iterations > 0:
                self.eroded_crop_labelimage = erode_labels(
                    self.crop_labelimage.astype("uint16"),
                    self.erosion_iterations,
                )
            else:
                self.eroded_crop_labelimage = self.crop_labelimage
            eroded_binary_image = self.eroded_crop_labelimage > 0
            eventid = datetime.now().strftime("%Y%m-%d%H-%M%S-") + str(uuid4())
            imwrite(
                os.path.join(
                    binary_mask_patch_dir,
                    name + eventid + str(count) + self.pattern,
                ),
                eroded_binary_image.astype("uint16"),
            )
            imwrite(
                os.path.join(
                    raw_save_dir, name + eventid + str(count) + self.pattern
                ),
                self.crop_image.astype("float32"),
            )
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
