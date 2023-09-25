#!/usr/bin/env python3
"""
Created on Mon Sep 30 14:38:04 2019

@author: aimachine
"""

import numpy as np
import os
from tifffile import imread, imwrite
from csbdeep.utils import axes_dict
from skimage.morphology import binary_dilation, binary_erosion
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import find_objects
from csbdeep.data import RawData, create_patches, create_patches_reduced_target
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
from csbdeep.utils import normalize
from .utils import plot_train_history
import glob

# from IPython.display import clear_output
from stardist.models import Config2D, StarDist2D
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
from skimage.measure import label, regionprops
from scipy import ndimage
from pathlib import Path
from scipy.ndimage import zoom


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
        eroded = ndimage.binary_erosion(
            only_current_label_id, iterations=erosion_iterations
        )
        relabeled_eroded = np.where(eroded == 1, label_id, 0)
        return relabeled_eroded

    for i in range(len(regions)):
        label_id = regions[i].label
        erode = erode + erode_mask(segmentation, label_id, erosion_iterations)

    # convert list of numpy arrays to stacked numpy array
    return erode


class SmartSeeds2D:
    def __init__(
        self,
        base_dir,
        model_name,
        model_dir,
        npz_filename=None,
        n_patches_per_image=1,
        raw_dir=None,
        real_mask_dir=None,
        binary_mask_dir=None,
        binary_erode_mask_dir=None,
        val_raw_dir=None,
        val_real_mask_dir=None,
        def_shape=None,
        def_label_shape=None,
        downsample_factor=1,
        startfilter=48,
        RGB=False,
        axes="YX",
        axis_norm=(0, 1),
        pattern=".tif",
        validation_split=0.01,
        n_channel_in=1,
        erosion_iterations=2,
        train_seed_unet=False,
        train_unet=False,
        train_star=False,
        load_data_sequence=False,
        grid=(1, 1),
        generate_npz=False,
        patch_x=256,
        patch_y=256,
        use_gpu=False,
        unet_n_first=64,
        batch_size=1,
        depth=3,
        kern_size=7,
        n_rays=16,
        epochs=400,
        learning_rate=0.0001,
    ):

        self.npz_filename = npz_filename
        self.base_dir = base_dir
        self.downsample_factor = downsample_factor
        self.train_unet = train_unet
        self.train_seed_unet = train_seed_unet
        self.train_star = train_star
        self.load_data_sequence = load_data_sequence
        self.model_dir = model_dir
        self.raw_dir = raw_dir
        self.real_mask_dir = real_mask_dir
        self.binary_mask_dir = binary_mask_dir
        self.binary_erode_mask_dir = binary_erode_mask_dir
        self.val_raw_dir = val_raw_dir
        self.val_real_mask_dir = val_real_mask_dir
        self.model_name = model_name
        self.generate_npz = generate_npz
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.depth = depth
        self.axes = axes
        self.def_shape = def_shape
        self.def_label_shape = def_label_shape

        self.n_channel_in = n_channel_in
        self.erosion_iterations = erosion_iterations
        self.n_rays = n_rays
        self.kern_size = kern_size
        self.patch_x = patch_x
        self.patch_y = patch_y
        self.RGB = RGB
        self.axis_norm = axis_norm
        self.pattern = pattern
        self.validation_split = validation_split
        self.startfilter = startfilter
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.grid = grid
        self.search_pattern = "*" + self.pattern
        self.unet_n_first = unet_n_first
        self.n_patches_per_image = n_patches_per_image

        # Load training and validation data
        self.Train()

    class DataSequencer(Sequence):
        def __init__(self, files, axis_norm, Normalize=True, labelMe=False):
            super().__init__()

            self.files = files

            self.axis_norm = axis_norm
            self.labelMe = labelMe
            self.Normalize = Normalize

        def __len__(self):
            return len(self.files)

        def __getitem__(self, i):

            # Read Raw images
            if self.Normalize is True:
                x = ReadFloat(self.files[i])
                x = normalize(x, 1, 99.8, axis=self.axis_norm)
                x = x
            if self.labelMe is True:
                # Read Label images
                x = read_int(self.files[i])
                x = x
            return x

    def Train(self):

        raw_path = os.path.join(self.base_dir, self.raw_dir)
        raw = os.listdir(raw_path)
        if self.load_data_sequence and self.val_raw_dir is not None:
            val_raw_path = os.path.join(self.base_dir, self.val_raw_dir)
            val_raw = os.listdir(val_raw_path)
            val_real_mask_path = os.path.join(self.base_dir, self.val_real_mask_dir)

            Path(val_real_mask_path).mkdir(exist_ok=True)
            val_real_mask = os.listdir(val_real_mask_path)
        if self.binary_mask_dir is not None:
            mask_path = os.path.join(self.base_dir, self.binary_mask_dir)
            Path(mask_path).mkdir(exist_ok=True)
            mask = os.listdir(mask_path)
        if self.real_mask_dir is not None:
            real_mask_path = os.path.join(self.base_dir, self.real_mask_dir)
            Path(real_mask_path).mkdir(exist_ok=True)
            real_mask = os.listdir(real_mask_path)

        if (
            self.binary_mask_dir is not None
            and self.train_star
            and len(mask) > 0
            and len(real_mask) < len(mask)
        ):

            print("Making labels")
            mask = sorted(
                glob.glob(self.base_dir + self.binary_mask_dir + "*" + self.pattern)
            )

            for fname in mask:
                if any(fname.endswith(f) for f in self.acceptable_formats):
                    image = imread(os.path.join(self.base_dir, fname))

                    Name = os.path.basename(os.path.splitext(fname)[0])
                    if np.max(image) == 1:
                        image = image * 255
                    binary_image = label(image)
                    if self.real_mask_dir is not None:
                        imwrite(
                            (
                                os.path.join(
                                    self.base_dir,
                                    self.real_mask_dir + Name + self.pattern,
                                )
                            ),
                            binary_image.astype("uint16"),
                        )

        if (
            self.real_mask_dir is not None
            and len(real_mask) > 0
            and len(mask) < len(real_mask)
        ):
            print("Generating Binary images")

            real_files_mask = os.listdir(real_mask_path)

            for fname in real_files_mask:
                if any(fname.endswith(f) for f in self.acceptable_formats):
                    image = imread(os.path.join(real_mask_path, fname))
                    if self.erosion_iterations > 0:
                        image = erode_labels(
                            image.astype("uint16"), self.erosion_iterations
                        )
                    Name = os.path.basename(os.path.splitext(fname)[0])

                    binary_image = image > 0
                    if self.binary_mask_dir is not None:
                        imwrite(
                            (
                                os.path.join(
                                    self.base_dir,
                                    self.binary_mask_dir + Name + self.pattern,
                                )
                            ),
                            binary_image.astype("uint16"),
                        )

        if self.generate_npz:
            if self.RGB:
                if self.train_unet:
                    raw_data = RawData.from_folder(
                        basepath=self.base_dir,
                        source_dirs=[self.raw_dir],
                        target_dir=self.binary_mask_dir,
                        axes="YXC",
                    )

                    X, Y, XY_axes = create_patches_reduced_target(
                        raw_data=raw_data,
                        patch_size=(self.patch_y, self.patch_x, None),
                        n_patches_per_image=self.n_patches_per_image,
                        target_axes="YX",
                        reduction_axes="C",
                        save_file=self.base_dir + self.npz_filename + ".npz",
                    )
                if self.train_seed_unet:
                    print("Eroded Masks")
                    raw_data = RawData.from_folder(
                        basepath=self.base_dir,
                        source_dirs=[self.raw_dir],
                        target_dir=self.binary_erode_mask_dir,
                        axes="YXC",
                    )

                    X, Y, XY_axes = create_patches_reduced_target(
                        raw_data=raw_data,
                        patch_size=(self.patch_y, self.patch_x, None),
                        n_patches_per_image=self.n_patches_per_image,
                        target_axes="YX",
                        reduction_axes="C",
                        save_file=self.base_dir + self.npz_filename + "Erode" + ".npz",
                    )

            else:

                if self.train_unet:
                    raw_data = RawData.from_folder(
                        basepath=self.base_dir,
                        source_dirs=[self.raw_dir],
                        target_dir=self.binary_mask_dir,
                        axes="YX",
                    )

                    X, Y, XY_axes = create_patches(
                        raw_data=raw_data,
                        patch_size=(self.patch_y, self.patch_x),
                        n_patches_per_image=self.n_patches_per_image,
                        save_file=self.base_dir + self.npz_filename + ".npz",
                    )

                if self.train_seed_unet:
                    raw_data = RawData.from_folder(
                        basepath=self.base_dir,
                        source_dirs=[self.raw_dir],
                        target_dir=self.binary_erode_mask_dir,
                        axes="YX",
                    )

                    X, Y, XY_axes = create_patches(
                        raw_data=raw_data,
                        patch_size=(self.patch_y, self.patch_x),
                        n_patches_per_image=self.n_patches_per_image,
                        save_file=self.base_dir + self.npz_filename + "Erode" + ".npz",
                    )

        # Training UNET model
        if self.train_unet:
            print("Training UNET model")
            load_path = self.base_dir + self.npz_filename + ".npz"

            (X, Y), (X_val, Y_val), axes = load_training_data(
                load_path, validation_split=self.validation_split, verbose=True
            )
            c = axes_dict(axes)["C"]
            n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

            config = Config(
                axes,
                n_channel_in,
                n_channel_out,
                unet_n_depth=self.depth,
                train_epochs=self.epochs,
                train_batch_size=self.batch_size,
                unet_kern_size=self.kern_size,
                unet_n_first=self.startfilter,
                train_learning_rate=self.learning_rate,
                train_reduce_lr={"patience": 5, "factor": 0.5},
            )
            print(config)
            vars(config)

            model = CARE(config, name="unet_" + self.model_name, basedir=self.model_dir)

            if os.path.exists(
                self.model_dir + "unet_" + self.model_name + "/" + "weights_now.h5"
            ):
                print("Loading checkpoint model")
                model.load_weights(
                    self.model_dir + "unet_" + self.model_name + "/" + "weights_now.h5"
                )

            if os.path.exists(
                self.model_dir + "unet_" + self.model_name + "/" + "weights_last.h5"
            ):
                print("Loading checkpoint model")
                model.load_weights(
                    self.model_dir + "unet_" + self.model_name + "/" + "weights_last.h5"
                )

            if os.path.exists(
                self.model_dir + "unet_" + self.model_name + "/" + "weights_best.h5"
            ):
                print("Loading checkpoint model")
                model.load_weights(
                    self.model_dir + "unet_" + self.model_name + "/" + "weights_best.h5"
                )

            history = model.train(X, Y, validation_data=(X_val, Y_val))
            plot_train_history(
                history,
                self.model_dir,
                "unet_" + self.model_name,
                ["loss", "val_loss"],
                ["mse", "val_mse", "mae", "val_mae"],
            )

        # Training UNET model
        if self.train_seed_unet:
            print("Training Seed UNET model")
            load_path = self.base_dir + self.npz_filename + "Erode" + ".npz"

            (X, Y), (X_val, Y_val), axes = load_training_data(
                load_path, validation_split=self.validation_split, verbose=True
            )
            c = axes_dict(axes)["C"]
            n_channel_in, n_channel_out = X.shape[c], Y.shape[c]

            config = Config(
                axes,
                n_channel_in,
                n_channel_out,
                unet_n_depth=self.depth,
                train_epochs=self.epochs,
                train_batch_size=self.batch_size,
                unet_kern_size=self.kern_size,
                unet_n_first=self.unet_n_first,
                train_learning_rate=self.learning_rate,
                train_reduce_lr={"patience": 5, "factor": 0.5},
            )
            print(config)
            vars(config)

            model = CARE(
                config, name="seed_unet_" + self.model_name, basedir=self.model_dir
            )

            if os.path.exists(
                self.model_dir + "seed_unet_" + self.model_name + "/" + "weights_now.h5"
            ):
                print("Loading checkpoint model")
                model.load_weights(
                    self.model_dir
                    + "seed_unet_"
                    + self.model_name
                    + "/"
                    + "weights_now.h5"
                )

            if os.path.exists(
                self.model_dir
                + "seed_unet_"
                + self.model_name
                + "/"
                + "weights_last.h5"
            ):
                print("Loading checkpoint model")
                model.load_weights(
                    self.model_dir
                    + "seed_unet_"
                    + self.model_name
                    + "/"
                    + "weights_last.h5"
                )

            if os.path.exists(
                self.model_dir
                + "seed_unet_"
                + self.model_name
                + "/"
                + "weights_best.h5"
            ):
                print("Loading checkpoint model")
                model.load_weights(
                    self.model_dir
                    + "seed_unet_"
                    + self.model_name
                    + "/"
                    + "weights_best.h5"
                )

            history = model.train(X, Y, validation_data=(X_val, Y_val))
            plot_train_history(
                history,
                self.model_dir,
                "seed_unet_" + self.model_name,
                ["loss", "val_loss"],
                ["mse", "val_mse", "mae", "val_mae"],
            )

        if self.train_star:
            print("Training StarDistModel model")

            self.axis_norm = (0, 1)  # normalize channels independently

            if self.load_data_sequence is False:
                assert len(raw) > 1, "not enough training data"
                print(len(raw))
                rng = np.random.RandomState(42)
                ind = rng.permutation(len(raw))

                X_train = list(map(ReadFloat, raw))
                Y_train = list(map(read_int, real_mask))

                self.Y = [
                    label(DownsampleData(y, self.downsample_factor))
                    for y in tqdm(Y_train)
                ]
                self.X = [
                    normalize(
                        DownsampleData(x, self.downsample_factor),
                        1,
                        99.8,
                        axis=self.axis_norm,
                    )
                    for x in tqdm(X_train)
                ]
                n_val = max(1, int(round(self.validation_split * len(ind))))
                ind_train, ind_val = ind[:-n_val], ind[-n_val:]

                self.X_val, self.Y_val = [self.X[i] for i in ind_val], [
                    self.Y[i] for i in ind_val
                ]
                self.X_trn, self.Y_trn = [self.X[i] for i in ind_train], [
                    self.Y[i] for i in ind_train
                ]

                print("number of images: %3d" % len(self.X))
                print("- training:       %3d" % len(self.X_trn))
                print("- validation:     %3d" % len(self.X_val))
                self.train_sample_cache = True

            if self.load_data_sequence:
                self.X_trn = self.DataSequencer(
                    raw, self.axis_norm, Normalize=True, labelMe=False
                )
                self.Y_trn = self.DataSequencer(
                    real_mask, self.axis_norm, Normalize=False, labelMe=True
                )

                self.X_val = self.DataSequencer(
                    val_raw, self.axis_norm, Normalize=True, labelMe=False
                )
                self.Y_val = self.DataSequencer(
                    val_real_mask, self.axis_norm, Normalize=False, labelMe=True
                )
                self.train_sample_cache = False

            print(Config2D.__doc__)

            conf = Config2D(
                n_rays=self.n_rays,
                train_epochs=self.epochs,
                train_learning_rate=self.learning_rate,
                unet_n_depth=self.depth,
                train_patch_size=(self.patch_y, self.patch_x),
                n_channel_in=self.n_channel_in,
                unet_n_filter_base=self.unet_n_first,
                train_checkpoint=self.model_dir + self.model_name + ".h5",
                grid=self.grid,
                train_loss_weights=(1, 0.05),
                use_gpu=self.use_gpu,
                train_batch_size=self.batch_size,
                train_sample_cache=self.train_sample_cache,
            )
            print(conf)
            vars(conf)

            Starmodel = StarDist2D(conf, name=self.model_name, basedir=self.model_dir)

            if os.path.exists(
                self.model_dir + self.model_name + "/" + "weights_now.h5"
            ):
                print("Loading checkpoint model")
                Starmodel.load_weights(
                    self.model_dir + self.model_name + "/" + "weights_now.h5"
                )

            if os.path.exists(
                self.model_dir + self.model_name + "/" + "weights_last.h5"
            ):
                print("Loading checkpoint model")
                Starmodel.load_weights(
                    self.model_dir + self.model_name + "/" + "weights_last.h5"
                )

            if os.path.exists(
                self.model_dir + self.model_name + "/" + "weights_best.h5"
            ):
                print("Loading checkpoint model")
                Starmodel.load_weights(
                    self.model_dir + self.model_name + "/" + "weights_best.h5"
                )

            Starhistory = Starmodel.train(
                self.X_trn,
                self.Y_trn,
                validation_data=(self.X_val, self.Y_val),
                epochs=self.epochs,
            )
            plot_train_history(
                Starhistory,
                self.model_dir,
                self.model_name,
                ["loss", "val_loss"],
                [
                    "dist_relevant_mae",
                    "val_dist_relevant_mae",
                    "dist_relevant_mse",
                    "val_dist_relevant_mse",
                ],
            )
            Starmodel.optimize_thresholds(self.X_val, self.Y_val)


def ReadFloat(fname):

    return imread(fname).astype("float32")


def read_int(fname):

    return imread(fname).astype("uint16")


def eroder(fname, erosion_iterations):

    image = imread(fname)
    if erosion_iterations > 0:
        image = erode_labels(image.astype("uint16"), erosion_iterations)
    name = os.path.basename(os.path.splitext(fname)[0])
    Binaryimage = image > 0
    return Binaryimage, name


def binarer(fname):

    image = imread(fname)
    name = os.path.basename(os.path.splitext(fname)[0])
    Binaryimage = image > 0
    return Binaryimage, name


def DownsampleData(image, downsample_factor):

    scale_percent = int(100 / downsample_factor)  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = zoom(image.astype("float32"), dim)

    return image
