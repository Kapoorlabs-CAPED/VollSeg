import numpy as np
import os

# from IPython.display import clear_output
from stardist.models import Config3D, StarDist3D
from stardist import Rays_GoldenSpiral, calculate_extents
from csbdeep.utils import normalize
import glob
from csbdeep.io import load_training_data
from csbdeep.models import Config
from .CARE import CARE
from tensorflow.keras.utils import Sequence
from csbdeep.data import RawData, create_patches
from skimage.measure import label, regionprops
from scipy import ndimage
import matplotlib.pyplot as plt
from pathlib import Path
from tifffile import imread, imwrite
from csbdeep.utils import plot_history
from scipy.ndimage import zoom
from scipy.ndimage import binary_fill_holes


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


class SmartSeeds3D:
    def __init__(
        self,
        base_dir,
        unet_model_name,
        star_model_name,
        unet_model_dir,
        star_model_dir,
        npz_filename=None,
        n_patches_per_image=1,
        train_loss="mae",
        raw_dir="raw/",
        real_mask_dir="real_mask/",
        binary_mask_dir="binary_mask/",
        val_raw_dir="val_raw/",
        val_real_mask_dir="val_real_mask/",
        n_channel_in=1,
        n_channel_out=1,
        pattern=".tif",
        downsample_factor=1,
        backbone="resnet",
        load_data_sequence=True,
        train_unet=True,
        train_star=True,
        generate_npz=True,
        validation_split=0.01,
        erosion_iterations=2,
        patch_size=(16, 256, 256),
        grid_x=1,
        grid_y=1,
        annisotropy=(1, 1, 1),
        use_gpu=True,
        val_size: int = None,
        batch_size=4,
        depth=3,
        kern_size=3,
        startfilter=48,
        n_rays=16,
        epochs=400,
        learning_rate=0.0001,
    ):

        self.npz_filename = npz_filename
        self.base_dir = base_dir
        self.downsample_factor = downsample_factor
        self.unet_model_dir = unet_model_dir
        self.star_model_dir = star_model_dir
        self.backbone = backbone
        self.raw_dir = raw_dir
        self.real_mask_dir = real_mask_dir
        self.val_raw_dir = val_raw_dir
        self.val_real_mask_dir = val_real_mask_dir
        self.binary_mask_dir = binary_mask_dir
        self.generate_npz = generate_npz
        self.annisotropy = annisotropy
        self.train_unet = train_unet
        self.train_star = train_star
        self.unet_model_name = unet_model_name
        self.star_model_name = star_model_name
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.depth = depth
        self.n_channel_in = n_channel_in
        self.n_channel_out = n_channel_out
        self.n_rays = n_rays
        self.pattern = pattern
        self.train_loss = train_loss
        self.erosion_iterations = erosion_iterations
        self.kern_size = kern_size
        self.patch_size = patch_size
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.search_pattern = "*" + self.pattern
        self.startfilter = startfilter
        self.n_patches_per_image = n_patches_per_image
        self.load_data_sequence = load_data_sequence
        self.val_size = val_size
        self.acceptable_formats = [".tif", ".TIFF", ".TIF", ".png"]
        self.axis_norm = (0, 1, 2)
        self.axes = "ZYXC"
        self.Train()

    class UnetSequencer(Sequence):
        def __init__(
            self,
            filesraw,
            filesmask,
            axis_norm,
            batch_size=1,
            shape=(16, 256, 256),
        ):
            super().__init__()

            self.filesraw = filesraw
            self.filesmask = filesmask
            self.axis_norm = axis_norm
            self.batch_size = batch_size
            self.shape = shape

        def __len__(self):
            return len(self.filesraw) // self.batch_size

        def __getitem__(self, idx):

            batch_x = self.filesraw[
                idx * self.batch_size : (idx + 1) * self.batch_size
            ]
            batch_y = self.filesmask[
                idx * self.batch_size : (idx + 1) * self.batch_size
            ]
            rawlist = []
            masklist = []

            for fname in batch_x:
                raw = read_float(fname)
                if raw.shape == self.shape:
                    raw = normalize(raw, 1, 99.8, axis=self.axis_norm)
                    rawlist.append(raw)
            for fname in batch_y:
                mask = read_int(fname)
                if mask.shape == self.shape:
                    mask = mask > 0
                    mask = binary_fill_holes(mask)
                    masklist.append(mask)

            return np.asarray(rawlist, dtype=np.float32), np.asarray(
                masklist, dtype=np.float32
            )

    class DataSequencer(Sequence):
        def __init__(
            self,
            files,
            axis_norm,
            normalize=True,
            label_me=False,
            binary_me=False,
        ):
            super().__init__()

            self.files = files

            self.axis_norm = axis_norm
            self.label_me = label_me
            self.binary_me = binary_me
            self.normalize = normalize

        def __len__(self):
            return len(self.files)

        def __getitem__(self, i):

            # Read raw images
            if self.normalize is True:
                x = read_float(self.files[i])
                x = normalize(x, 1, 99.8, axis=self.axis_norm)
            if self.label_me is True:
                # Read Label images
                x = read_int(self.files[i])
                if self.binary_me:
                    x = x > 0
                x = x.astype(np.uint16)
            return x

    def Train(self):

        raw_path = os.path.join(self.base_dir, self.raw_dir)
        raw = os.listdir(raw_path)
        if self.load_data_sequence:
            val_raw_path = os.path.join(self.base_dir, self.val_raw_dir)
            val_raw = os.listdir(val_raw_path)
            val_real_mask_path = os.path.join(
                self.base_dir, self.val_real_mask_dir
            )

            Path(val_real_mask_path).mkdir(exist_ok=True)
            val_real_mask = os.listdir(val_real_mask_path)

        mask_path = os.path.join(self.base_dir, self.binary_mask_dir)
        Path(mask_path).mkdir(exist_ok=True)
        mask = os.listdir(mask_path)

        real_mask_path = os.path.join(self.base_dir, self.real_mask_dir)
        Path(real_mask_path).mkdir(exist_ok=True)
        real_mask = os.listdir(real_mask_path)

        print("Instance segmentation masks:", len(real_mask))
        print("Semantic segmentation masks:", len(mask))
        if self.train_star and len(mask) > 0 and len(real_mask) < len(mask):

            print("Making labels")
            mask = sorted(
                glob.glob(
                    self.base_dir + self.binary_mask_dir + "*" + self.pattern
                )
            )

            for fname in mask:
                if any(fname.endswith(f) for f in self.acceptable_formats):
                    image = imread(os.path.join(self.base_dir, fname))

                    Name = os.path.basename(os.path.splitext(fname)[0])
                    if np.max(image) == 1:
                        image = image * 255
                    binary_image = label(image)

                    imwrite(
                        (
                            os.path.join(
                                self.base_dir,
                                self.real_mask_dir + Name + self.pattern,
                            )
                        ),
                        binary_image.astype("uint16"),
                    )

        if len(real_mask) > 0 and len(mask) < len(real_mask):
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

            raw_data = RawData.from_folder(
                basepath=self.base_dir,
                source_dirs=[self.raw_dir],
                target_dir=self.binary_mask_dir,
                pattern=self.search_pattern,
                axes="ZYX",
            )

            X, Y, XY_axes = create_patches(
                raw_data=raw_data,
                patch_size=self.patch_size,
                n_patches_per_image=self.n_patches_per_image,
                save_file=os.path.join(
                    self.base_dir, self.npz_filename + ".npz"
                ),
            )

        # Training UNET model
        if self.train_unet:
            print("Training UNET model")

            if not self.load_data_sequence:
                load_path = os.path.join(
                    self.base_dir, self.npz_filename + ".npz"
                )
                (X, Y), (X_val, Y_val), axes = load_training_data(
                    load_path,
                    validation_split=self.validation_split,
                    verbose=True,
                )
                self.axes = axes
            else:
                raw_path_list = []
                for fname in raw:
                    if any(fname.endswith(f) for f in self.acceptable_formats):
                        raw_path_list.append(os.path.join(raw_path, fname))
                val_raw_path_list = []
                for fname in val_raw:
                    if any(fname.endswith(f) for f in self.acceptable_formats):
                        val_raw_path_list.append(
                            os.path.join(val_raw_path, fname)
                        )

                mask_path_list = []
                for fname in mask:
                    if any(fname.endswith(f) for f in self.acceptable_formats):
                        mask_path_list.append(os.path.join(mask_path, fname))
                val_real_mask_path_list = []
                for fname in val_real_mask:
                    if any(fname.endswith(f) for f in self.acceptable_formats):
                        val_real_mask_path_list.append(
                            os.path.join(val_real_mask_path, fname)
                        )

                XY = self.UnetSequencer(
                    raw_path_list,
                    mask_path_list,
                    self.axis_norm,
                    self.batch_size,
                    self.patch_size,
                )

                XY_val = self.UnetSequencer(
                    val_raw_path_list,
                    val_real_mask_path_list,
                    self.axis_norm,
                    self.batch_size,
                    self.patch_size,
                )

            config = Config(
                self.axes,
                self.n_channel_in,
                self.n_channel_out,
                unet_n_depth=self.depth,
                train_epochs=self.epochs,
                train_batch_size=self.batch_size,
                unet_n_first=self.startfilter,
                train_loss=self.train_loss,
                unet_kern_size=self.kern_size,
                train_learning_rate=self.learning_rate,
                train_reduce_lr={"patience": 5, "factor": 0.5},
            )
            print(config)
            vars(config)

            model = CARE(
                config, name=self.unet_model_name, basedir=self.unet_model_dir
            )

            if os.path.exists(
                os.path.join(
                    self.unet_model_dir,
                    os.path.join(self.unet_model_name, "weights_now.h5"),
                )
            ):
                print("Loading checkpoint model")
                model.load_weights(
                    os.path.join(
                        self.unet_model_dir,
                        os.path.join(self.unet_model_name, "weights_now.h5"),
                    )
                )

            if os.path.exists(
                os.path.join(
                    self.unet_model_dir,
                    os.path.join(self.unet_model_name, "weights_last.h5"),
                )
            ):
                print("Loading checkpoint model")
                model.load_weights(
                    os.path.join(
                        self.unet_model_dir,
                        os.path.join(self.unet_model_name, "weights_last.h5"),
                    )
                )

            if os.path.exists(
                os.path.join(
                    self.unet_model_dir,
                    os.path.join(self.unet_model_name, "weights_best.h5"),
                )
            ):
                print("Loading checkpoint model")
                model.load_weights(
                    os.path.join(
                        self.unet_model_dir,
                        os.path.join(self.unet_model_name, "weights_best.h5"),
                    )
                )
            if not self.load_data_sequence:
                history = model.train(
                    X,
                    Y,
                    validation_data=(X_val, Y_val),
                    load_data_sequence=self.load_data_sequence,
                )
            else:
                history = model.train(
                    XY,
                    validation_data=XY_val,
                    load_data_sequence=self.load_data_sequence,
                )

            plt.figure(figsize=(16, 5))
            plot_history(
                history,
                ["loss", "val_loss"],
                ["mse", "val_mse", "mae", "val_mae"],
            )

            print(sorted(list(history.history.keys())))

        if self.train_star:
            print(
                "Training StarDistModel model with", self.backbone, "backbone"
            )

            real_files_mask = os.listdir(real_mask_path)
            if not self.load_data_sequence:
                rng = np.random.RandomState(len(raw) // 2)
                ind = rng.permutation(len(raw))
                self.Y = []
                self.X = []
                for fname in real_files_mask:
                    if any(fname.endswith(f) for f in self.acceptable_formats):
                        self.Y.append(
                            read_int(os.path.join(real_mask_path, fname))
                        )

                for fname in raw:
                    if any(fname.endswith(f) for f in self.acceptable_formats):
                        self.X.append(
                            read_float(os.path.join(raw_path, fname))
                        )

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
                print(Config3D.__doc__)

                extents = calculate_extents(self.Y_trn)
                self.annisotropy = tuple(np.max(extents) / extents)
                rays = Rays_GoldenSpiral(
                    self.n_rays, anisotropy=self.annisotropy
                )
            if self.load_data_sequence:
                rays = self.n_rays
                self.annisotropy = None
                raw_path_list = []
                for fname in raw:
                    if any(fname.endswith(f) for f in self.acceptable_formats):
                        raw_path_list.append(os.path.join(raw_path, fname))
                val_raw_path_list = []
                for fname in val_raw:
                    if any(fname.endswith(f) for f in self.acceptable_formats):
                        val_raw_path_list.append(
                            os.path.join(val_raw_path, fname)
                        )

                self.X_trn = self.DataSequencer(
                    raw_path_list,
                    self.axis_norm,
                    normalize=True,
                    label_me=False,
                )
                real_mask_path_list = []
                for fname in real_mask:
                    if any(fname.endswith(f) for f in self.acceptable_formats):
                        real_mask_path_list.append(
                            os.path.join(real_mask_path, fname)
                        )
                val_real_mask_path_list = []
                for fname in val_real_mask:
                    if any(fname.endswith(f) for f in self.acceptable_formats):
                        val_real_mask_path_list.append(
                            os.path.join(val_real_mask_path, fname)
                        )

                self.Y_trn = self.DataSequencer(
                    real_mask_path_list,
                    self.axis_norm,
                    normalize=False,
                    label_me=True,
                )

                self.X_val = self.DataSequencer(
                    val_raw_path_list,
                    self.axis_norm,
                    normalize=True,
                    label_me=False,
                )
                self.Y_val = self.DataSequencer(
                    val_real_mask_path_list,
                    self.axis_norm,
                    normalize=False,
                    label_me=True,
                )
                self.train_sample_cache = False

            if self.backbone == "resnet":

                conf = Config3D(
                    rays=rays,
                    anisotropy=self.annisotropy,
                    backbone=self.backbone,
                    train_epochs=self.epochs,
                    train_learning_rate=self.learning_rate,
                    resnet_n_blocks=self.depth,
                    train_checkpoint=self.star_model_dir
                    + self.star_model_name
                    + ".h5",
                    resnet_kernel_size=(
                        self.kern_size,
                        self.kern_size,
                        self.kern_size,
                    ),
                    train_patch_size=self.patch_size,
                    train_batch_size=self.batch_size,
                    resnet_n_filter_base=self.startfilter,
                    train_dist_loss="mse",
                    grid=(1, self.grid_y, self.grid_x),
                    use_gpu=self.use_gpu,
                    n_channel_in=self.n_channel_in,
                )

            if self.backbone == "unet":

                conf = Config3D(
                    rays=rays,
                    anisotropy=self.annisotropy,
                    backbone=self.backbone,
                    train_epochs=self.epochs,
                    train_learning_rate=self.learning_rate,
                    unet_n_depth=self.depth,
                    train_checkpoint=self.star_model_dir
                    + self.star_model_name
                    + ".h5",
                    unet_kernel_size=(
                        self.kern_size,
                        self.kern_size,
                        self.kern_size,
                    ),
                    train_patch_size=self.patch_size,
                    train_batch_size=self.batch_size,
                    unet_n_filter_base=self.startfilter,
                    train_dist_loss="mse",
                    grid=(1, self.grid_y, self.grid_x),
                    use_gpu=self.use_gpu,
                    n_channel_in=self.n_channel_in,
                    train_sample_cache=False,
                )

            print(conf)
            vars(conf)

            Starmodel = StarDist3D(
                conf, name=self.star_model_name, basedir=self.star_model_dir
            )
            print(
                Starmodel._axes_tile_overlap("ZYX"),
                os.path.exists(
                    os.path.join(
                        self.star_model_dir,
                        os.path.join(self.star_model_name, "weights_now.h5"),
                    )
                ),
            )

            if os.path.exists(
                os.path.join(
                    self.star_model_dir,
                    os.path.join(self.star_model_name, "weights_now.h5"),
                )
            ):
                print("Loading checkpoint model")
                Starmodel.load_weights(
                    os.path.join(
                        self.star_model_dir,
                        os.path.join(self.star_model_name, "weights_now.h5"),
                    )
                )

            if os.path.exists(
                os.path.join(
                    self.star_model_dir,
                    os.path.join(self.star_model_name, "weights_last.h5"),
                )
            ):
                print("Loading checkpoint model")
                Starmodel.load_weights(
                    os.path.join(
                        self.star_model_dir,
                        os.path.join(self.star_model_name, "weights_last.h5"),
                    )
                )

            if os.path.exists(
                os.path.join(
                    self.star_model_dir,
                    os.path.join(self.star_model_name, "weights_best.h5"),
                )
            ):
                print("Loading checkpoint model")
                Starmodel.load_weights(
                    os.path.join(
                        self.star_model_dir,
                        os.path.join(self.star_model_name, "weights_best.h5"),
                    )
                )

            historyStar = Starmodel.train(
                self.X_trn,
                self.Y_trn,
                validation_data=(self.X_val, self.Y_val),
                epochs=self.epochs,
            )
            print(sorted(list(historyStar.history.keys())))
            plt.figure(figsize=(16, 5))
            plot_history(
                historyStar,
                ["loss", "val_loss"],
                [
                    "dist_relevant_mae",
                    "val_dist_relevant_mae",
                    "dist_relevant_mse",
                    "val_dist_relevant_mse",
                ],
            )


def read_float(fname):

    return imread(fname).astype("float32")


def read_int(fname):

    return imread(fname).astype("uint16")


def DownsampleData(image, downsample_factor):

    scale_percent = int(100 / downsample_factor)  # percent of original size
    width = int(image.shape[2] * scale_percent / 100)
    height = int(image.shape[1] * scale_percent / 100)
    dim = (width, height)
    smallimage = np.zeros([image.shape[0], height, width])
    for i in range(0, image.shape[0]):
        # resize image
        smallimage[i, :] = zoom(image[i, :].astype("float32"), dim)

    return smallimage
