from torch.utils.data import Dataset
import os
import h5py
import csv
import numpy as np
from skimage import io


class TrainTiled(Dataset):
    def __init__(
        self,
        list_path,
        data_root="",
        patch_size=(64, 128, 128),
        samples_per_epoch=-1,
        image_groups=("data/image",),
        mask_groups=("data/distance", "data/seeds", "data/boundary"),
        patches_from_fg=0.0,
        dist_handling="bool_inv",
        correspondence=True,
        no_img=False,
        no_mask=False,
    ):

        # Sanity checks
        assert len(patch_size) == 3, "Patch size must be 3-dimensional."

        # Save parameters
        self.data_root = data_root
        self.list_path = list_path
        self.patch_size = patch_size
        self.patches_from_fg = patches_from_fg
        self.dist_handling = dist_handling
        self.correspondence = correspondence
        self.no_img = no_img
        self.no_mask = no_mask
        self.samples_per_epoch = samples_per_epoch
        self.axis_norm = (0, 1, 2)
        # Read the filelist and construct full paths to each file
        self.image_groups = image_groups
        self.mask_groups = mask_groups
        self.data_list = self._read_list()

    def test(self, test_folder="", num_files=20):

        os.makedirs(test_folder, exist_ok=True)

        for i in range(num_files):
            test_sample = self.__getitem__(i % self.__len__())

            if not self.no_img:
                for num_img in range(test_sample["image"].shape[0]):
                    io.imsave(
                        os.path.join(test_folder, f"test_img{i}_group{num_img}.tif"),
                        test_sample["image"][num_img, ...],
                        check_contrast=False,
                    )

            if not self.no_mask:
                for num_mask in range(test_sample["mask"].shape[0]):
                    io.imsave(
                        os.path.join(test_folder, f"test_mask{i}_group{num_mask}.tif"),
                        test_sample["mask"][num_mask, ...],
                        check_contrast=False,
                    )

    def _read_list(self):

        # Read the filelist and create full paths to each file
        filelist = []
        with open(self.list_path) as f:
            reader = csv.reader(f, delimiter=";")
            for row in reader:
                if len(row) == 0 or np.sum([len(r) for r in row]) == 0:
                    continue
                row = [os.path.join(self.data_root, r) for r in row]
                filelist.append(row)

        return filelist

    def __len__(self):

        if self.samples_per_epoch < 1 or self.samples_per_epoch is None:
            return len(self.data_list)
        else:
            return self.samples_per_epoch

    def _normalize(self, data, group_name):

        # Normalization

        if "distance" in group_name:

            if self.dist_handling == "bool":
                data = data < 0
            elif self.dist_handling == "bool_inv":
                data = data >= 0

        return data

    def __getitem__(self, idx):

        idx = idx % len(self.data_list)

        # Get the paths to the image and mask
        filepath = self.data_list[idx]

        # Permute patch dimensions
        patch_size = list(self.patch_size)
        if self.permute_dim:
            swaps = np.random.choice(range(3), 2, replace=False)
            patch_size[swaps[0]], patch_size[swaps[1]] = (
                patch_size[swaps[1]],
                patch_size[swaps[0]],
            )

        sample = {}

        if not self.no_mask:

            # Load the mask patch
            mask = np.zeros(
                (len(self.mask_groups),) + self.patch_size, dtype=np.float32
            )
            with h5py.File(filepath[1], "r") as f_handle:
                for num_group, group_name in enumerate(self.mask_groups):

                    mask_tmp = f_handle[group_name]

                    # determine the patch position for the first mask
                    if num_group == 0:

                        # obtain patch position from foreground indices or random
                        if self.patches_from_fg > np.random.random():
                            fg_indices = np.where(mask_tmp)
                            rnd_start = [
                                np.maximum(
                                    0, f[np.random.randint(len(fg_indices[0]))] - p
                                )
                                for f, p in zip(fg_indices, patch_size)
                            ]
                        else:
                            rnd_start = [
                                np.random.randint(
                                    0, np.maximum(1, mask_dim - patch_dim)
                                )
                                for patch_dim, mask_dim in zip(
                                    patch_size, mask_tmp.shape
                                )
                            ]
                        rnd_end = [
                            start + patch_dim
                            for start, patch_dim in zip(rnd_start, patch_size)
                        ]
                        slicing = tuple(map(slice, rnd_start, rnd_end))

                    # extract the patch
                    mask_tmp = mask_tmp[slicing]

                    # Pad if neccessary
                    pad_width = [
                        (0, np.maximum(0, p - i))
                        for p, i in zip(patch_size, mask_tmp.shape)
                    ]
                    mask_tmp = np.pad(mask_tmp, pad_width, mode="reflect")

                    # Permute dimensions
                    if self.permute_dim:
                        mask_tmp = np.swapaxes(mask_tmp, *swaps)
                        # sanity check
                        assert mask_tmp.shape == tuple(
                            self.patch_size
                        ), f"Mask dimension missmatch after rotation. {mask_tmp.shape} instead of {self.patch_size}."

                    if "flow" in group_name and self.permute_dim:
                        num_group = (
                            num_group
                            if not num_group - 1 in swaps
                            else int(swaps[swaps != num_group - 1] + 1)
                        )

                    # Store current mask
                    mask[num_group, ...] = mask_tmp

                mask = mask.astype(np.float32)

                sample["mask"] = mask

        if not self.no_img:

            # Load the image patch
            image = np.zeros(
                (len(self.image_groups),) + self.patch_size, dtype=np.float32
            )
            with h5py.File(filepath[0], "r") as f_handle:
                for num_group, group_name in enumerate(self.image_groups):

                    image_tmp = f_handle[group_name]
                    # Check if positioning  have to be reset
                    reset = (self.no_mask or not self.correspondence) and num_group == 0

                    # Determine the patch position
                    if reset:
                        rnd_start = [
                            np.random.randint(0, np.maximum(1, image_dim - patch_dim))
                            for patch_dim, image_dim in zip(patch_size, image_tmp.shape)
                        ]
                        rnd_end = [
                            start + patch_dim
                            for start, patch_dim in zip(rnd_start, patch_size)
                        ]
                        slicing = tuple(map(slice, rnd_start, rnd_end))
                    image_tmp = image_tmp[slicing].astype(np.float32)

                    # Pad if neccessary
                    pad_width = [
                        (0, np.maximum(0, p - i))
                        for p, i in zip(patch_size, image_tmp.shape)
                    ]
                    image_tmp = np.pad(image_tmp, pad_width, mode="reflect")

                    # Permute dimensions
                    if self.permute_dim:
                        image_tmp = np.swapaxes(image_tmp, *swaps)
                        # sanity check
                        assert image_tmp.shape == tuple(
                            self.patch_size
                        ), f"Image dimension missmatch after rotation. {image.shape} instead of {self.patch_size}."

                    image[num_group, ...] = image_tmp

                image = image.astype(np.float32)

                sample["image"] = image

        return sample
