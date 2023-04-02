import numpy as np
from torch.utils.data import Dataset
import itertools
from scipy.ndimage import distance_transform_edt


class PredictTiled(Dataset):

    """
    Dataset of fluorescently labeled cell membranes
    """

    def __init__(
        self,
        image: np.ndarray,
        patch_size=(64, 128, 128),
        overlap=(1, 16, 16),
        crop=(2, 32, 32),
    ):

        # Sanity checks
        assert len(patch_size) == 3, "Patch size must be 3-dimensional."

        # Save parameters
        self.image = image
        self.patch_size = patch_size
        self.overlap = overlap
        self.crop = crop

        self.set_data()

    def get_fading_map(self):

        fading_map = np.ones(self.patch_size)

        if all([c == 0 for c in self.crop]):
            self.crop = [1, 1, 1]

        # Exclude crop region
        crop_masking = np.zeros_like(fading_map)
        crop_masking[
            self.crop[0] : self.patch_size[0] - self.crop[0],
            self.crop[1] : self.patch_size[1] - self.crop[1],
            self.crop[2] : self.patch_size[2] - self.crop[2],
        ] = 1
        fading_map = fading_map * crop_masking

        fading_map = distance_transform_edt(fading_map).astype(np.float32)

        # Normalize
        fading_map = fading_map / fading_map.max()

        return fading_map

    def set_data_idx(self, idx):

        self.data_idx = idx
        self.data_shape = self.image.shape
        # Calculate the position of each tile
        locations = []
        for i, p, o, c in zip(
            self.data_shape, self.patch_size, self.overlap, self.crop
        ):
            # get starting coords
            coords = (
                np.arange(
                    np.ceil((i + o + c) / np.maximum(p - o - 2 * c, 1)), dtype=np.int16
                )
                * np.maximum(p - o - 2 * c, 1)
                - o
                - c
            )
            locations.append(coords)
        self.locations = list(itertools.product(*locations))
        self.global_crop_before = np.abs(np.min(np.array(self.locations), axis=0))
        self.global_crop_after = (
            np.array(self.data_shape)
            - np.max(np.array(self.locations), axis=0)
            - np.array(self.patch_size)
        )

    def __len__(self):

        return len(self.locations)

    def __getitem__(self, idx):

        self.patch_start = np.array(self.locations[idx])
        self.patch_end = self.patch_start + np.array(self.patch_size)

        pad_before = np.maximum(-self.patch_start, 0)
        pad_after = np.maximum(self.patch_end - np.array(self.data_shape), 0)
        pad_width = list(zip(pad_before, pad_after))

        slicing = tuple(map(slice, np.maximum(self.patch_start, 0), self.patch_end))

        sample = {}

        image_tmp = self.image
        image_tmp = image_tmp[slicing]
        # Pad if neccessary
        image_tmp = np.pad(image_tmp, pad_width, mode="reflect")
        self.image = image_tmp

        sample["image"] = self.image

        return sample
