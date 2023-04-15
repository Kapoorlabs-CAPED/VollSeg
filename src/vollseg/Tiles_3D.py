from typing import Tuple, Union

__all__ = ["VolumeSlicer"]
import numpy as np
from scipy.ndimage import distance_transform_edt
import itertools


class VolumeSlicer:
    """
    Helper class to slice 3d volume into smaller volumes and merge them back
    """

    def __init__(
        self,
        data,
        patch_size: Union[int, Tuple[int, int, int]],
        overlap: Tuple[int, int, int],
        crop: Tuple[int, int, int],
    ):
        """
        :param volume_shape: Shape of the source image (D, H, W, C)
        :param patch_size: Tile size (Scalar or tuple (D, H, W, C)
        :param voxel_step: Step in pixels between voxels (Scalar or tuple (D, H, W))
        :param weight: Fusion algorithm. 'mean' - avergaing
        """
        self.data = data
        self.data_shape = np.array(data.shape)

        if isinstance(patch_size, (tuple, list)):
            if len(patch_size) != 3:
                raise ValueError()
            self.patch_size = np.array(patch_size, dtype=int)
        else:
            self.patch_size = np.array([int(patch_size)] * 3)

        if isinstance(overlap, (tuple, list)):
            if len(overlap) != 3:
                raise ValueError()
            self.overlap = np.array(overlap, dtype=int)
        else:
            self.overlap = np.array([int(overlap)] * 3)

        if isinstance(crop, (tuple, list)):
            if len(crop) != 3:
                raise ValueError()
            self.crop = np.array(crop, dtype=int)
        else:
            self.crop = np.array([int(crop)] * 3)

        assert all(
            [
                p - 2 * o - 2 * c > 0
                for p, o, c in zip(self.patch_size, self.overlap, self.crop)
            ]
        ), "Invalid combination of patch size, overlap and crop size."
        # Calculate the position of each tile
        locations = []
        for i, p, o, c in zip(
            self.data_shape, self.patch_size, self.overlap, self.crop
        ):
            # get starting coords
            coords = (
                np.arange(
                    np.ceil((i + o + c) / np.maximum(p - o - 2 * c, 1)),
                    dtype=np.int16,
                )
                * np.maximum(p - o - 2 * c, 1)
                - o
                - c
            )
            locations.append(coords)
        self.locations = list(itertools.product(*locations))
        self.global_crop_before = np.abs(
            np.min(np.array(self.locations), axis=0)
        )
        self.global_crop_after = (
            np.array(self.data_shape)
            - np.max(np.array(self.locations), axis=0)
            - np.array(self.patch_size)
        )

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
        self.fading_map = fading_map / fading_map.max()

    def split(self, idx):

        self.patch_start = np.array(self.locations[idx])
        self.patch_end = self.patch_start + np.array(self.patch_size)
        pad_before = np.maximum(-self.patch_start, 0)
        pad_after = np.maximum(self.patch_end - np.array(self.data_shape), 0)
        pad_width = list(zip(pad_before, pad_after))

        slicing = tuple(
            map(slice, np.maximum(self.patch_start, 0), self.patch_end)
        )
        self.tile = self.data[slicing]
        self.tile = np.pad(self.tile, pad_width, mode="reflect")
