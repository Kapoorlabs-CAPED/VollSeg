import numpy as np
from torch.utils.data import Dataset
from .Tiles_3D import VolumeSlicer


class PredictTiled(Dataset):

    """
    Dataset of fluorescently labeled cell membranes
    """

    def __init__(
        self,
        tiler: VolumeSlicer,
        patch_size=(8, 256, 256),
        overlap=(1, 16, 16),
        crop=(2, 32, 32),
    ):

        # Sanity checks
        assert len(patch_size) == 3, "Patch size must be 3-dimensional."
        # Save parameters
        self.tiler = tiler
        self.patch_size = patch_size
        self.overlap = overlap
        self.crop = crop

    def __len__(self):

        return len(self.tiler.locations)

    def __getitem__(self, idx):

        self.tiler.split(idx)

        return (
            self.tiler.tile[np.newaxis, ...],
            self.tiler.patch_start,
            self.tiler.patch_end,
        )
