import numpy as np
from torch.utils.data import Dataset
from .Tiles_3D import VolumeSlicer


class PredictTiled(Dataset):

    """
    Dataset of fluorescently labeled cell membranes
    """

    def __init__(
        self,
        image: np.ndarray,
        patch_size=(8, 256, 256),
        patch_step=(2, 64, 64),
    ):

        # Sanity checks
        assert len(patch_size) == 3, "Patch size must be 3-dimensional."

        # Save parameters
        self.image = image
        self.patch_size = patch_size
        self.patch_step = patch_step

        self.tiler = VolumeSlicer(
            self.image.shape,
            voxel_size=self.patch_size,
            voxel_step=self.patch_step,
        )

        self.tiles = self.tiler.split(self.image)

    def __len__(self):

        return len(self.tiles)

    def __getitem__(self, idx):

        tiles_batch = np.array(self.tiles[idx])

        coords_batch = np.array(self.tiler.crops[idx])

        print(tiles_batch.size, coords_batch.size)

        return tiles_batch, coords_batch
