from .SmartSeeds3D import SmartSeeds3D
from .SmartSeeds2D import SmartSeeds2D
from .seedpool import SeedPool
from .unetstarmask import UnetStarMask
from .nmslabel import NMSLabel
from .OptimizeThreshold import OptimizeThreshold
from .pretrained import (
    register_model,
    register_aliases,
    clear_models_and_aliases,
)
from .UNET import UNET
from .MASKUNET import MASKUNET
from .StarDist2D import StarDist2D
from .StarDist3D import StarDist3D
from .Projection3D import Projection3D
from .CARE import CARE
from .SmartPatches import SmartPatches
from .SimplePatches import SimplePatches
from csbdeep.utils.tf import keras_import
from .CellPose import CellPose
from .PredictTiledLoader import PredictTiled
from .CellPose3D import CellPose3DModel, CellPose3DTrain, CellPose3DPredict
from .TrainTiledLoader import TrainTiled
from .UNet3D import AbstractUNet, UNet3D, ResidualUNet3D
from .se import ChannelSELayer3D, ChannelSpatialSELayer3D, SpatialSELayer3D
from .Tiles_3D import VolumeSlicer
from .utils import (
    VollCellPose3D,
    VollCellSeg,
    VollSeg,
    VollSeg2D,
    VollSeg3D,
    VollSeg_unet,
    VollSeg_nolabel_precondition,
    VollSeg_label_expansion,
    VollSeg_label_precondition,
    VollSeg_nolabel_expansion,
    merge_labels_across_volume,
    SimplePrediction,
    Skel,
    SmartSkel,
    STARPrediction3D,
    SuperSTARPrediction,
    SuperUNETPrediction,
    SuperWatershedwithMask,
    CellPose3DWater,
    CCLabels,
    CellPoseWater,
    CleanCellPose,
    CleanMask,
    VollOne,
)

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

get_file = keras_import("utils", "get_file")


__all__ = (
    "ResidualUNet3D",
    "SmartSeeds3D",
    "SmartSeeds2D",
    "SeedPool",
    "UnetStarMask",
    "NMSLabel",
    "OptimizeThreshold",
    "UNET",
    "MASKUNET",
    "StarDist2D",
    "StarDist3D",
    "Projection3D",
    "SimplePatches",
    "CARE",
    "SmartPatches",
    "CellPose",
    "UNet3D_cellpose",
    "PredictTiled",
    "CellPose3DModel",
    "TrainTiled",
    "CellPose3DTrain",
    "AbstractUNet",
    "UNet3D",
    "ChannelSELayer3D",
    "ChannelSpatialSELayer3D",
    "SpatialSELayer3D",
    "CellPose3DPredict",
    "VolumeSlicer",
    "VollCellPose3D",
    "VollCellSeg",
    "VollSeg",
    "VollSeg2D",
    "VollSeg3D",
    "VollSeg_unet",
    "VollSeg_nolabel_precondition",
    "VollSeg_label_expansion",
    "VollSeg_label_precondition",
    "VollSeg_nolabel_expansion",
    "merge_labels_across_volume",
    "SimplePrediction",
    "Skel",
    "SmartSkel",
    "STARPrediction3D",
    "SuperSTARPrediction",
    "SuperUNETPrediction",
    "SuperWatershedwithMask",
    "CellPose3DWater",
    "CCLabels",
    "CellPoseWater",
    "CleanCellPose",
    "CleanMask",
    "VollOne",
)

clear_models_and_aliases(StarDist2D, StarDist3D, UNET, CARE, MASKUNET)

register_model(
    CARE,
    "Denoise_3D_cells",
    "https://zenodo.org/record/6671170/files/GenericDenoising3D.zip",
    "a0eb25ffd794e2b3b31a4de5b72a392f",
)
register_model(
    CARE,
    "Denoise_carcinoma",
    "https://zenodo.org/record/5910645/files/denoise_carcinoma.zip",
    "fd33199738f0b17761272118cbffdf04",
)
register_model(
    UNET,
    "Embryo Cell Model (3D)",
    "https://zenodo.org/record/6337699/files/embryo_cell_model.zip",
    "c84fdec38a5b3cc6c1869c94ff23f3ba",
)
register_model(
    UNET,
    "Xenopus Tissue (2D)",
    "https://zenodo.org/record/6060378/files/Xenopus_tissue_model.zip",
    "2694d8b05fa828aceb055eef8cd5ca1f",
)
register_model(
    StarDist2D,
    "White_Blood_Cells",
    "https://zenodo.org/record/5815521/files/WBCSeg.zip",
    "7889f5902d8562766a4dee2726c90d49",
)
register_model(
    StarDist3D,
    "Carcinoma_cells",
    "https://zenodo.org/record/6354077/files/carcinoma_stardist.zip",
    "b92b9d5347862e52279629be575fe0b7",
)
register_model(
    UNET,
    "Microtubule Kymograph Segmentation",
    "https://zenodo.org/record/6355705/files/microtubule_kymograph_segmentation.zip",
    "a42fcd4ba732734d36eda3dbbb3d5673",
)
register_model(
    UNET,
    "Unet_White_Blood_Cells",
    "https://zenodo.org/record/5815588/files/UNETWBC.zip",
    "9645f004db478f661811d6da615ccc0b",
)
register_model(
    MASKUNET,
    "Xenopus_Cell_Tissue_Segmentation",
    "https://zenodo.org/record/6060378/files/Xenopus_tissue_model.zip",
    "2694d8b05fa828aceb055eef8cd5ca1f",
)
register_model(
    MASKUNET,
    "Unet_Arabidopsis_Mask",
    "https://zenodo.org/record/6670732/files/Unet_Arabidopsis_Mask.zip",
    "114df78e0153b39d80d0253a4dcc236f",
)
register_model(
    UNET,
    "Unet_Arabidopsis",
    "https://zenodo.org/record/6670747/files/Unet_Arabidopsis.zip",
    "ed7bdead6ebb11c3e13c22a156288f60",
)
register_model(
    UNET,
    "Unet_Cyto_White_Blood_Cells",
    "https://zenodo.org/record/5815603/files/UNETcytoWBC.zip",
    "dd3bf8b8e2a04536144954e882445a5e",
)
register_model(
    UNET,
    "Unet_Lung_Segmentation",
    "https://zenodo.org/record/6060177/files/Montgomery_county.zip",
    "be41937a00693e28961358440d242417",
)


register_aliases(UNET, "Embryo Cell Model (3D)", "Embryo Cell Model (3D)")
register_aliases(StarDist2D, "White_Blood_Cells", "White_Blood_Cells")
register_aliases(StarDist3D, "Carcinoma_cells", "Carcinoma_cells")
register_aliases(UNET, "Unet_White_Blood_Cells", "Unet_White_Blood_Cells")
register_aliases(
    UNET, "Unet_Cyto_White_Blood_Cells", "Unet_Cyto_White_Blood_Cells"
)
register_aliases(
    UNET,
    "Microtubule Kymograph Segmentation",
    "Microtubule Kymograph Segmentation",
)
register_aliases(UNET, "Xenopus Tissue (2D)", "Xenopus Tissue (2D)")
register_aliases(UNET, "Unet_Lung_Segmentation", "Unet_Lung_Segmentation")
register_aliases(UNET, "Unet_Arabidopsis", "Unet_Arabidopsis")
register_aliases(
    MASKUNET,
    "Xenopus_Cell_Tissue_Segmentation",
    "Xenopus_Cell_Tissue_Segmentation",
)
register_aliases(MASKUNET, "Unet_Arabidopsis_Mask", "Unet_Arabidopsis_Mask")
register_aliases(CARE, "Denoise_3D_cells", "Denoise_3D_cells")


del register_model, register_aliases, clear_models_and_aliases


def abspath(path):
    import os

    base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, path)


def test_image_ascadian_3d():
    from tifffile import imread

    url = "https://zenodo.org/record/5965906/files/Astec-Pm2_fuse_t001.tif"
    hash = "fdf1b78bc4ce4817000d1846db226118"

    img = imread(
        abspath(get_file(fname="Ascadian", origin=url, file_hash=hash))
    )
    return img


def test_image_arabidopsis_3d():
    from tifffile import imread

    url = "https://zenodo.org/record/6670569/files/04.tif"
    hash = "68204a6c871d6eeca9870728bfd1b8b7"

    img = imread(
        abspath(get_file(fname="Arabidopsis", origin=url, file_hash=hash))
    )
    return img


def test_image_carcinoma_3dt():
    from tifffile import imread

    url = "https://zenodo.org/record/6403439/files/carcinoma_xyzt.tif"
    hash = "713911848cf5263393e479d5cb3e5d59"
    img = imread(
        abspath(get_file(fname="Carcinoma", origin=url, file_hash=hash))
    )
    return img
