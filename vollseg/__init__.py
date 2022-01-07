from .SmartSeeds3D import SmartSeeds3D
from .SmartSeeds2D import SmartSeeds2D
from .helpers import *
from .helpers import SegCorrect
from .OptimizeThreshold import OptimizeThreshold
from .Augmentation3D import Augmentation3D
from .Augmentation2D import Augmentation2D
from .spatial_image import *
from .inrimage import *
#from .klb import *
from .h5 import *
from .tif import *
from .folder import *
from .pretrained import register_model, register_aliases, clear_models_and_aliases
from .UNET import  UNET 
from .StarDist2D import StarDist2D
from .StarDist3D import StarDist3D
from csbdeep.models import CARE



clear_models_and_aliases(StarDist2D, StarDist3D, UNET, CARE)

register_model(CARE,   'Denoise_3D_cells',  'https://zenodo.org/record/5813521/files/GenericDenoising3D.zip', 'be8dffd239193361a9c289090425dd12')              

register_model(StarDist3D,   'Ascadian_Embryo_Model_A',  'https://zenodo.org/record/5825801/files/Ascadian_large_td.zip', 'f59ea5a15eebc9832aded8e4f40e7298')

register_model(UNET,   'Unet_Ascadian_Embryo_Model_A',  'https://zenodo.org/record/5825808/files/UnetAscadian_large_td.zip', 'bd04a44a43516dd901d6317de65a6cab')

register_model(StarDist2D,   'White_Blood_Cells',  'https://zenodo.org/record/5815521/files/WBCSeg.zip', '7889f5902d8562766a4dee2726c90d49')

register_model(UNET,   'Unet_White_Blood_Cells',  'https://zenodo.org/record/5815588/files/UNETWBC.zip', '9645f004db478f661811d6da615ccc0b')

register_model(UNET,   'Unet_Cyto_White_Blood_Cells',  'https://zenodo.org/record/5815603/files/UNETcytoWBC.zip', 'dd3bf8b8e2a04536144954e882445a5e') 


register_aliases(StarDist3D, 'Ascadian_Embryo_Model_A',   'Ascadian_Embryo_Model_A')
register_aliases(UNET, 'Unet_Ascadian_Embryo_Model_A',  'Unet_Ascadian_Embryo_Model_A')
register_aliases(StarDist2D, 'White_Blood_Cells',   'White_Blood_Cells')
register_aliases(UNET, 'Unet_White_Blood_Cells',  'Unet_White_Blood_Cells')
register_aliases(UNET, 'Unet_Cyto_White_Blood_Cells',  'Unet_Cyto_White_Blood_Cells')
register_aliases(CARE, 'Denoise_3D_cells',  'Denoise_3D_cells')



del register_model, register_aliases, clear_models_and_aliases
