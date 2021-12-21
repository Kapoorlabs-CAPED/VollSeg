from .SmartSeeds3D import SmartSeeds3D
from .SmartSeeds2D import SmartSeeds2D
from .helpers import *
from .helpers import SegCorrect
from .OptimizeThreshold import OptimizeThreshold
from .Augmentation3D import Augmentation3D
from .Augmentation2D import Augmentation2D
from .IO import imread, imsave
from .spatial_image import *
from .inrimage import *
from .klb import *
from .h5 import *
from .tif import *
from .folder import *
from .pretrained import register_model, register_aliases, clear_models_and_aliases
from csbdeep.models import CARE as Unet2D 
from csbdeep.models import CARE as Unet3D
from csbdeep.models import CARE
clear_models_and_aliases(StarDist2D, StarDist3D, Unet2D, Unet3D, CARE)
register_model(StarDist2D,   '2D_cells', 'some model zoo 2D', 'somehash')
register_model(StarDist3D,   '3D_cells',  'somemodel zoo 3D', 'somehash3d')
register_model(Unet2D,   'Unet_2D_cells', 'Unet some model zoo 2D', 'unetsomehash')
register_model(Unet3D,   'Unet_3D_cells',  'Unet somemodel zoo 3D', 'unetsomehash3d')
register_model(CARE,   'Denoise_3D_cells',  'Unet somemodel zoo 3D', 'unetsomehash3d')              
register_model(CARE,   'Denoise_2D_cells',  'Unet somemodel zoo 3D', 'unetsomehash3d') 


register_aliases(StarDist2D, '2D_cells', 'Alias some model zoo 2D')
register_aliases(StarDist3D, '3D_cells',   'Alias some model zoo 3D')
register_aliases(Unet2D, 'Unet_2D_cells',  'Alias Unet some model zoo 2D')
register_aliases(Unet3D, 'Unet_3D_cells',  'Alias Unet some model zoo 3D')
register_aliases(CARE, 'Denoise_2D_cells',  'Alias Unet some model zoo 2D')
register_aliases(CARE, 'Denoise_3D_cells',  'Alias Unet some model zoo 3D')

del register_model, register_aliases, clear_models_and_aliases
