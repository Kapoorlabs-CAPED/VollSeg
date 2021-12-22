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
from .UNET2D import  UNET2D 
from .UNET3D import  UNET3D 
from csbdeep.models import CARE



clear_models_and_aliases(StarDist2D, StarDist3D, Unet2D, Unet3D, CARE)
register_model(StarDist2D,   '2D_cells', 'https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_fluo.zip', '8db40dacb5a1311b8d2c447ad934fb8a')
register_model(StarDist3D,   '3D_cells',  'https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_fluo.zip', '8db40dacb5a1311b8d2c447ad934fb8a')
register_model(Unet2D,   'Unet_2D_cells', 'https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_fluo.zip', '8db40dacb5a1311b8d2c447ad934fb8a')
register_model(Unet3D,   'Unet_3D_cells',  'https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_fluo.zip', '8db40dacb5a1311b8d2c447ad934fb8a')
register_model(CARE,   'Denoise_3D_cells',  'https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_fluo.zip', '8db40dacb5a1311b8d2c447ad934fb8a')              
register_model(CARE,   'Denoise_2D_cells',  'https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_fluo.zip', '8db40dacb5a1311b8d2c447ad934fb8a') 
register_model(StarDist2D,   '2D_cells_sec', 'https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_fluo.zip', '8db40dacb5a1311b8d2c447ad934fb8a')
register_model(StarDist3D,   '3D_cells_sec',  'https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_fluo.zip', '8db40dacb5a1311b8d2c447ad934fb8a')
register_model(Unet2D,   'Unet_2D_cells_sec', 'https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_fluo.zip', '8db40dacb5a1311b8d2c447ad934fb8a')
register_model(Unet3D,   'Unet_3D_cells_sec',  'https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_fluo.zip', '8db40dacb5a1311b8d2c447ad934fb8a')
register_model(CARE,   'Denoise_3D_cells_sec',  'https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_fluo.zip', '8db40dacb5a1311b8d2c447ad934fb8a')              
register_model(CARE,   'Denoise_2D_cells_sec',  'https://github.com/stardist/stardist-models/releases/download/v0.1/python_2D_versatile_fluo.zip', '8db40dacb5a1311b8d2c447ad934fb8a') 

register_aliases(StarDist2D, '2D_cells', 'Alias some model zoo 2D')
register_aliases(StarDist3D, '3D_cells',   'Alias some model zoo 3D')
register_aliases(Unet2D, 'Unet_2D_cells',  'Alias Unet some model zoo 2D')
register_aliases(Unet3D, 'Unet_3D_cells',  'Alias Unet some model zoo 3D')
register_aliases(CARE, 'Denoise_2D_cells',  'Alias Unet some model zoo 2D den')
register_aliases(CARE, 'Denoise_3D_cells',  'Alias Unet some model zoo 3D den ')
register_aliases(StarDist2D, '2D_cells_sec', 'Alias some model zoo 2D sec')
register_aliases(StarDist3D, '3D_cells_sec',   'Alias some model zoo 3D sec')
register_aliases(Unet2D, 'Unet_2D_cells_sec',  'Alias Unet some model zoo 2D sec')
register_aliases(Unet3D, 'Unet_3D_cells_sec',  'Alias Unet some model zoo 3D sec')
register_aliases(CARE, 'Denoise_2D_cells_sec',  'Alias Unet some model zoo 2D den sec')
register_aliases(CARE, 'Denoise_3D_cells_sec',  'Alias Unet some model zoo 3D den sec')

del register_model, register_aliases, clear_models_and_aliases
