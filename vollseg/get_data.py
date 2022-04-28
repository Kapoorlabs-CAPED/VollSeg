
from dask.array.image import imread as daskread

def abspath(path):
    import os
    base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, path)


def get_test_data():
    
    image = daskread(abspath('data/carcinoma_xyzt.tif'))[0]
    return image

def get_stardist_modelpath():

    return abspath('models/Carcinoma_cells/')

def get_maskunet_modelpath():    

   return abspath('models/Roi_Nuclei_Xenopus/')


def get_denoising_modelpath():

       return abspath('models/denoise_carcinoma/')