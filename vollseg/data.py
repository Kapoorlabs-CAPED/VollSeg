from tifffile import imread


def abspath(path):
    import os
    base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, path)


def get_test_data():
    
    image = imread(abspath('data/carcinoma_xyzt.tif'))
    return image

def get_stardist_modelpath():

    return abspath('models/Carcinoma_cells/')

def get_maskunet_modelpath():    

   return abspath('models/Roi_Nuclei_Xenopus/')


def get_denoising_modelpath():

       return abspath('models/denoise_carcinoma/')