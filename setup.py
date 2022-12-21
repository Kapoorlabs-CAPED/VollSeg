from setuptools import find_packages, setup

from os import path

_dir = path.dirname(__file__)


with open('README.md') as f:
    long_description = f.read()
with open(path.join(_dir,'vollseg','_version.py'), encoding="utf-8") as f:
    exec(f.read())

setup(
    name="vollseg",

    version=__version__,

    author='Varun Kapoor,Claudia Carabana Garcia,Mari Tolonen',
    author_email='randomaccessiblekapoor@gmail.com',
    url='https://github.com/kapoorlab/vollseg/',
    description='Segmentation tool for biological cells of irregular size and shape in 3D and 2D.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        
        "pandas",
        "stardist",
        "scipy",
        "tifffile",
        "matplotlib",
        "napari",
        "cellpose",
       
    ],
    
    packages = find_packages(),
    package_data={'vollseg': [ 'data/*','models/Carcinoma_cells/Carcinoma_cells/*', 'models/denoise_carcinoma/Denoise_carcinoma/*', 'models/Roi_Nuclei_Xenopus/Xenopus_Cell_Tissue_Segmentation/*' ]},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.9',
    ],
)
