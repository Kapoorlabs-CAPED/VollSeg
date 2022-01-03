import setuptools
from setuptools import find_packages, setup

with open('README.md') as f:
    long_description = f.read()


setup(
    name="vollseg",

    version='2.5.3',

    author='Varun Kapoor,Claudia Carabana Garcia,Mari Tolonen',
    author_email='randomaccessiblekapoor@gmail.com',
    url='https://github.com/kapoorlab/vollseg/',
    description='Segmentation tool for biological cells of irregular size and shape in 3D and 2D.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        
        "pandas",
        "stardist>=0.7.0",
        "numpy==1.20.0",
        "scipy",
        "tifffile",
        "matplotlib",
        "imagecodecs",
        "napari",
        "diplib",
        "opencv-python" 
       
    ],
    dependeny_links = ['https://github.com/bhoeckendorf/pyklb.git@skbuild'],
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.9',
    ],
)
