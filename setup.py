import setuptools
from setuptools import find_packages, setup

with open('README.md') as f:
    long_description = f.read()


setup(
    name="vollseg",

    version='1.6.6',

    author='Varun Kapoor,Claudia Carabana Garcia,Mari Tolonen',
    author_email='randomaccessiblekapoor@gmail.com',
    url='https://github.com/kapoorlab/vollseg/',
    description='Segmentation tool for biological cells of irregular size and shape in 3D and 2D.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        "numpy",
        "pandas",
        "csbdeep",
        "stardist",
        "scikit-image",
        "scipy",
        "tifffile",
        "matplotlib",
        "imagecodecs",
        "n2v",
        
    ],
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
    ],
)
