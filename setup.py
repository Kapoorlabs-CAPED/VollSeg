from setuptools import find_packages, setup
from os import path

try:
    from ... import version as __version__
except ImportError:
    __version__ = "unknown"
_dir = path.dirname(__file__)


with open("README.md") as f:
    long_description = f.read()
with open(
    path.join(_dir, "src/vollseg", "_version.py"), encoding="utf-8"
) as f:
    exec(f.read())

setup(
    name="vollseg",
    version=__version__,
    author="Varun Kapoor,Claudia Carabana Garcia,Mari Tolonen, Jakub Sedzinski",
    author_email="randomaccessiblekapoor@gmail.com",
    url="https://github.com/Kapoorlabs-CAPED/VollSeg/",
    description="Segmentation tool for biological cells of irregular size and shape in 3D and 2D, using \
    StarDist, U-NET, CARE, CellPose and SAM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "pandas",
        "stardist",
        "scipy",
        "tifffile",
        "matplotlib",
        "napari",
        "cellpose-vollseg",
        "torch",
        "torchvision",
        "torchaudio",
        "test_tube",
        "lightning",
        "segment-anything",
        "opencv-contrib-python-headless",
        "cellpose",
    ],
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
    ],
)
