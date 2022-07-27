# VollSeg

[![Build Status](https://travis-ci.com/kapoorlab/vollseg.svg?branch=master)](https://travis-ci.com/github/kapoorlab/vollseg)
[![PyPI version](https://img.shields.io/pypi/v/vollseg.svg?maxAge=2591000)](https://pypi.org/project/vollseg/)
[![License](https://img.shields.io/pypi/l/napari-metroid.svg?color=green)](https://github.com/kapoorlab/napari-vollseg/raw/main/LICENSE)
[![Twitter Badge](https://badgen.net/badge/icon/twitter?icon=twitter&label)](https://twitter.com/entracod)

3D segmentation tool for irregular shaped cells
![Segmentation](https://github.com/kapoorlab/VollSeg/blob/main/images/Seg_compare-big.png)


## Installation
This package can be installed by 


`pip install --user vollseg`



If you are building this from the source, clone the repository and install via

```bash
git clone https://github.com/kapoorlab/vollseg/

cd vollseg

pip install --user -e .

```


### Pipenv install

Pipenv allows you to install dependencies in a virtual environment.

```bash
# install pipenv if you don't already have it installed
pip install --user pipenv

# clone the repository and sync the dependencies
git clone https://github.com/kapoorlab/vollseg/
cd vollseg
pipenv sync

# make the current package available
pipenv run python setup.py develop

# you can run the example notebooks by starting the jupyter notebook inside the virtual env
pipenv run jupyter notebook
```

Access the `example` folder and run the cells.

## Algorithm
![Algorithm](https://github.com/kapoorlab/VollSeg/blob/main/images/Seg_pipe-git.png)

Schematic representation showing the segmentation approach used in VollSeg. First, we input the raw fluorescent image in 3D (A) and preprocess it to remove noise. Next, we obtain the star convex approximation to the cells using Stardist (B) and the U-Net prediction labelled via connected components (C). We then obtain seeds from the centroids of labelled image in B, for each labelled region of C in order to create bounding boxes and centroids. If there is no seed from B in the bounding box region from U-Net, we add the new centroid (in yellow) to the seed pool (D). Finally, we do a marker controlled watershed in 3D using skimage implementation on the probability map shown in (E) to obtain final cell segmentation result shown in (F). All images are displayed in Napari viewer with 3D display view.
     
## Example

To try the provided notebooks we provide an example dataset of Arabidopsis, [Binary Images](https://doi.org/10.5281/zenodo.5217367), [Raw Images](https://doi.org/10.5281/zenodo.5217394) and [Labelled images](https://doi.org/10.5281/zenodo.5217341) and trained models: [stardist](https://doi.org/10.5281/zenodo.5227304), [Denoising](https://doi.org/10.5281/zenodo.5227316), [U-Net](https://doi.org/10.5281/zenodo.5227301). For training the networks use this notebook in [Colab](https://github.com/kapoorlab/VollSeg/blob/main/examples/Train/ColabTrainModel.ipynb). To train a denoising model using noise to void use this [notebook](https://github.com/kapoorlab/VollSeg/blob/main/examples/Train/ColabN2VTrain.ipynb) 



## Docker

A Docker image can be used to run the code in a container. Once inside the project's directory, build the image with:

~~~bash
docker build -t voll .
~~~

Now to run the `track` command:

~~~bash
# show help
docker run --rm -it voll
~~~


## Requirements

- Python 3.7 and above.


## License

Under MIT license. See [LICENSE](LICENSE).

## Authors

- Varun Kapoor <randomaccessiblekapoor@gmail.com>
- Claudia Carabaña
- Mari Tolonen
