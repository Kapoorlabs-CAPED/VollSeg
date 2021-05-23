# VollSeg
3D segmentation tool for irregular shaped cells

[![Build Status](https://travis-ci.com/kapoorlab/vollseg.svg?branch=master)](https://travis-ci.com/github/kapoorlab/vollseg)
[![PyPI version](https://img.shields.io/pypi/v/vollseg.svg?maxAge=2591000)](https://pypi.org/project/vollseg/)
## Installation
This package can be installed by 


`pip install --user vollseg`

If you are building this from the source, clone the repository and install via

```bash
git clone https://github.com/kapoorlab/vollseg/

cd vollseg

pip install --user -e .

# or, to install in editable mode AND grab all of the developer tools
# (this is required if you want to contribute code back to NapaTrackMater)
pip install --user -r requirements.txt
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

## Usage

## Example

To try the provided notebooks we provide an example dataset of MDA231 human breast carcinoma cells infected with a pMSCV vector including the GFP sequence, embedded in a collagen matrix from Dr. R. Kamm. Dept. of Biological Engineering, Massachusetts Institute of Technology, Cambridge MA (USA)[tracking challenge](http://celltrackingchallenge.net/3d-datasets/), download the zip file of hyperstacks of the Raw, instance and semantic segmentation masks from [here](https://drive.google.com/drive/folders/1ze8KsrFI0-UTrsMnAPomiyf4sN8aCm__?usp=sharing). Pretrained model weights for denoising done via noise to void, segmentation done via U-Net and Staardist are also in the directory. For training the networks use this notebook in [Colab](https://github.com/kapoorlab/VollSeg/blob/main/examples/ColabTrainModel.ipynb). We provide  pre-trained model weights for stardist and U-Net. To train a denoising model using noise to void use this [notebook](https://github.com/kapoorlab/VollSeg/blob/main/examples/ColabN2VTrain.ipynb) 

## Requirements

- Python 3.7 and above.


## License

Under MIT license. See [LICENSE](LICENSE).

## Authors

- Varun Kapoor <randomaccessiblekapoor@gmail.com>
- Claudia Carabana Garcia
