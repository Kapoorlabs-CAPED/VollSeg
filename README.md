# VollSeg



# Developed by KapoorLabs


<img src="images/mtrack.png" alt="Logo1" width="150"/>
<img src="images/kapoorlablogo.png" alt="Logo2" width="150"/>

This product is a testament to our expertise at KapoorLabs, where we specialize in creating cutting-edge solutions. We offer bespoke pipeline development services, transforming your developmental biology questions into publishable figures with our advanced computer vision and AI tools. Leverage our expertise and resources to achieve end-to-end solutions that make your research stand out.

**Note:** The tools and pipelines showcased here represent only a fraction of what we can achieve. For tailored and comprehensive solutions beyond what was done in the referenced publication, engage with us directly. Our team is ready to provide the expertise and custom development you need to take your research to the next level. Visit us at [KapoorLabs](https://www.kapoorlabs.org/).

## Segmentation Algorithm
VollSeg is more than just a single segmentation algorithm; it is a meticulously designed modular segmentation tool tailored to diverse model organisms and imaging methods. While a U-Net might suffice for certain image samples, others might benefit from utilizing StarDist, and some could require a blend of both, potentially coupled with denoising or region of interest models. The pivotal decision left to make is how to select the most appropriate VollSeg configuration for your dataset, a question we comprehensively address in our [documentation website](https://kapoorlabs-caped.github.io/vollseg-napari/).



[![PyPI version](https://img.shields.io/pypi/v/vollseg.svg?maxAge=2591000)](https://pypi.org/project/vollseg/)
[![License](https://img.shields.io/pypi/l/napari-metroid.svg?color=green)](https://github.com/kapoorlab/napari-vollseg/raw/main/LICENSE)
[![Twitter Badge](https://badgen.net/badge/icon/twitter?icon=twitter&label)](https://twitter.com/entracod)
![Segmentation](https://github.com/kapoorlab/VollSeg/blob/main/images/Seg_compare-big.png)



## Installation
This package can be installed by 

`pip install  vollseg`

If you are building this from the source, clone the repository and install via

```bash
git clone https://github.com/kapoorlab/vollseg/

cd vollseg

pip install -e .


```

![Algorithm](https://github.com/kapoorlab/VollSeg/blob/main/images/Seg_pipe-git.png)
- **Algorithm**
  - Schematic representation showing the segmentation approach used in VollSeg.
  - First, we input the raw fluorescent image in 3D (A) and preprocess it to remove noise.
  - Next, we obtain the star convex approximation to the cells using Stardist (B) and the U-Net prediction labeled via connected components (C).
  - We then obtain seeds from the centroids of labeled image in B, for each labeled region of C in order to create bounding boxes and centroids.
  - If there is no seed from B in the bounding box region from U-Net, we add the new centroid (in yellow) to the seed pool (D).
  - Finally, we do a marker controlled watershed in 3D using skimage implementation on the probability map shown in (E) to obtain the final cell segmentation result shown in (F).
  - All images are displayed in Napari viewer with 3D display view.
## Requirements

- Python 3.7 and above.


## License

Under MIT license. See [LICENSE](LICENSE).

## Authors

- Varun Kapoor <randomaccessiblekapoor@gmail.com>
- Claudia Carabaña
- Mari Tolonen
