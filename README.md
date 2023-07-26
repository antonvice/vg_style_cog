# Van Gogh Style Transfer
This project fine-tunes a Stable Diffusion model on paintings by Van Gogh and provides a style transfer implementation to turn images into Van Gogh style paintings.

# Usage
To train the model:

```bash

python data_prep.py && python training.py
This will download the Van Gogh paintings dataset and fine-tune the model.
```
To run training transfer:

```bash

python style.py --input <PATH_TO_IMAGE>
```
Pass the input image

# Model
The core model is a Stable Diffusion pipeline pre-trained on wikimedia images. This is then fine-tuned on the Van Gogh paintings to adapt the style.


# Installation
The required libraries are:

* diffusers
* datasets
* torch
* torchvision
* pandas

### Install with:
```bash
pip install -r requirements.txt
```
A GPU is recommended for efficient training and inference.

# References
Kaggle Van Gogh Paintings Dataset
Stable Diffusion
Style Transfer with Stable Diffusion