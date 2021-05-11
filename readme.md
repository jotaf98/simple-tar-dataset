# Simple Tar Dataset

An unopinionated replacement for PyTorch's Dataset and ImageFolder classes, for datasets stored as uncompressed Tar archives.

**Just Tar it:** No particular structure is enforced in the Tar archive. This means that you can just archive your files with no modification, and handle any data/meta-data with your dataset code.

**Why?** Storing a dataset as millions of small files makes access inefficient, and can create other difficulties in large-scale scenarios (e.g. running out of inodes, inneficient operations in distributed filesystems which are optimised for fewer large files). A Tar file is a simple and uncompressed archive format for which numerous utilities exist, and it allows fast random access into a single archive file.


## Example

The default `TarDataset` simply loads all PNG, JPG and JPEG images from a Tar file, and allows you to iterate them.

Images are returned as `Tensor`. Here some RGB values are printed.

```python
from tardataset import TarDataset

dataset = TarDataset('example-data/colors.tar')

for (idx, image) in enumerate(dataset):
  print(f"Image #{idx}, color: {image[:,0,0]}")
```

## Usage

For image classification datasets, where images are usually stored in one folder per class (e.g. ImageNet), `TarImageFolder` is a drop-in replacement for `torchvision.dataset.ImageFolder`.

For more complex scenarios -- say, you store some data in one or more JSON files, or you have folders with video frames in specific formats -- you can subclass `TarDataset`, and read the data in any format you like.


## Jupyter notebook tutorial

There is a more comprehensive set of examples as a Jupyter notebook in [`example.ipynb`](example.ipynb).


## Full "ImageNet in a Tar file" example

A large-scale data loading example is given in `imagenet-example.py`. Only the section of code responsible for data loading was modified from the [official PyTorch ImageNet example](https://github.com/pytorch/examples/tree/master/imagenet).

First, ensure that the data is in the expected format for the original example to work, in a folder named `ILSVRC12`. Then, create a Tar archive from it (`tar cf ILSVRC12.tar ILSVRC12` on Linux or a utility like 7-Zip on Windows). Finally, run our modified `imagenet-example.py`, passing it the path to the Tar archive instead.


## Author

[João Henriques](http://www.robots.ox.ac.uk/~joao/), [Visual Geometry Group (VGG)](http://www.robots.ox.ac.uk/~vgg/), University of Oxford
