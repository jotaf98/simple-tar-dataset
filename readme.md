# Simple Tar Dataset

An unopinionated replacement for PyTorch's Dataset and ImageFolder classes, for datasets stored as uncompressed Tar archives.

**Just Tar it:** No particular structure is enforced in the Tar archive. This means that you can just archive your files with no modification, and handle any data/meta-data with your dataset code.

For image classification datasets, where images are usually stored in one folder per class (e.g. ImageNet), `TarImageFolder` is a drop-in replacement for `torchvision.dataset.ImageFolder`.


# Examples

These imports are common to most examples:

```
from tardataset import TarDataset
from tarimagefolder import TarImageFolder
```

### Just load a dataset of images, and print one pixel (RGB) of each

```
dataset = TarDataset('example-data/colors.tar')

for (idx, image) in enumerate(dataset):
  print(f"Image #{idx}, color: {image[:,0,0]}")
```

### Interpret folders as class labels (like torchvision's ImageFolder)

The folders follow the structure:
- red/a.png
- green/b.png
- blue/c.png

```
dataset = TarImageFolder('example-data/colors.tar')

for (idx, (image, label)) in enumerate(dataset):
  print(f"Image #{idx}, label: {label} ({dataset.idx_to_class[label]}), color: {image[:,0,0]}")
```

### Use DataLoaders (multiple processes) and return a batch tensor

```
from torch.utils.data import DataLoader

if __name__ == '__main__':  # needed for dataloaders
  dataset = TarImageFolder('example-data/colors.tar')
  loader = DataLoader(dataset, batch_size=3, num_workers=2, shuffle=True)

  for (image, label) in loader:
    print(f"Dimensions of image batch: {image.shape}")
    print(f"Labels in batch: {label}")
```

### Load stacks of video frames

```
import torch

class VideoDataset(TarDataset):
  """Example video dataset, each folder has the frames of a video as images"""
  def __init__(self, archive):
    # folders starting with 'vid' are considered samples
    super().__init__(archive=archive, is_valid_file=lambda m: m.isdir() and m.name.startswith('vid'))

  def __getitem__(self, index):
    """Load and return a stack of 3 frames from this folder"""
    folder = self.samples[index]
    images = [self.get_image(f"{folder}/{frame:02}.png") for frame in range(3)]
    return torch.stack(images)


dataset = VideoDataset('example-data/videos.tar')

for (idx, video) in enumerate(dataset):
  print(f"Video #{idx}, stack of frames with dimensions: {video.shape}")
```

### Load non-image files, such as pickled Python objects

```
import pickle

class PickleDataset(TarDataset):
  """Example non-image dataset"""
  def __init__(self, archive):
    super().__init__(archive=archive, extensions=('.pickle'))

  def __getitem__(self, index):
    """Return a pickled Python object"""
    filename = self.samples[index]
    return pickle.load(self.get_file(filename))


dataset = PickleDataset('example-data/objects.tar')

for (idx, obj) in enumerate(dataset):
  print(f"Sample #{idx}, object: {obj}")
```

### Load meta-data file with custom information about each sample

```
class RedDataset(TarDataset):
  """Example dataset, which loads from a CSV file a binary label of
  whether each image is red or not."""
  def __init__(self, archive):
    super().__init__(archive=archive)

    # read a CSV text file from the TAR file, containing the labels 'red'
    # or 'not-red' for each image (one per line)
    self.image_is_red = {}
    for line in self.get_text_file('custom-data.txt').splitlines():
      (name, redness) = line.split(',')
      self.image_is_red[name] = (redness == 'red')

  def __getitem__(self, index):
    """Return the image and the binary label"""
    filename = self.samples[index]
    image = self.get_image(filename)
    is_red = self.image_is_red[filename]
    return (image, is_red)


dataset = RedDataset('colors.tar')

for (idx, (image, label)) in enumerate(dataset):
  print(f"Image #{idx}, redness: {label}, color: {image[:,0,0]}")
```
