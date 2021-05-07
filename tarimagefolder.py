
import tarfile

from tardataset import TarDataset

try:  # make torchvision optional
  from torchvision.transforms.functional import to_tensor
except:
  to_tensor = None


class TarImageFolder(TarDataset):
  """Dataset that supports Tar archives (uncompressed), with a folder per class.

  Similarly to torchvision.datasets.ImageFolder, assumes that the images inside
  the Tar archive are arranged in this way by default:

    root/dog/xxx.png
    root/dog/xxy.png
    root/dog/[...]/xxz.png

    root/cat/123.png
    root/cat/nsdf3.png
    root/cat/[...]/asd932_.png
  
  Args:
    archive (string): Path to the Tar file containing the dataset.
    root_in_archive (string): Root folder within the archive, directly below
      the folders with class names.
    extensions (tuple): Extensions (strings starting with a dot), only files
      with these extensions will be iterated. Default: png/jpg/jpeg.
    is_valid_file (callable): Optional function that takes file information as
      input (tarfile.TarInfo) and outputs True for files that need to be
      iterated; overrides extensions argument.
      Example: lambda m: m.isfile() and m.name.endswith('.png')
    transform (callable): Function applied to each image by __getitem__ (see
      torchvision.transforms). Default: ToTensor (convert PIL image to tensor).

  Attributes:
    samples (list): Image file names to iterate.
    targets (list): Numeric label corresponding to each image.
    class_to_idx (dict): Maps class names to numeric labels.
    idx_to_class (dict): Maps numeric labels to class names.
    members_by_name (dict): Members (files and folders) found in the Tar archive,
      with their names as keys and their tarfile.TarInfo structures as values.

  Author: Joao F. Henriques
  """
  def __init__(self, archive, transform=to_tensor, extensions=('.png', '.jpg', '.jpeg'),
    is_valid_file=None, root_in_archive=''):
    self.root_in_archive = root_in_archive

    # load the archive meta information, and filter the samples
    super().__init__(archive=archive, transform=transform, is_valid_file=is_valid_file)

    # assign labels to each sample
    self.assign_labels()


  def filter_samples(self, is_valid_file=None, extensions=('.png', '.jpg', '.jpeg')):
    """In addition to TarDataset's filtering by extension (or user-supplied),
    filter further to select only samples within the given root path."""
    super().filter_samples(is_valid_file, extensions)
    root = path_with_slash(self.root_in_archive)
    self.samples = [filename for filename in self.samples if filename.startswith(root)]


  def assign_labels(self):
    """Assign a label to each image, based on the first folder after the root
    that it's placed in"""
    root = path_with_slash(self.root_in_archive)
    self.class_to_idx = {}
    self.targets = []
    for filename in self.samples:
      # extract the class name from the file's path inside the Tar archive
      if self.root_in_archive:
        assert filename.startswith(root)  # sanity check (filter_samples should ensure this)
        filename = filename[len(root):]  # make path relative to root
      (class_name, _, _) = filename.partition('/')  # first folder level

      # assign increasing label indexes to each class name
      label = self.class_to_idx.setdefault(class_name, len(self.class_to_idx))
      self.targets.append(label)
    
    if len(self.class_to_idx) == 0:
      raise IOError("No classes (top-level folders) were found with the given criteria. The given\n"
        "extensions, is_valid_file or root_in_archive are too strict, or the archive is empty.")
    elif len(self.class_to_idx) == 1:
      raise IOError(f"Only one class (top-level folder) was found: {next(iter(self.class_to_idx))}.\n"
        f"To choose the correct path in the archive where the label folders are located, specify\n"
        f"root_in_archive in the TarImageFolder's constructor.")
    
    # the inverse mapping is often useful
    self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}


  def __getitem__(self, index):
    """Return a single sample.

    By default, this simply applies the given transforms or converts the image to
    a tensor if none are specified.

    Args:
      index (int): Index of item.
    
    Returns:
      tuple[Tensor, int]: The image and the corresponding label index.
    """
    image = self.get_image(self.samples[index], pil=True)
    image = image.convert('RGB')  # if it's grayscale, convert to RGB
    if self.transform:  # apply any custom transforms
      image = self.transform(image)
    
    label = self.targets[index]

    return (image, label)


def path_with_slash(path):
  """Helper function to return a path string ending with a slash (unless it's empty)."""
  if path and not path.endswith('/'):
    return path + '/'
  return path
