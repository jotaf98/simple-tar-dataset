
import tarfile
from io import BytesIO
from PIL import Image, ImageFile

from torch.utils.data import Dataset, get_worker_info
from torch.multiprocessing import get_start_method

try:  # make torchvision optional
  from torchvision.transforms.functional import to_tensor
except:
  to_tensor = None

ImageFile.LOAD_TRUNCATED_IMAGES = True

class TarDataset(Dataset):
  """Dataset that supports Tar archives (uncompressed).

  Args:
    archive (string): Path to the Tar file containing the dataset.
    extensions (tuple): Extensions (strings starting with a dot), only files
      with these extensions will be iterated. Default: png/jpg/jpeg.
    is_valid_file (callable): Optional function that takes file information as
      input (tarfile.TarInfo) and outputs True for files that need to be
      iterated; overrides extensions argument.
      Example: lambda m: m.isfile() and m.name.endswith('.png')
    transform (callable): Function applied to each image by __getitem__ (see
      torchvision.transforms). Default: ToTensor (convert PIL image to tensor).

  Attributes:
    members_by_name (dict): Members (files and folders) found in the Tar archive,
      with their names as keys and their tarfile.TarInfo structures as values.
    samples (list): Items to iterate (can be ignored by overriding __getitem__
      and __len__).

  Author: Joao F. Henriques
  """
  def __init__(self, archive, transform=to_tensor, extensions=('.png', '.jpg', '.jpeg'),
    is_valid_file=None):
    # open tar file, and store headers of all files and folders by name
    self.archive = archive
    self.tar_obj = tarfile.open(archive)

    members = sorted(self.tar_obj.getmembers(), key=lambda m: m.name)
    self.members_by_name = {m.name: m for m in members}

    self.filter_samples(is_valid_file, extensions)
    
    self.transform = transform
    self.first_use = True

  def filter_samples(self, is_valid_file=None, extensions=('.png', '.jpg', '.jpeg')):
    """Filter the Tar archive's files/folders to obtain the list of samples.
    
    Args:
      extensions (tuple): Extensions (strings starting with a dot), only files
        with these extensions will be iterated. Default: png/jpg/jpeg.
      is_valid_file (callable): Optional function that takes file information as
        input (tarfile.TarInfo) and outputs True for files that need to be
        iterated; overrides extensions argument.
        Example: lambda m: m.isfile() and m.name.endswith('.png')
    """
    # by default, filter files by extension
    if is_valid_file is None:
      def is_valid_file(m):
        return (m.isfile() and m.name.lower().endswith(extensions))

    # filter the files to create the samples list
    self.samples = [m.name for m in self.members_by_name.values() if is_valid_file(m)]

  def __getitem__(self, index):
    """Return a single sample.
    
    Should be overriden by a subclass to support custom data other than images (e.g.
    class labels). The methods get_image/get_file can be used to read from the Tar
    archive, and a dict of files/folders is held in the property members_by_name.

    By default, this simply applies the given transforms or converts the image to
    a tensor if none are specified.

    Args:
      index (int): Index of item.
    
    Returns:
      Tensor: The image.
    """
    image = self.get_image(self.samples[index], pil=True)
    image = image.convert('RGB')  # if it's grayscale, convert to RGB
    if self.transform:  # apply any custom transforms
      image = self.transform(image)
    return image

  def __len__(self):
    """Return the length of the dataset (length of self.samples)

    Returns:
      int: Number of samples.
    """
    return len(self.samples)

  def get_image(self, name, pil=False):
    """Read an image from the Tar archive, returned as a PIL image or PyTorch tensor.

    Args:
      name (str): File name to retrieve.
      pil (bool): If true, a PIL image is returned (default is a PyTorch tensor).

    Returns:
      Image or Tensor: The image, possibly in PIL format.
    """
    image = Image.open(BytesIO(self.get_file(name).read()))
    if pil:
      return image
    return to_tensor(image)

  def get_text_file(self, name, encoding='utf-8'):
    """Read a text file from the Tar archive, returned as a string.

    Args:
      name (str): File name to retrieve.
      encoding (str): Encoding of file, default is utf-8.

    Returns:
      str: Content of text file.
    """
    return self.get_file(name).read().decode(encoding)

  def get_file(self, name):
    """Read an arbitrary file from the Tar archive.

    Args:
      name (str): File name to retrieve.

    Returns:
      io.BufferedReader: Object used to read the file's content.
    """
    if self.first_use:
      self.first_use = False
      if get_worker_info() is not None and get_start_method() != 'spawn':
        raise OSError("TarDataset is being used with multiple workers of a DataLoader.\n"
          "To ensure each worker has its own file handle, the 'spawn' multiprocessing\n"
          "method must be used. Since this is not the default on Unix, you should call:\n"
          "torch.multiprocessing.set_start_method('spawn')")
    return self.tar_obj.extractfile(self.members_by_name[name])

  def __del__(self):
    """Close the Tar file handle on exit."""
    if self.tar_obj:
      self.tar_obj.close()

  def __getstate__(self):
    """Serialize without the TarFile reference, for multi-process compatibility."""
    state = dict(self.__dict__)
    state['tar_obj'] = None
    return state

  def __setstate__(self, state):
    """Restore the TarFile reference and reopen the Tar archive."""
    self.__dict__.update(state)
    self.tar_obj = tarfile.open(self.archive)
