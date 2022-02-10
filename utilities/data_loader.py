from torchvision import transforms as T
from torch.utils.data import Dataset
from skimage.color import gray2rgb
from skimage.io import imread
from PIL import Image
import torchvision
import numpy as np
import torch
import glob



class CustomDataset(Dataset):
     """Given an image directory path return target patchs and 8 neighbour patches.
        Given an Image we devide the image into 4x4 grid (16 Patches). (Grid size can be changed by changing the grid_size parameter.)
        For Patch we gather the 8 neighbour patches. Padding is done for the border patches so that every patch has 8 neighbours.
     """
    
    def __init__(self, path:str, transforms=None, grid_size=4):        
        self.image_paths = glob.glob(path)

        if not self.image_paths:
          raise Exception('Directory path is not correct!!')

        self.transforms = transforms
        self.grid_size = grid_size

    def __getitem__(self, index):
      image_path = self.image_paths[index]
      image = imread(image_path)

      if len(image.shape) == 2:
          image = gray2rgb(image)


      if self.transforms is not None:
        image = self.transforms(image)


      shape = np.array(image.shape)
      patch_rw, patch_cl = shape[1]//self.grid_size, shape[2]//self.grid_size
      
      scale = T.Compose([T.Resize((patch_rw*self.grid_size, patch_cl*self.grid_size))])
      padding = torch.nn.ZeroPad2d((patch_cl, patch_cl, patch_rw, patch_rw))
      img = padding(scale(image))
     
      patches = img.data.unfold(0, 3, 3).unfold(1, patch_rw, patch_rw).unfold(2, patch_cl, patch_cl)
      
      neighbours = torch.zeros(self.grid_size*self.grid_size, 8, shape[0], patch_rw, patch_cl)
      target = torch.zeros(self.grid_size*self.grid_size, shape[0], patch_rw, patch_cl)

      k = 0

      for i in range(1, self.grid_size+1):
        for j in range(1, self.grid_size+1):

          neighbours[k, 0, :, :, :] = patches[0, i-1, j-1, :, :, :]
          neighbours[k, 1, :, :, :] = patches[0, i-1, j, :, :, :]
          neighbours[k, 2, :, :, :] = patches[0, i-1, j+1, :, :, :]
          neighbours[k, 3, :, :, :] = patches[0, i, j-1, :, :, :]
          target[k, :, :, :] = patches[0, i, j, :, :, :]
          neighbours[k, 4, :, :, :] = patches[0, i, j+1, :, :, :]
          neighbours[k, 5, :, :, :] = patches[0, i+1, j-1, :, :, :]
          neighbours[k, 6, :, :, :] = patches[0, i+1, j, :, :, :]
          neighbours[k, 7, :, :, :] = patches[0, i+1, j+1, :, :, :]

          k += 1

      return neighbours, target
 
    def __len__(self):
      return len(self.image_paths)



class PatchCIFAR100(torchvision.datasets.CIFAR100):
  """Override torchvision CIFAR100 to return a target patch and its neighbours.
     Given an Image we devide the image into 4x4 grid (16 Patches).
     For Each Grid Patch we gather the 8 neighbour patches. Padding is done for the border patches.
  """
  def __init__(self, transforms=None, grid_size=4, **kwds):
    super().__init__(**kwds)
    self.transforms = transforms
    self.grid_size = grid_size
      
  def __getitem__(self, index):
    image, class_targets = self.data[index], self.targets[index]

    if len(image.shape) == 2:
        image = gray2rgb(image)

    if self.transforms is not None:
      image = self.transforms(image)


    shape = np.array(image.shape)
    patch_rw, patch_cl = shape[1]//self.grid_size, shape[2]//self.grid_size
    scale = T.Compose([T.Resize((patch_rw*self.grid_size, patch_cl*self.grid_size))])
    padding = torch.nn.ZeroPad2d((patch_cl, patch_cl, patch_rw, patch_rw))
    img = padding(scale(image))

    patches = img.data.unfold(0, 3, 3).unfold(1, patch_rw, patch_rw).unfold(2, patch_cl, patch_cl)
    
    neighbours = torch.zeros(self.grid_size*self.grid_size, 8, shape[0], patch_rw, patch_cl)
    target = torch.zeros(self.grid_size*self.grid_size, shape[0], patch_rw, patch_cl)

    k = 0

    for i in range(1, self.grid_size+1):
      for j in range(1, self.grid_size+1):

        neighbours[k, 0, :, :, :] = patches[0, i-1, j-1, :, :, :]
        neighbours[k, 1, :, :, :] = patches[0, i-1, j, :, :, :]
        neighbours[k, 2, :, :, :] = patches[0, i-1, j+1, :, :, :]
        neighbours[k, 3, :, :, :] = patches[0, i, j-1, :, :, :]
        target[k, :, :, :] = patches[0, i, j, :, :, :]
        neighbours[k, 4, :, :, :] = patches[0, i, j+1, :, :, :]
        neighbours[k, 5, :, :, :] = patches[0, i+1, j-1, :, :, :]
        neighbours[k, 6, :, :, :] = patches[0, i+1, j, :, :, :]
        neighbours[k, 7, :, :, :] = patches[0, i+1, j+1, :, :, :]

        k += 1           

    return neighbours, target
