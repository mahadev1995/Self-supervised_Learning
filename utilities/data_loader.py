from torchvision import transforms as T
from torch.utils.data import Dataset
from skimage.color import gray2rgb
from skimage.io import imread
from PIL import Image
import torchvision
import numpy as np
import torch
import glob
import copy



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


class CustomDatasetInnerPatches(Dataset):
  """Given an image directory path return  inner target patchs and 8 neighbour patches.
    Given an Image we devide the image into 5x5 grid (25 Patches). (Grid size can be changed by changing the grid_size parameter.)
    For inner 16 Patches we gather the 8 neighbour patches for each patch. 
  """
    
  def __init__(self, path:str, transforms=None, grid_size=4):        
      self.image_paths = glob.glob(path)

      if not self.image_paths:
        raise Exception('Directory path is not correct!!')

      self.transforms = transforms
      self.grid_size = grid_size 
      self.grid_size_padded = grid_size + 2

  def __getitem__(self, index):
    image_path = self.image_paths[index]
    image = imread(image_path)

    if len(image.shape) == 2:
        image = gray2rgb(image)


    if self.transforms is not None:
      image = self.transforms(image)


    shape = np.array(image.shape)
    patch_rw, patch_cl = shape[1]//self.grid_size_padded, shape[2]//self.grid_size_padded
    
    scale = T.Compose([T.Resize((patch_rw*self.grid_size_padded, patch_cl*self.grid_size_padded))])
    # padding = torch.nn.ZeroPad2d((patch_cl, patch_cl, patch_rw, patch_rw))
    img = scale(image)
    
    patches = img.data.unfold(0, 3, 3).unfold(1, patch_rw, patch_rw).unfold(2, patch_cl, patch_cl)
    # print(patches.shape)
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


class InnerPatchCIFAR100(torchvision.datasets.CIFAR100):
  """Override torchvision CIFAR100 to return a target patch and its neighbours.
     Given an Image we devide the image into 4x4 grid (16 Patches).
     For Each Grid Patch we gather the 8 neighbour patches. Padding is done for the border patches.
  """
  def __init__(self, transforms=None, grid_size=4, **kwds):
    super().__init__(**kwds)
    self.transforms = transforms
    self.grid_size = grid_size
    self.grid_size_padded = grid_size + 2
      
  def __getitem__(self, index):
    image, class_targets = self.data[index], self.targets[index]

    if len(image.shape) == 2:
        image = gray2rgb(image)

    if self.transforms is not None:
      image = self.transforms(image)

    shape = np.array(image.shape)
    patch_rw, patch_cl = shape[1]//self.grid_size_padded, shape[2]//self.grid_size_padded
    scale = T.Compose([T.Resize((patch_rw*self.grid_size_padded, patch_cl*self.grid_size_padded))])
    # padding = torch.nn.ZeroPad2d((patch_cl, patch_cl, patch_rw, patch_rw))
    img = scale(image)

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

  
class InnerPatchCIFAR10(torchvision.datasets.CIFAR10):
  """Override torchvision CIFAR10 to return a target patch and its neighbours.
     Given an Image we devide the image into ((n+2)x(n+2)) grid.
     For Each (n x n) inner Grid Patches we gather the 8 neighbour patches. 
  """
  def __init__(self, transforms=None, grid_size=4, **kwds):
    super().__init__(**kwds)
    self.transforms = transforms
    self.grid_size = grid_size
    self.grid_size_padded = grid_size + 2
      
  def __getitem__(self, index):
    image, class_targets = self.data[index], self.targets[index]

    if len(image.shape) == 2:
        image = gray2rgb(image)

    if self.transforms is not None:
      image = self.transforms(image)

    shape = np.array(image.shape)
    patch_rw, patch_cl = shape[1]//self.grid_size_padded, shape[2]//self.grid_size_padded
    scale = T.Compose([T.Resize((patch_rw*self.grid_size_padded, patch_cl*self.grid_size_padded))])
    # padding = torch.nn.ZeroPad2d((patch_cl, patch_cl, patch_rw, patch_rw))
    img = scale(image)

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


class PatchCIFAR10(torchvision.datasets.CIFAR10):
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


class MultiPatchCIFAR100(torchvision.datasets.CIFAR100):
  """
     Override torchvision CIFAR100 to return a target patch and its neighbours.
     Given an Image we augment image and create K augmented images. 
     Devide the image into 4x4 grid (16 Patches).
     For Each Grid Patch we gather the 8 neighbour patches. 
     Padding is done for the border patches.
     Finally, return neighbour patches and the target patches.
  """
  def __init__(self, transforms=None, grid_size=4, K=3, **kwds):
    super().__init__(**kwds)
    self.transforms = transforms
    self.grid_size = grid_size
    self.K = K
      
  def __getitem__(self, index):
    image, class_targets = self.data[index], self.targets[index]

    if len(image.shape) == 2:
        image = gray2rgb(image)

    shape = np.array(image.shape)
    # step 2 Basic transformation of the image and
    # resize the image according to the grid_size

    patch_rw, patch_cl = shape[0]//self.grid_size, shape[1]//self.grid_size
    

    scale = T.Compose([
                                T.ToPILImage(),
                                T.Resize(
                                (patch_rw*self.grid_size, 
                                patch_cl*self.grid_size)
                                ),
                                ])
    tensor_transform = T.Compose([T.ToTensor()])
    img_pil  = scale(image)
    img_pil_t = tensor_transform(img_pil)
  
    # step 3 Apply augmentation transformations
    img_tensor = torch.zeros(
                              (self.K+1, *img_pil_t.shape)
                              )

    for i in range(self.K):
      img_tensor[i] = self.transforms(copy.deepcopy(img_pil))

    img_tensor[self.K] = img_pil_t
    # step 4 Pad the image tensor conatining all the augmented images
    padding = torch.nn.ConstantPad3d((patch_cl, patch_cl, 
                                      patch_rw, patch_rw, 
                                      0, 0), 
                                      0)
    padded_img_tensor = padding(img_tensor)
    # step 5 Convert them into patches
    img_patches = padded_img_tensor.data.unfold(1, 3, 3).unfold(2, patch_rw, patch_rw).unfold(3, patch_cl, patch_cl)      
    img_patches = torch.squeeze(img_patches, 1)   

    # step 6 get neighbours of each target patch and return the neigbours & target patch
    neighbours = torch.zeros(self.K+1, self.grid_size*self.grid_size, 8, 3, patch_rw, patch_cl)
    target = torch.zeros(self.K+1, self.grid_size*self.grid_size, 3, patch_rw, patch_cl)

    k = 0

    for i in range(1, self.grid_size+1):
        for j in range(1, self.grid_size+1):
            # print(neighbours[:, k, 0, :, :, :].shape)
            # print(img_patches[:, i-1, j-1, :, :, :].shape)
            neighbours[:, k, 0, :, :, :] = img_patches[:, i-1, j-1, :, :, :]
            neighbours[:, k, 1, :, :, :] = img_patches[:, i-1, j, :, :, :]
            neighbours[:, k, 2, :, :, :] = img_patches[:, i-1, j+1, :, :, :]
            neighbours[:, k, 3, :, :, :] = img_patches[:, i, j-1, :, :, :]
            target[:, k, :, :, :] = img_patches[:, i, j, :, :, :]
            neighbours[:, k, 4, :, :, :] = img_patches[:, i, j+1, :, :, :]
            neighbours[:, k, 5, :, :, :] = img_patches[:, i+1, j-1, :, :, :]
            neighbours[:, k, 6, :, :, :] = img_patches[:, i+1, j, :, :, :]
            neighbours[:, k, 7, :, :, :] = img_patches[:, i+1, j+1, :, :, :]

            # target[0, k, :, :, :] = img_patches[0, i, j, :, :, :]

            k += 1
            
    neighbours_shape = neighbours.shape
    target_shape = target.shape
    neighbours = neighbours.resize_((neighbours_shape[0]*neighbours_shape[1], 
                                     neighbours_shape[2], neighbours_shape[3], 
                                     neighbours_shape[4], neighbours_shape[5]))

    target = target.resize_((target_shape[0]*target_shape[1], 
                             target_shape[2], target_shape[3], 
                             target_shape[4]))

    return neighbours, target