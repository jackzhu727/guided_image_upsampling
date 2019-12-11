import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, utils
from utils.tools import *

class NYU_Dataset(Dataset):

    def __init__(self, root_dir, train=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            scale (float): dataset scale
            train (bool): train or test
            transform (callable, optional): Optional transform to be applied on a sample.
            
        """
        self.transform = transform
        
        if train:
            self.depths = (np.load('%s/train_depth_split.npy'%root_dir)//1).astype('uint8')
            self.images = np.load('%s/train_images_split.npy'%root_dir)
            self.targets = np.load('%s/train_targets_split.npy'%root_dir)
            self.masks = np.load('%s/train_masks_split.npy'%root_dir)
        else:
            self.depths = (np.load('%s/test_depth_split.npy'%root_dir)//1).astype('uint8')
            self.images = np.load('%s/test_images_split.npy'%root_dir)
            self.targets = np.load('%s/test_targets_split.npy'%root_dir)
            self.masks = np.load('%s/test_masks_split.npy'%root_dir)

    def __len__(self):
        return self.depths.shape[0]

    def __getitem__(self, idx):
        depth = self.depths[idx]
        h, w = depth.shape
        depth = np.array(Image.fromarray(depth).resize((w, h), Image.BICUBIC))
        image = np.array(Image.fromarray(self.images[idx]).resize((w, h), Image.BICUBIC))
        mask = self.masks[idx]
        target = np.array(Image.fromarray(self.targets[idx]) \
                          .resize((w, h),Image.BICUBIC))
        
        if self.transform:
            image = [self.transform(x).float() for x in crop_img(image)]
            depth = [self.transform(np.expand_dims(x,2)).float() for x in crop_img(depth)]
            target = [self.transform(np.expand_dims(x,2)).float() for x in crop_img(target)]
            mask = [torch.ByteTensor(x) for x in crop_img(mask)]
        else:
            image = crop_img(image)
            depth = crop_img(depth)
            target = crop_img(target)
            mask = crop_img(mask)
        sample = {'guidance': image, 'target': target, 'gt': depth, 'mask': mask}
        return sample
