import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, utils
import os
import cv2
import matplotlib.pyplot as plt
# Preprocessing
directory = ''
images = []
depths = []
targets = []
masks = []
for i, dir_name in sorted(enumerate(os.listdir(directory))):
    img_path = directory + dir_name + '/guidance.png'
    depth_path = directory + dir_name + '/ground_truth.png'
    target_path = directory + dir_name + '/target.png'
    guidence = cv2.imread(img_path)
    target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    mask = depth != 0
    if guidence is None or target is None or depth is None:
        raise ValueError(dir_name)
    minmax = [target.min(), target.max()]
    if depth.max() == depth.min():
        raise ValueError(dir_name)
    depth = ((depth - depth.min())/(depth.max()-depth.min()) * (minmax[1]-minmax[0]) + minmax[0])
    if guidence is None or target is None or depth is None:
        raise ValueError(dir_name)
    images.append(guidence)
    depths.append(depth)
    targets.append(target)
    masks.append(mask)

images = np.array(images)
depths = np.array(depths)
targets = np.array(targets)
masks = np.array(masks)


np.save('./train_depth_split.npy', depths[:80])
np.save('./train_images_split.npy', images[:80])
np.save('./train_targets_split.npy', targets[:80])
np.save('./train_masks_split.npy', masks[:80])

np.save('./val_depth_split.npy', depths[80:100])
np.save('./val_images_split.npy', images[80:100])
np.save('./val_targets_split.npy', targets[80:100])
np.save('./val_masks_split.npy', masks[80:100])

np.save('./test_depth_split.npy', depths[100:130])
np.save('./test_images_split.npy', images[100:130])
np.save('./test_targets_split.npy', targets[100:130])
np.save('./test_masks_split.npy', masks[100:130])