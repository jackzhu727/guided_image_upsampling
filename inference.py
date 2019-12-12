import torch
import numpy as np
import cv2
import argparse
from utils.tools import *
from models import *
from PIL import Image
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--rgb',  default='images/0_rgb.png', help='name of rgb image')
parser.add_argument('--depth',  default='images/0_lr.png', help='name of low resolution depth image')
parser.add_argument('--k', type=int, default=3, help='size of kernel')
parser.add_argument('--d', type=int, default=15, help='size of grid area')
parser.add_argument('--scale', type=int, default=8, help='scale factor')
parser.add_argument('--parameter',  default='parameter/FDKN_8x', help='name of parameter file')
parser.add_argument('--model',  default='FDKN', help='choose model FDKN or DKN')
parser.add_argument('--output',  default='images/0_dkn.png', help='name of output image')
opt = parser.parse_args()
print(opt)


if opt.model == 'FDKN':
    net = FDKN(kernel_size=opt.k, filter_size=opt.d, residual=True)
elif opt.model == 'DKN':
    net = DKN(kernel_size=opt.k, filter_size=opt.d, residual=True)

net.load_state_dict(torch.load(opt.parameter, map_location=torch.device('cpu')))
net.eval()
print('parameter \"%s\" has loaded'%opt.parameter)


rgb = cv2.imread(opt.rgb)
h, w = rgb.shape[:2]
lr = cv2.imread(opt.depth, cv2.IMREAD_GRAYSCALE)
lr = np.array(Image.fromarray(lr).resize((w, h), Image.BICUBIC))
data_transform = transforms.Compose([
    transforms.ToTensor()
])

images = [data_transform(x).float().unsqueeze(0) for x in crop_img(rgb)]
targets = [data_transform(np.expand_dims(x,2)).float().unsqueeze(0) for x in crop_img(lr)]

outputs = []
for idx in range(len(images)):
    with torch.no_grad():
        outputs.append(net((images[idx], targets[idx])).cpu().numpy()[0,0])
cv2.imwrite(opt.output, merge_img(outputs)*255)
print('Job Complete')