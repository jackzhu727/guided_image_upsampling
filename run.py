import torch
import numpy as np
import cv2
import argparse
from models import *
from nyu_dataloader import *
from utils.tools import *
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, utils
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import logging
from datetime import datetime
import os
torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=3, help='size of kernel')
parser.add_argument('--d', type=int, default=15, help='size of grid area')
parser.add_argument('--data ',  default='preprocessed_data/SAMSUNG', help='directory of data file')
parser.add_argument('--parameter',  default='parameter/FDKN_8x', help='name of parameter file')
parser.add_argument('--model',  default='FDKN', help='choose model FDKN or DKN')
parser.add_argument('--lr',  default='0.0001', type=float, help='learning rate')
parser.add_argument('--result',  default='./result', help='learning rate')
parser.add_argument('--epoch',  default=20, type=int, help='max epoch')


output = './test_output/'

opt = parser.parse_args()

parameter, data_path, model, result, epoch, lr, k, d = opt.parameter, opt.data, opt.model, opt.result, \
                                                  opt.epoch, opt.lr, opt.k, opt.d

data_transform = transforms.Compose([
    transforms.ToTensor()
])

nyu_dataset = NYU_Dataset(root_dir=data_path, transform=data_transform)
dataloader = torch.utils.data.DataLoader(nyu_dataset, batch_size=1, shuffle=True)

s = datetime.now().strftime('%Y%m%d%H%M%S')
result_root = '%s/%s-lr_%s-k_%s-d_%s'%(result, s, lr, k, d)
if not os.path.exists(result_root): 
    os.mkdir(result_root)
logging.basicConfig(filename='%s/train.log'%result_root,format='%(asctime)s %(message)s', level=logging.INFO)

if model == 'FDKN':
    net = FDKN(kernel_size=k, filter_size=d, residual=True).cuda()
elif model == 'DKN':
    net = DKN(kernel_size=k, filter_size=d, residual=True).cuda()
    
criterion = nn.L1Loss(reduction = 'none')
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
net.train()


def validate(net, root_dir='./dataset'):
    net.eval()
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = NYU_Dataset(root_dir=data_path, transform=data_transform, train=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    net.eval()
    rmse = np.zeros(30)
    t = tqdm(iter(test_dataloader), leave=True, total=len(test_dataloader))
    for idx, data in enumerate(t):
        guidance, target, gt, masks = data['guidance'], data['target'], data['gt'], data['mask']
        assert len(guidance) == len(target)
        assert len(guidance) == len(gt)
        num_of_crops = len(guidance)
        rmse[idx] = 0 
        for i in range(num_of_crops):
            out = net((guidance[i].cuda(), target[i].cuda()))
            ground_truth = gt[i][0,0].numpy()
            rmse[idx] += calc_rmse(ground_truth, out[0,0].detach().cpu().numpy(),
                                   [ground_truth.min(), ground_truth.max()], masks[i][0].cpu().numpy())/16
        t.set_description('[validate] rmse: %f' %rmse[:idx+1].mean())
        t.refresh()
    return rmse


max_epoch = epoch
for epoch in range(max_epoch):
    if epoch % 10 == 0:
        rmse = validate(net)
    net.train()
    running_loss = 0.0
    
    t = tqdm(iter(dataloader), leave=True, total=len(dataloader))
    for idx, data in enumerate(t):
        optimizer.zero_grad()
        scheduler.step()
        guidance, target, gt, masks = data['guidance'], data['target'], data['gt'], data['mask']
        assert len(guidance) == len(target)
        assert len(guidance) == len(gt)
        num_of_crops = len(guidance)
        
        for i in range(num_of_crops):
            out = net((guidance[i].cuda(), target[i].cuda()))
            mask = masks[i].float().cuda()
            non_neg = mask.sum()
            loss = criterion(out, gt[i].cuda()) * mask/non_neg
            loss = loss.sum()
            loss.backward()
            optimizer.step()
            running_loss += loss.data.item()
        
        if idx % 10 == 0:
            t.set_description('[train epoch:%d] loss: %.8f' % (epoch+1, running_loss))
            t.refresh()
    logging.info('epoch:%d mean_rmse:%f'%(epoch+1, 0))
    torch.save(net.state_dict(), "%s/parameter%d"%(result_root, epoch+1))

