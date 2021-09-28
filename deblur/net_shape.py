import torch
from torch.utils.tensorboard import SummaryWriter
import os
import cv2
import torch.nn.functional as F 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np 
from models import * 
from utils.denoising_utils import *
from skimage.measure import compare_psnr
dtype = torch.cuda.FloatTensor
input_depth=1
pad = 'reflection'
mse=torch.nn.MSELoss().type(dtype)
writer=SummaryWriter('log')
for i in range(100):
    net = get_net(input_depth,'multi_skip', pad,
                    skip_n33d=128, 
                    skip_n33u=128, 
                    skip_n11=4, 
                    num_scales=5,
                    n_channels=1, 
                    upsample_mode='bilinear').type(dtype)
    num=sum(param.numel() for param in net.parameters())
    print('# generator parameters:', sum(param.numel() for param in net.parameters()))
    net.load_state_dict(torch.load('net_down_13to19.pth'))
    net = torch.nn.DataParallel(net)
    net=net.type(dtype)
    writer.add_scalar('value',num,num)