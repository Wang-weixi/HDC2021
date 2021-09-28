import torch
from torch.utils.tensorboard import SummaryWriter
import os
import cv2
from SSIM import SSIM
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import numpy as np  
from models import * 
from models import DLSS
from utils.denoising_utils import *
from skimage.measure import compare_psnr
dtype = torch.cuda.FloatTensor
total_number=90
epoch_each=1
epoch=80
step0_file_name_V = [['./step0/Verdana/CAM01/focusStep_0_verdanaRef_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step0/Verdana/CAM02/focusStep_0_verdanaRef_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step0_file_name_T = [['./step0/Times/CAM01/focusStep_0_timesR_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step0/Times/CAM02/focusStep_0_timesR_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step1_file_name_V = [['./step1/Verdana/CAM01/focusStep_1_verdanaRef_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step1/Verdana/CAM02/focusStep_1_verdanaRef_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step1_file_name_T = [['./step1/Times/CAM01/focusStep_1_timesR_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step1/Times/CAM02/focusStep_1_timesR_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step2_file_name_V = [['./step2/Verdana/CAM01/focusStep_2_verdanaRef_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step2/Verdana/CAM02/focusStep_2_verdanaRef_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step2_file_name_T = [['./step2/Times/CAM01/focusStep_2_timesR_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step2/Times/CAM02/focusStep_2_timesR_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step3_file_name_V = [['./step3/Verdana/CAM01/focusStep_3_verdanaRef_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step3/Verdana/CAM02/focusStep_3_verdanaRef_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step3_file_name_T = [['./step3/Times/CAM01/focusStep_3_timesR_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step3/Times/CAM02/focusStep_3_timesR_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step4_file_name_V = [['./step4/Verdana/CAM01/focusStep_4_verdanaRef_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step4/Verdana/CAM02/focusStep_4_verdanaRef_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step4_file_name_T = [['./step4/Times/CAM01/focusStep_4_timesR_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step4/Times/CAM02/focusStep_4_timesR_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step5_file_name_V = [['./step5/Verdana/CAM01/focusStep_5_verdanaRef_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step5/Verdana/CAM02/focusStep_5_verdanaRef_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step5_file_name_T = [['./step5/Times/CAM01/focusStep_5_timesR_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step5/Times/CAM02/focusStep_5_timesR_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step6_file_name_V = [['./step6/Verdana/CAM01/focusStep_6_verdanaRef_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step6/Verdana/CAM02/focusStep_6_verdanaRef_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step6_file_name_T = [['./step6/Times/CAM01/focusStep_6_timesR_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step6/Times/CAM02/focusStep_6_timesR_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step7_file_name_V = [['./step7/Verdana/CAM01/focusStep_7_verdanaRef_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step7/Verdana/CAM02/focusStep_7_verdanaRef_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step7_file_name_T = [['./step7/Times/CAM01/focusStep_7_timesR_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step7/Times/CAM02/focusStep_7_timesR_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step8_file_name_V = [['./step8/Verdana/CAM01/focusStep_8_verdanaRef_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step8/Verdana/CAM02/focusStep_8_verdanaRef_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step8_file_name_T = [['./step8/Times/CAM01/focusStep_8_timesR_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step8/Times/CAM02/focusStep_8_timesR_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step9_file_name_V = [['./step9/Verdana/CAM01/focusStep_9_verdanaRef_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step9/Verdana/CAM02/focusStep_9_verdanaRef_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step9_file_name_T = [['./step9/Times/CAM01/focusStep_9_timesR_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step9/Times/CAM02/focusStep_9_timesR_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step10_file_name_V = [['./step10/Verdana/CAM01/focusStep_10_verdanaRef_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step10/Verdana/CAM02/focusStep_10_verdanaRef_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step10_file_name_T = [['./step10/Times/CAM01/focusStep_10_timesR_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step10/Times/CAM02/focusStep_10_timesR_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step11_file_name_V = [['./step11/Verdana/CAM01/focusStep_11_verdanaRef_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step11/Verdana/CAM02/focusStep_11_verdanaRef_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step11_file_name_T = [['./step11/Times/CAM01/focusStep_11_timesR_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step11/Times/CAM02/focusStep_11_timesR_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step12_file_name_V = [['./step12/Verdana/CAM01/focusStep_12_verdanaRef_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step12/Verdana/CAM02/focusStep_12_verdanaRef_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step12_file_name_T = [['./step12/Times/CAM01/focusStep_12_timesR_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step12/Times/CAM02/focusStep_12_timesR_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step13_file_name_V = [['./step13/Verdana/CAM01/focusStep_13_verdanaRef_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step13/Verdana/CAM02/focusStep_13_verdanaRef_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step13_file_name_T = [['./step13/Times/CAM01/focusStep_13_timesR_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step13/Times/CAM02/focusStep_13_timesR_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step14_file_name_V = [['./step14/Verdana/CAM01/focusStep_14_verdanaRef_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step14/Verdana/CAM02/focusStep_14_verdanaRef_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step14_file_name_T = [['./step14/Times/CAM01/focusStep_14_timesR_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step14/Times/CAM02/focusStep_14_timesR_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step15_file_name_V = [['./step15/Verdana/CAM01/focusStep_15_verdanaRef_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step15/Verdana/CAM02/focusStep_15_verdanaRef_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step15_file_name_T = [['./step15/Times/CAM01/focusStep_15_timesR_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step15/Times/CAM02/focusStep_15_timesR_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step16_file_name_V = [['./step16/Verdana/CAM01/focusStep_16_verdanaRef_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step16/Verdana/CAM02/focusStep_16_verdanaRef_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step16_file_name_T = [['./step16/Times/CAM01/focusStep_16_timesR_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step16/Times/CAM02/focusStep_16_timesR_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step17_file_name_V = [['./step17/Verdana/CAM01/focusStep_17_verdanaRef_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step17/Verdana/CAM02/focusStep_17_verdanaRef_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step17_file_name_T = [['./step17/Times/CAM01/focusStep_17_timesR_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step17/Times/CAM02/focusStep_17_timesR_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step18_file_name_V = [['./step18/Verdana/CAM01/focusStep_18_verdanaRef_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step18/Verdana/CAM02/focusStep_18_verdanaRef_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step18_file_name_T = [['./step18/Times/CAM01/focusStep_18_timesR_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step18/Times/CAM02/focusStep_18_timesR_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step19_file_name_V = [['./step19/Verdana/CAM01/focusStep_19_verdanaRef_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step19/Verdana/CAM02/focusStep_19_verdanaRef_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
step19_file_name_T = [['./step19/Times/CAM01/focusStep_19_timesR_size_30_sample_0%03d.tif'% (k+1) for k in range(total_number)],['./step19/Times/CAM02/focusStep_19_timesR_size_30_sample_0%03d.tif' % (k+1) for k in range(total_number)]]
writer=SummaryWriter('./log')
total_file_name_V=[]
total_file_name_T=[]
for i in range(20):
    total_file_name_V.append(locals()['step%d_file_name_V'%i])
for i in range(20):
    total_file_name_T.append(locals()['step%d_file_name_T'%i])

def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d, 
                img.size[1] - img.size[1] % d)

    bbox = [
            int((img.size[0] - new_size[0])/2), 
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),
    ]
    img_cropped = img.crop(bbox)
    return img_cropped 
input_depth=1
pad = 'reflection'
mse=torch.nn.MSELoss().type(dtype)
ssim=SSIM().type(dtype)
net = get_net(input_depth,'multi_skip', pad,
                skip_n33d=128, 
                skip_n33u=128, 
                skip_n11=4, 
                num_scales=5,
                n_channels=1,
                upsample_mode='bilinear').type(dtype)
#net.load_state_dict(torch.load('net_down_12to19.pth'))
net=net.type(dtype)
net = torch.nn.DataParallel(net)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
#optimizer = torch.optim.SGD(net.parameters(),lr=1e-2)

import math 
kernel_size = 59
sigma = 20
# Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
x_cord = torch.arange(kernel_size)
x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
y_grid = x_grid.t()
xy_grid = torch.stack([x_grid, y_grid], dim=-1)
mean = (kernel_size - 1)/2.
variance = sigma**2.
gaussian_kernel = (1./(2.*math.pi*variance)) *\
                torch.exp(
                    -torch.sum((xy_grid - mean)**2., dim=-1) /\
                    (2*variance)
                )
gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
Kernel=gaussian_kernel.unsqueeze(0).unsqueeze(0).type(dtype)
def defocus(x): 
    out = nn.functional.conv2d(x,Kernel,padding=int((kernel_size-1)/2),bias=None)
    out=x+15*(x-out)
    return out

for _ in range(epoch): 
    show=0 
    for i in range(total_number):     
        for ff in range(8):
            f=ff+12
            img_V0 = np.array(cv2.imread(total_file_name_V[f][0][i], -1), dtype=np.float32)/255.
            img_V1 = np.array(cv2.imread(total_file_name_V[f][1][i], -1), dtype=np.float32)/255.
            img_T0 = np.array(cv2.imread(total_file_name_T[f][0][i], -1), dtype=np.float32)/255.
            img_T1 = np.array(cv2.imread(total_file_name_T[f][1][i], -1), dtype=np.float32)/255.
            for jj in range(1):
                img0=cv2.resize(img_V0, dsize=(1180, 730), interpolation=cv2.INTER_CUBIC)
                img1=cv2.resize(img_V1, dsize=(1180, 730), interpolation=cv2.INTER_CUBIC)
        
                img0=1-img0/255.
                img1=1-img1/255.
                #img0 = np.array(cv2.imread('./barbara.tif', -1), dtype=np.float32)/255.
                if img0.ndim == 2:
                    Img0 = np.expand_dims(img0, axis=0)
                    Img1 = np.expand_dims(img1, axis=0)
                else:
                    Img0 = np.transpose(img0,(2,0,1))
                    Img1 = np.transpose(img1,(2,0,1))
                #print(Img0.shape)
                c,w,h = Img0.shape
                Img0 = np.expand_dims(Img0,axis=0)
                Img1 = np.expand_dims(Img1,axis=0)
                Img_tensor0 =  torch.FloatTensor(Img0).cuda()
                Img_tensor1 =  torch.FloatTensor(Img1).cuda()
                
                img0=cv2.resize(img_T0, dsize=(1180, 730), interpolation=cv2.INTER_CUBIC)
                #img0=img_T0
                img1=cv2.resize(img_T1, dsize=(1180, 730), interpolation=cv2.INTER_CUBIC)
            
                img0=1-img0/255.
                img1=1-img1/255.
                #img0 = np.array(cv2.imread('./barbara.tif', -1), dtype=np.float32)/255.
                if img0.ndim == 2:
                    Img0 = np.expand_dims(img0, axis=0)
                    Img1 = np.expand_dims(img1, axis=0)
                else:
                    Img0 = np.transpose(img0,(2,0,1))
                    Img1 = np.transpose(img1,(2,0,1))
                #print(Img0.shape)
                c,w,h = Img0.shape
                Img0 = np.expand_dims(Img0,axis=0)
                Img1 = np.expand_dims(Img1,axis=0)
                Img_tensor02=torch.FloatTensor(Img0).cuda()
                Img_tensor12=torch.FloatTensor(Img1).cuda()
                Img_tensor0 = torch.cat((Img_tensor0,Img_tensor02),0)
                Img_tensor1 = torch.cat((Img_tensor1,Img_tensor12),0)
                for k in range(epoch_each):
                    out=net(defocus(Img_tensor1))
                    loss=-0.7*ssim(out,Img_tensor0)+mse(out,Img_tensor0)
                    #loss=mse(out,Img_tensor0)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    psnr=compare_psnr(Img0[0], out.detach().cpu().numpy()[1])
                    #psnr= compare_psnr(Img0[0], Img1[0])
                    print('epoch:',_)
                    print('num_img:',i)
                    print('loss_step%d_V:'%f ,loss.item())
                    print('psnr_step%d_V:'%f,psnr)
                    show=show+loss.item()
    show=show/720.
    writer.add_scalar('value',show,_)
    torch.save(net.module.state_dict(), 'net_down_12to19.pth')