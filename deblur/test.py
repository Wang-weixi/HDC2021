import torch
import os
import cv2
import torch.nn.functional as F 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np 
from models import * 
from models import DLSS 
from utils.denoising_utils import *
from skimage.measure import compare_psnr
dtype = torch.cuda.FloatTensor
total_number=100
epoch_each=1
epoch=1
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

input_depth=1
pad = 'reflection'
mse=torch.nn.MSELoss().type(dtype)
net = get_net(input_depth,'multi_skip', pad,
                skip_n33d=128, 
                skip_n33u=128, 
                skip_n11=4, 
                num_scales=5,
                n_channels=1, 
                upsample_mode='bilinear').type(dtype)
#net=torch.load('net_down_final.pth')
net.load_state_dict(torch.load('net_down_13to19.pth'))
net = torch.nn.DataParallel(net)

#net2 = get_net(input_depth,'skip', pad,
                #skip_n33d=96, 
                #skip_n33u=96, 
                #skip_n11=4, 
                #num_scales=5,
                #n_channels=1, 
                #upsample_mode='bilinear').type(dtype)
net2=DLSS.DLSS(1)
#net2=torch.load('net_final_7.pth')
net2.load_state_dict(torch.load('net_up.pth'))
net2=net2.type(dtype)
net2 = torch.nn.DataParallel(net2)
psnr=0
#optimizer = torch.optim.SGD(net.parameters(),lr=1e-3)

import math
kernel_size = 49
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
 
psnr_total=0  
for k in range(10):
    i=90+k
    file_name=step15_file_name_T
    img_V0 = np.array(cv2.imread(file_name[0][i], -1), dtype=np.float32)/255.
    img_V1 = np.array(cv2.imread(file_name[1][i], -1), dtype=np.float32)/255.
    #img_V0=cv2.resize(img_V0, dsize=(1180, 730), interpolation=cv2.INTER_CUBIC)
    img_V1=cv2.resize(img_V1, dsize=(1180, 730), interpolation=cv2.INTER_CUBIC)
   
    img0=img_V0/255. 
    img1=1-img_V1/255.

  
    if img0.ndim == 2:
        Img0 = np.expand_dims(img0, axis=0)
        Img1 = np.expand_dims(img1, axis=0)

    else:
        Img0 = np.transpose(img0,(2,0,1))
        Img1 = np.transpose(img1,(2,0,1))

 
    c,w,h = Img0.shape
    Img0 = np.expand_dims(Img0,axis=0)
    Img1 = np.expand_dims(Img1,axis=0)

    Img_tensor0 =  torch.FloatTensor(Img0).cuda()
    Img_tensor1 =  torch.FloatTensor(Img1).cuda()
    with torch.no_grad():
        out=1-net2(net(defocus(Img_tensor1))).data
        psnr=compare_psnr(Img0[0], out.detach().cpu().numpy()[0])
        psnr_total=psnr_total+psnr
    path='result/'+'truth_Ver%d.png'%k
    out_img=torch.clamp(out.squeeze().cpu(),0.,1.)
    #out_img=torch.clamp((1-(defocus(Img_tensor1)).data).squeeze().cpu(),0.,1.)
    out_img=out_img.detach().numpy()
    #out_img=np.sign(out_img)
    print(out_img.shape)
    cv2.imwrite(path,255*out_img)
    print(psnr) 
print(psnr_total/10)