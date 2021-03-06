# the main file for HDC2021
import os
import cv2
import numpy as np
import torch
from models import * 
from models import DLSS 
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='example/blur_tif/', help='the input blurred image path')
parser.add_argument('--output_path', type=str, default='result/', help='output path of the deblurred images')
parser.add_argument('--blur_level', type=int, default= None, help='the blur level of input')
parser.add_argument('--gpu', dest='gpu_id', type=str, default='0', help='use gpu or cpu')
parser.add_argument('--side_by_side', action='store_false', default=False, help='put the blur/deblur iamges side-by-side')
opt = parser.parse_args()

def main():
    dtype = torch.cuda.FloatTensor
    input_depth=1
    pad = 'reflection'
    net = get_net(input_depth,'multi_skip', pad,
                    skip_n33d=128, 
                    skip_n33u=128, 
                    skip_n11=4, 
                    num_scales=5,
                    n_channels=1, 
                    upsample_mode='bilinear').type(dtype)
    net.load_state_dict(torch.load('net_down.pth'))
    net = torch.nn.DataParallel(net)
    net2=DLSS.DLSS(1)
    net2.load_state_dict(torch.load('net_up.pth'))
    net2=net2.type(dtype)
    net2 = torch.nn.DataParallel(net2)
    Img_list=sorted(os.listdir(opt.input_path))
    

    for img_name in Img_list:
        file_name=opt.input_path+img_name
        img = np.array(cv2.imread(file_name, -1), dtype=np.float32)/255.
        H,W=img.shape
        img=cv2.resize(img, dsize=(int(W/2), int(H/2)), interpolation=cv2.INTER_CUBIC)
        img=1-img/255.
        if img.ndim == 2:
            Img = np.expand_dims(img, axis=0)
        else:
            Img = np.transpose(img,(2,0,1))
        Img = np.expand_dims(Img,axis=0)
        Img_tensor =  torch.FloatTensor(Img).cuda()
        with torch.no_grad():
            out=1-net2(net(Img_tensor)).data
        path=opt.output_path+img_name[0:-4]+'.png'
        out_img=torch.clamp(out.squeeze().cpu(),0.,1.)
        out_img=out_img.detach().numpy()
        cv2.imwrite(path,255*out_img)



if __name__ == '__main__':
    # Fire(main)
    main()