import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from model.DSCNet import DSCNet
from data import test_dataset
import time


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='/home/hemraj/Downloads/BDCNet/data/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')

#load the model
model = DSCNet(nInputChannels=3, n_classes=1, os=16)
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load('/home/hemraj/Downloads/BDCNet/results/demo-03-19-0/epoch_100.pth'))
model.cuda()
model.eval()

#test
test_datasets = ['DAVIS','FBMS','DAVSOD', 'MCL', 'SegTrack-V2']
for dataset in test_datasets:
    save_path = './results/' + dataset
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root=dataset_path +dataset +'/Flow/'
    test_loader = test_dataset(image_root, gt_root,depth_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt,depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.cuda()
        torch.cuda.synchronize()
        time_s = time.time()
        res, res_r,res_d= model(image,depth)
        torch.cuda.synchronize()
        time_e = time.time()
        print('Speed: %f FPS' % (1 / (time_e - time_s)))
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ',os.path.join(save_path, name))
        cv2.imwrite(os.path.join(save_path, name),res*255)

    print('Test Done!')
