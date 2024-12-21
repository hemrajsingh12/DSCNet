import os
import sys
import time
import torch
import shutil
import logging
import argparse
import numpy as np
import torch.nn as nn
from os.path import defpath
from datetime import datetime
from data import get_loader
import torch.nn.functional as F
sys.path.append('./models')
from model.DSCNet import DSCNet
from options import opt,save_path
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from utils import clip_gradient, adjust_lr, ComboLoss, IoULoss, DiceBCELoss

#...... Argument....
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=20, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=60, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default='/content/drive/MyDrive/models/resnet50-19c8e357.pth', help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
parser.add_argument('--test_rgb_root', type=str, default='/content/drive/MyDrive/data/RGB/', help='the test rgb images root')
parser.add_argument('--test_depth_root', type=str, default='/content/drive/MyDrive/data/Flow/', help='the test depth images root')
parser.add_argument('--test_gt_root', type=str, default='/content/drive/MyDrive/data/GT/', help='the test gt images root')
parser.add_argument('--save_path', type=str, default='', help='the path to save models and logs')
parser.add_argument('--train_type', type=str, default='finetune', help='finetune or pretrain_rgb or pretrain_flow')
opt = parser.parse_args()

def save_path():
    run = 0
    save_folder = "/content/drive/MyDrive/results/demo-%s-%s-%d" % (time.strftime("%m"), time.strftime("%d"), run)
    while os.path.exists(save_folder) :
        run += 1
        save_folder = "/content/drive/MyDrive/results/demo-%s-%s-%d" % (time.strftime("%m"), time.strftime("%d"), run)

    if os.path.exists(save_folder):
        is_exist_pth = 0
        for i in os.listdir(save_folder):
            if 'pth' in i:
                is_exist_pth = 1
        save_folder = "/content/results/demo-%s-%s-%d" % (time.strftime("%m"), time.strftime("%d"), run)
        if is_exist_pth == 0:
            shutil.rmtree(save_folder)

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)


    return save_folder

#bce = F.binary_cross_entropy_with_logits()
#train function
def train(train_loader, model, optimizer, epoch, save_path):

    global step
    model.train() # For training
    loss_all=0
    epoch_step=0
    try:
        for i, (images, gts, flow) in enumerate(train_loader, start=1):
            #  normalize the dataset in tensor
            images = images.cuda()
            gts = gts.cuda()
            depths=flow.cuda()

            # Feed-forward input data through the network
            s, s_r, s_d = model(images, depths)
            
            # Compute feed-forward loss/error
            loss1 =  F.binary_cross_entropy_with_logits(s, gts)
            loss2 =  F.binary_cross_entropy_with_logits(s_r, gts)
            loss3 =  F.binary_cross_entropy_with_logits(s_d, gts)

            # IoU loss
            #loss4 = iou(s, gts)
            #loss5 = iou(s_r, gts)
            #loss6 = iou(s_d, gts)

            # Combo loss
            #loss7 = combo(s, images, gts)
            #loss8 = combo(s_r, images,  gts)
            #loss9 = combo(s_d, images, gts)

            loss = (loss1 + loss2 / 2 + loss3 / 2)# +(loss4 + loss5 / 2 + loss6 /2) + (loss7 + loss8 / 2 + loss9 / 2)

            optimizer.zero_grad() # # Initialize gradients to zero
            loss.backward()  # backpropagate the loss through the model

            clip_gradient(optimizer, opt.clip)  
            optimizer.step() #  Accumulate loss per batch
            step+=1
            epoch_step+=1 # increment the epoch step
            loss_all+=loss.data # here we are summing up the losses as we go
            if i % 100 == 0 or i == total_step or i==1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} '.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss1.data))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} '.
                    format( epoch, opt.epoch, i, total_step, loss1.data))
                writer.add_scalar('Loss', loss1.data, global_step=step)
                writer.add_scalar('Loss_r', loss2.data, global_step=step)
                writer.add_scalar('Loss_d', loss3.data, global_step=step)
                res = s[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('s', torch.tensor(res), step, dataformats='HW')
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                res = s_r[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('s_r', torch.tensor(res), step, dataformats='HW')
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                res = s_d[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('s_d', torch.tensor(res), step, dataformats='HW')
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)

                writer.add_image('Ground_truth', grid_image, step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(depths[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('depth', grid_image, step)

        loss_all/=epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format( epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 50 == 0:
            torch.save(model.state_dict(), save_path+'/epoch_{}.pth'.format(epoch)) # save the weighted of the model in terms of epoch_1.....100
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path+'/epoch_{}.pth'.format(epoch+1))
        print('save checkpoints successfully!')
        raise
        
 
if __name__ == '__main__':

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    cudnn.benchmark = True

    model = DSCNet(nInputChannels=3, n_classes=1, os=32)
    if (opt.load is not None):
        model.load_state_dict(torch.load(opt.load),strict=False)
        print('load model from ', opt.load)
    model = nn.DataParallel(model)

    model.cuda()
    params = model.parameters()
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(num_params)
    # using the optimizer
    
    optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad,model.parameters()), 0.01,weight_decay=0.0005)#opt.lr)
    #optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),opt.lr)

    # set the path of the dateset and training type
    if opt.train_type == 'finetune':
        #save_path = '../snapshot/{}/'.format(opt.trainset)
        save_path = save_path()
        # ---- data preparing ----
        src_dir = '/content/drive/MyDrive/data'
        image_root = src_dir + '/RGB/'
        flow_root = src_dir + '/depth/'
        gt_root = src_dir + '/GT/'

        train_loader = get_loader(image_root=image_root, depth_root=flow_root, gt_root=gt_root,
                                        batchsize=opt.batchsize, trainsize=opt.trainsize, shuffle=True,
                                        num_workers=4, pin_memory=True)
        total_step = len(train_loader)
        #
    else:
        raise AttributeError('No train_type: {}'.format(opt.train_type))
    print('load data...')
    logging.basicConfig(filename=save_path + '/log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Train")
    logging.info("Config")
    logging.info(
        'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
            opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
            opt.decay_epoch))

    # set loss function
    #bce = nn.BCEWithLogitsLoss()
    #iou = nn.SmoothL1Loss()
    #combo = nn.TripletMarginLoss() 

    step = 0
    writer = SummaryWriter(save_path + '/summary')
    #best_mae = 1
    best_epoch = 0

    print("Start train...")
    for epoch in range(1, opt.epoch):
        train(train_loader, model, optimizer, epoch, save_path)
        
