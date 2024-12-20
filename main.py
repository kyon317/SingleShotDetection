import argparse
import os
import numpy as np
import time
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import WeightedRandomSampler
from torchmetrics.functional import precision_recall_curve
from torchmetrics.classification import MulticlassPrecisionRecallCurve
import wandb
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import *
from model import *
from utils import *
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
#please google how to use argparse
#a short intro:
#to train: python main.py
#to test:  python main.py --test


class_num = 4 #cat dog person background

num_epochs = 1
batch_size = 32


boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])


#Create network
network = SSD(class_num)
network.cuda()
network.load_state_dict(torch.load('network_150.pth'))
cudnn.benchmark = True


if not args.test:
    dataset = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = True,test=False,image_size=320)
    dataset_train, dataset_val = random_split(dataset, (0.9, 0.1)) # split train vs validation
    dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)
    
    optimizer = optim.Adam(network.parameters(), lr = 1e-4, weight_decay = 5e-4)
    # optimizer = optim.SGD(network.parameters(), lr = 1e-3, momentum = 0.9, weight_decay = 5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    
    start_time = time.time()
    precision_, recall_, thres = 0,0,0.6
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(project="assignment3_ssd_training", config={
        "learning_rate": 1e-3,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "image_size": 320
    })
    #TRAINING
    for epoch in range(num_epochs):

        network.train()

        avg_loss = 0
        avg_count = 0
        current_loss = 0
        # Wrap the training loop with tqdm for progress bar
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}")
        for i, data in enumerate(progress_bar, 0):
            images_, ann_box_, ann_confidence_ = data
            images = images_.to(device).float()

            ann_box = ann_box_.to(device)
            ann_confidence = ann_confidence_.to(device)

            optimizer.zero_grad()
            pred_confidence, pred_box = network(images)
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            loss_net.backward()
            optimizer.step()
            
            avg_loss += loss_net.item()
            avg_count += 1
            current_loss = avg_loss / avg_count

            progress_bar.set_description(f"Training Epoch {epoch + 1}/{num_epochs}, Train_Loss: {current_loss:.4f}")
        # reduce lr if stuck in a plateau for 5 epochs
        scheduler.step(current_loss)
        print('[%d] time: %f train loss: %f' % (epoch, time.time()-start_time, current_loss))
        wandb.log({'epoch_loss': current_loss})

        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        # visualize_pred("train", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        
        
        #VALIDATION
        network.eval()

        # use the training set to train and the validation set to evaluate
        progress_bar = tqdm(dataloader_val, desc=f"Validation Epoch {epoch + 1}/{num_epochs}")
        cnt = 0
        total_mAP = 0
        metric = MulticlassPrecisionRecallCurve(num_classes=3)
        # Use tqdm for the validation loop as well
        for i, data in enumerate(progress_bar, 0):
            images_, ann_box_, ann_confidence_ = data
            images = images_.to(device)

            ann_box = ann_box_.to(device)
            ann_confidence = ann_confidence_.to(device)

            pred_confidence, pred_box = network(images)
            
            pred_confidence_ = pred_confidence.detach().cpu().numpy()
            pred_box_ = pred_box.detach().cpu().numpy()
            #pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
            #pred_box_ = pred_box[0].detach().cpu().numpy()
            pred_confidence_suppressed, pred_box_suppressed = non_maximum_suppression(pred_confidence[0].detach().cpu().numpy(), pred_box[0].detach().cpu().numpy(), boxs_default)

            #optional: implement a function to accumulate precision and recall to compute mAP or F1.
            map_res, preds, targets = generate_mAP(
                pred_confidence_=pred_confidence_suppressed, pred_box_=pred_box_suppressed,
                ann_confidence_=ann_confidence_[0].detach().cpu().numpy(), ann_box_=ann_box_[0].detach().cpu().numpy(),
                boxs_default=boxs_default
            )
            mAP = map_res["map_50"].item() if map_res["map_50"].item() > 0 else 0
            total_mAP += mAP
            cnt += 1
            if preds is not None and targets is not None:
                preds = preds.unsqueeze(0) if preds.dim() == 1 else preds
                targets = targets.unsqueeze(0) if targets.dim() == 0 else targets
                metric.update(preds, targets)
            progress_bar.set_description(f"Validation Epoch {epoch + 1}/{num_epochs}, mAP: {mAP:.4f}")
        avg_mAP = total_mAP / cnt
        wandb.log({'mAP': avg_mAP * 100})
        fig_, ax_ = metric.plot(score=True)

        fig_.savefig("precision_recall_curve.png")

        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        # visualize_pred("val", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)

        #save weights
        if epoch%10==9:
            #save last network
            print('saving net...')
            torch.save(network.state_dict(), 'network.pth')


else:
    #TEST
    dataset_test = COCO("data/test/images/", "data/test/annotations/", class_num, boxs_default, train = False, test = True, image_size=320)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    # dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = False,test=True,image_size=320)
    # dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    network.load_state_dict(torch.load('network_150.pth'))
    network.eval()
    
    for i, data in enumerate(dataloader_test, 0):
        images_, ann_box_, ann_confidence_, image_id = data
        images = images_.cuda()
        ann_box = ann_box_.cuda()
        ann_confidence = ann_confidence_.cuda()

        pred_confidence, pred_box = network(images)

        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        
        pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default, overlap=0.1,threshold=0.9)


        # save predicted bounding boxes and classes to a txt file.
        save_txt(pred_box_, pred_confidence_, image_id.item(), boxs_default)

        # if i % 7 == 0:
        visualize_res(pred_confidence_, pred_box_, images_[0].numpy(), boxs_default)
        cv2.waitKey(1000)



