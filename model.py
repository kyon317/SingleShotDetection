import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F




def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    #input:
    #pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    #output:
    #loss -- a single number for the value of the loss function, [1]
    
    #TODO: write a loss function for SSD
    #
    #For confidence (class labels), use cross entropy (F.cross_entropy)
    #You can try F.binary_cross_entropy and see which loss is better
    #For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    #
    #Note that you need to consider cells carrying objects and empty cells separately.
    #I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    #and reshape box to [batch_size*num_of_boxes, 4].
    #Then you need to figure out how you can get the indices of all cells carrying objects,
    #and use confidence[indices], box[indices] to select those cells.




class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()
        
        self.class_num = class_num #num_of_classes, in this assignment, 4: cat, dog, person, background
        
        #TODO: define layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride= 2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride= 1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride= 1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride= 1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.r1_l = nn.Conv2d(256, 16, 3, stride=1, padding=1, bias=True)
        self.r1_r = nn.Conv2d(256, 16, 3, stride=1, padding=1, bias=True)

        self.r2_l = nn.Conv2d(256, 16, 3, stride=1, padding=1, bias=True)
        self.r2_r = nn.Conv2d(256, 16, 3, stride=1, padding=1, bias=True)

        self.r3_l = nn.Conv2d(256, 16, 3, stride=1, padding=1, bias=True)
        self.r3_r = nn.Conv2d(256, 16, 3, stride=1, padding=1, bias=True)

        self.r4_l = nn.Conv2d(256, 16, 1, stride=1, padding=1, bias=True)
        self.r4_r = nn.Conv2d(256, 16, 1, stride=1, padding=1, bias=True)
    def forward(self, x):
        #input:
        #x -- images, [batch_size, 3, 320, 320]
        
        #TODO: define forward
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        r0 = self.conv4(x)
        r1 = self.conv5(r0)
        r2 = self.conv6(r1)
        r3 = self.conv7(r2)

        B,C,H,W = r0.shape
        r0_l = self.r1_l(r0).view(B,16,H*W)     # reshape
        r0_r = self.r1_r(r0).view(B,16,H*W)

        B, C, H, W = r1.shape
        r1_l = self.r2_l(r0).view(B, 16, H * W)  # reshape
        r1_r = self.r2_r(r0).view(B, 16, H * W)

        B, C, H, W = r2.shape
        r2_l = self.r3_l(r0).view(B, 16, H * W)
        r2_r = self.r3_r(r0).view(B, 16, H * W)

        B, C, H, W = r3.shape
        r3_l = self.r4_l(r0).view(B, 16, H * W)
        r3_r = self.r4_r(r0).view(B, 16, H * W)

        output_box = torch.concat([r0_l, r1_l, r2_l, r3_l], 2)
        B = output_box.shape[0]
        output_box = output_box.permute(0,2,1)
        bboxes = output_box.reshape((B,540,4))

        output_conf = torch.concat([r0_r, r1_r, r2_r, r3_r], 2)
        B = output_conf.shape[0]
        output_conf = output_conf.permute(0,2,1)
        output_conf = output_conf.reshape((B,540,4))
        # apply softmax on classes
        confidence = torch.softmax(output_conf, 2)
        #should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.) If yes, which dimension should you apply softmax?
        
        #sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        #confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        #bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]
        
        return confidence,bboxes










