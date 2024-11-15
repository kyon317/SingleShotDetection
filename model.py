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
    batch_size, num_box,num_classes = pred_confidence.shape
    # Reshape
    pred_confidence = pred_confidence.view(batch_size * num_box, num_classes)
    pred_box = pred_box.view(batch_size * num_box, num_classes)
    ann_confidence = ann_confidence.view(batch_size * num_box, num_classes)
    ann_box = ann_box.view(batch_size * num_box, num_classes)

    # Get obj & noobj indices
    obj_indices = torch.where(ann_confidence[:, -1] != 1)[0]
    noobj_indices = torch.where(ann_confidence[:, -1] == 1)[0]

    obj_conf_pred = pred_confidence[obj_indices]
    obj_conf_gt = ann_confidence[obj_indices]

    noobj_conf_pred = pred_confidence[noobj_indices]
    noobj_conf_gt = ann_confidence[noobj_indices]

    obj_box = pred_box[obj_indices]
    obj_box_gt = ann_box[obj_indices]

    loss_cls = F.cross_entropy(obj_conf_pred, obj_conf_gt) + 3 * F.cross_entropy(noobj_conf_pred, noobj_conf_gt)
    loss_box = F.smooth_l1_loss(obj_box, obj_box_gt)

    return loss_cls + loss_box


class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()
        
        self.class_num = class_num #num_of_classes, in this assignment, 4: cat, dog, person, background
        
        #TODO: define layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride= 2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride= 1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride= 1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride= 1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.r1_l = nn.Conv2d(256, 16, 3, stride=1, padding=1, bias=True)
        self.r1_r = nn.Conv2d(256, 16, 3, stride=1, padding=1, bias=True)

        self.r2_l = nn.Conv2d(256, 16, 3, stride=1, padding=1, bias=True)
        self.r2_r = nn.Conv2d(256, 16, 3, stride=1, padding=1, bias=True)

        self.r3_l = nn.Conv2d(256, 16, 3, stride=1, padding=1, bias=True)
        self.r3_r = nn.Conv2d(256, 16, 3, stride=1, padding=1, bias=True)

        self.r4_l = nn.Conv2d(256, 16, 1, stride=1, padding=0, bias=True)
        self.r4_r = nn.Conv2d(256, 16, 1, stride=1, padding=0, bias=True)
    def forward(self, x):
        #input:
        #x -- images, [batch_size, 3, 320, 320]
        # define forward
        x = x / 255.0
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        r0 = self.conv4(x)
        r1 = self.conv5(r0)
        r2 = self.conv6(r1)
        r3 = self.conv7(r2)

        B, C, H, W = r0.shape
        r0_l = self.r1_l(r0).view(B, 16, H * W)     # reshape
        r0_r = self.r1_r(r0).view(B, 16, H * W)

        B, C, H, W = r1.shape
        r1_l = self.r2_l(r1).view(B, 16, H * W)  # reshape
        r1_r = self.r2_r(r1).view(B, 16, H * W)

        B, C, H, W = r2.shape
        r2_l = self.r3_l(r2).view(B, 16, H * W)
        r2_r = self.r3_r(r2).view(B, 16, H * W)

        B, C, H, W = r3.shape
        r3_l = self.r4_l(r3).view(B, 16, H * W)
        r3_r = self.r4_r(r3).view(B, 16, H * W)

        output_box = torch.concat([r0_l, r1_l, r2_l, r3_l], 2)
        B= output_box.shape[0]
        output_box = output_box.permute(0,2,1)
        bboxes = output_box.reshape((B,540,self.class_num))

        output_conf = torch.concat([r0_r, r1_r, r2_r, r3_r], 2)
        B = output_conf.shape[0]
        output_conf = output_conf.permute(0,2,1)
        output_conf = output_conf.reshape((B,540,self.class_num))
        # apply softmax on classes
        confidence = torch.softmax(output_conf, 2)
        #should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.) If yes, which dimension should you apply softmax?

        return confidence,bboxes










