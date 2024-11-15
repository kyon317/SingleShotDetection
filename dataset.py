import torch.utils.data
import numpy as np
import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import random_split

#generate default bounding boxes
def default_box_generator(layers, large_scale, small_scale):
    #input:
    #layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    #large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    #small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    
    #output:
    #boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.

    #TODO:
    #create an numpy array "boxes" to store default bounding boxes
    #you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    #the first dimension means number of cells, 10*10+5*5+3*3+1*1
    #the second dimension 4 means each cell has 4 default bounding boxes.
    #their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    #where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    #for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    #the last dimension 8 means each default bounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
    box_num = 4 * sum([layer**2 for layer in layers]) # 4*(10*10+5*5+3*3+1*1) by default
    boxes = []

    for i, layer in enumerate(layers):
        grid_size = layer # 10,5,3,1
        lsize = large_scale[i]
        ssize = small_scale[i]

        for y in range(grid_size):
            for x in range(grid_size):
                x_center = (x + 0.5)/ grid_size
                y_center = (y + 0.5)/ grid_size

                sizes = [
                    (ssize, ssize),
                    (lsize, lsize),
                    (lsize*np.sqrt(2), lsize/np.sqrt(2)),
                    (lsize/np.sqrt(2), lsize*np.sqrt(2)),
                ]

                for w,h in sizes:
                    x_min = x_center - w / 2 # apply clipping
                    x_max = x_center + w / 2
                    y_min = y_center - h / 2
                    y_max = y_center + h / 2

                    box = np.array([x_center, y_center, w, h, x_min, y_min, x_max, y_max])
                    boxes.append(box)
    boxes = np.array(boxes)
    boxes = np.clip(boxes, 0, 1)
    boxes.reshape(box_num, 8)
    return boxes


#this is an example implementation of IOU.
#It is different from the one used in YOLO, please pay attention.
#you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min,y_min,x_max,y_max):
    #input:
    #boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    #x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:,6]-boxs_default[:,4])*(boxs_default[:,7]-boxs_default[:,5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)



def match(ann_box,ann_confidence,boxs_default,threshold,cat_id,x_min,y_min,x_max,y_max):
    #input:
    #ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    #ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    #boxs_default            -- [num_of_boxes,8], default bounding boxes
    #threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    #cat_id                  -- class id, 0-cat, 1-dog, 2-person
    #x_min,y_min,x_max,y_max -- bounding box
    
    #compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min,y_min,x_max,y_max)
    ious_true = ious > threshold

    gx,gy = (x_max + x_min)/2, (y_max + y_min)/2 # get center of ground truth box
    gw,gh = x_max - x_min, y_max - y_min    # get width, height

    for i in np.where(ious_true == 1)[0]:
        px,py,pw,ph = boxs_default[i][:4] # get from default box
        tx = (gx - px) / pw
        ty = (gy - py) / ph
        tw = np.log(gw / pw)
        th = np.log(gh / ph)

        # ann_box[i] = [abs(tx),abs(ty),abs(tw),abs(th)]
        ann_box[i] = [tx,ty,tw,th]
        ann_confidence[i, :] = 0 # remove background label
        ann_confidence[i, cat_id] = 1 #cat dog person background

    if ious.max() < threshold:
        ious_true = np.argmax(ious)
        px, py, pw, ph = boxs_default[ious_true][:4]  # get from default box
        if pw == 0 or ph == 0:
            print(px,py,pw,ph)
        tx = (gx - px) / pw
        ty = (gy - py) / ph
        tw = np.log(gw / pw)
        th = np.log(gh / ph)

        ann_box[ious_true] = [tx, ty, tw, th]
        ann_confidence[ious_true,:] = 0 # remove background label
        ann_confidence[ious_true, cat_id] = 1  # cat dog person background

class COCO(torch.utils.data.Dataset):

    def __init__(self, imgdir, anndir, class_num, boxs_default, train = True, test = False, image_size=320):
        self.test = test
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        
        #overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        
        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size

        # split train set & val set
        self.train_set, self.val_set = random_split(self.img_names, (0.9, 0.1))
        if self.test: # test
            pass
        # elif self.train: # train
        #     self.img_names = self.train_set
        # else:           # validation
        #     self.img_names = self.val_set
        # reference: https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
        self.train_transforms = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.RandomCrop(224, 224, pad_if_needed = True, p = 0.5),
            A.GridDropout(ratio=0.3, unit_size_range=(10, 100), random_offset=True, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=10, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Resize(self.image_size, self.image_size),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc',min_area=100, label_fields=['labels']))
        self.test_transforms = A.Compose([
            A.Resize(self.image_size, self.image_size),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc',min_area=100, label_fields=['labels']))
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num,4], np.float32) #bounding boxes
        ann_confidence = np.zeros([self.box_num,self.class_num], np.float32) #one-hot vectors
        #one-hot vectors with four classes
        #[1,0,0,0] -> cat blue
        #[0,1,0,0] -> dog green
        #[0,0,1,0] -> person red
        #[0,0,0,1] -> background
        
        ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"
        
        img_name = self.imgdir+self.img_names[index]
        ann_name = self.anndir+self.img_names[index][:-3]+"txt"
        
        #TODO:
        #1. prepare the image [3,320,320], by reading image "img_name" first.
        image = cv2.imread(img_name)

        img_width,img_height,img_channel = image.shape

        bboxes = []
        labels = []
        #2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        if self.test: # only have txt for training and validation
            pass
        else:
            with open(ann_name, 'r') as f:
                for line in f:
                    ann_data = line.strip().split(' ')
                    class_id = int(ann_data[0])
                    gx, gy, gw, gh = float(ann_data[1]),float(ann_data[2]),float(ann_data[3]),float(ann_data[4])
                    x_min, y_min, x_max, y_max = gx, gy, gx + gw, gy + gh
                    if x_min >= x_max or y_min >= y_max: # illegal box
                        continue
                    bboxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id)


        #4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.
        if self.train:
            transformed = self.train_transforms(image=image,bboxes=bboxes,labels=labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']
        else:
            transformed= self.test_transforms(image=image,bboxes=bboxes,labels=labels)
            image = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']
        for i,box in enumerate(bboxes):
            x_min, y_min, x_max, y_max = box
            # Normalization
            x_min, y_min, x_max, y_max = x_min / self.image_size, y_min / self.image_size, x_max / self.image_size, y_max / self.image_size
            x_min, y_min, x_max, y_max = np.clip([x_min, y_min, x_max, y_max], 0, 1)
            if x_min >= x_max or y_min >= y_max:
                continue
            # 3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
            match(ann_box, ann_confidence, self.boxs_default, self.threshold, int(labels[i]), x_min, y_min, x_max, y_max)
        if self.test:
            return image, ann_box, ann_confidence, int(self.img_names[index][:-4])
        return image, ann_box, ann_confidence
