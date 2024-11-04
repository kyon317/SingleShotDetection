import cv2

from dataset import default_box_generator, COCO
import matplotlib.pyplot as plt
import numpy as np
from utils import visualize_pred
from vis import plot_boxes
class_num = 4 #cat dog person background

if __name__ == '__main__':
    boxs_default = default_box_generator([10, 5, 3, 1], [0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.7])

    dataset = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train=True, test = False,image_size=320)
    dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train=False, test=True, image_size=320)
    image, ann_box, ann_confidence = dataset[1]
    visualize_pred("dataset[2]", ann_confidence,ann_box,ann_confidence,ann_box,image,boxs_default)
