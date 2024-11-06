import cv2
from PIL import Image, ImageDraw

import utils
from dataset import default_box_generator, COCO
import matplotlib.pyplot as plt
import numpy as np
from utils import visualize_pred
from vis import plot_boxes
class_num = 4 #cat dog person background

def draw_annotation_box():
    image_path = "data/train/images/00003.jpg"
    image = Image.open(image_path)

    annotation_path = "data/train/annotations/00003.txt"
    with open(annotation_path, "r") as file:
        line = file.readline().strip()
        class_id, xmin, ymin, w, h = map(float, line.split())
    xmax = xmin + w
    ymax = ymin + h
    draw = ImageDraw.Draw(image)
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
    image.show()

if __name__ == '__main__':
    boxs_default = default_box_generator([10, 5, 3, 1], [0.2, 0.4, 0.6, 0.8], [0.1, 0.3, 0.5, 0.7])

    dataset = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train=True, test = False,image_size=320)
    dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train=False, test=True, image_size=320)
    image, ann_box, ann_confidence = dataset[0]
    # visualize_pred("dataset[2]", ann_confidence,ann_box,ann_confidence,ann_box,image,boxs_default)
    # draw_annotation_box()
    x,y = utils.non_maximum_suppression(ann_confidence,ann_box,boxs_default)
    visualize_pred("dataset[2]", x,y,ann_confidence,ann_box,image,boxs_default)
    print(x, y)


