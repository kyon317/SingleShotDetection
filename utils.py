import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
# pip install pycocotools
from torchmetrics.detection import MeanAveragePrecision

from dataset import iou

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


# use [blue, green, red] to represent different classes

def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default):
    # input:
    # windowname      -- the name of the window to display the images
    # pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    # pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    # ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    # ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    # image_          -- the input image to the network
    # boxs_default    -- default bounding boxes, [num_of_boxes, 8]

    _, class_num = pred_confidence.shape
    # class_num = 4
    class_num = class_num - 1
    # class_num = 3 now, because we do not need the last class (background)

    image = np.transpose(image_, (1, 2, 0)).astype(np.uint8)
    image1 = np.zeros(image.shape, np.uint8)
    image2 = np.zeros(image.shape, np.uint8)
    image3 = np.zeros(image.shape, np.uint8)
    image4 = np.zeros(image.shape, np.uint8)
    image1[:] = image[:]
    image2[:] = image[:]
    image3[:] = image[:]
    image4[:] = image[:]
    # image1: draw ground truth bounding boxes on image1
    # image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    # image3: draw network-predicted bounding boxes on image3
    # image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)

    w, h, _ = image.shape
    # draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i, j] == 1:  # if the network/ground_truth has high confidence on cell[i] with class[j]
                # TODO:
                # image1: draw ground truth bounding boxes on image1
                # [x_center, y_center, w, h, x_min, y_min, x_max, y_max]
                gt_x, gt_y = boxs_default[i, 2] * ann_box[i, 0] + boxs_default[i, 0], boxs_default[i, 3] * ann_box[
                    i, 1] + boxs_default[i, 1]
                gt_w, gt_h = boxs_default[i, 2] * np.exp(ann_box[i, 2]), boxs_default[i, 3] * np.exp(ann_box[i, 3])
                gt_start = (int((gt_x - gt_w / 2.0) * w), int((gt_y - gt_h / 2.0) * h))
                gt_end = (int((gt_x + gt_w / 2.0) * w), int((gt_y + gt_h / 2.0) * h))
                cv2.rectangle(image1, gt_start, gt_end, colors[j], thickness=2)
                # image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                dx_min, dy_min, dx_max, dy_max = boxs_default[i, 4:]
                d_start = (int(dx_min * w), int(dy_min * h))
                d_end = (int(dx_max * w), int(dy_max * h))
                cv2.rectangle(image2, d_start, d_end, colors[j], thickness=2)

    # pred
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i, j] > 0.5:
                # TODO:
                # image3: draw network-predicted bounding boxes on image3
                gt_x, gt_y = boxs_default[i, 2] * pred_box[i, 0] + boxs_default[i, 0], boxs_default[i, 3] * pred_box[
                    i, 1] + boxs_default[i, 1]
                gt_w, gt_h = boxs_default[i, 2] * np.exp(pred_box[i, 2]), boxs_default[i, 3] * np.exp(pred_box[i, 3])
                gt_start = (int((gt_x - gt_w / 2.0) * w), int((gt_y - gt_h / 2.0) * h))
                gt_end = (int((gt_x + gt_w / 2.0) * w), int((gt_y + gt_h / 2.0) * h))
                cv2.rectangle(image3, gt_start, gt_end, colors[j], thickness=2)
                # image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                dx_min, dy_min, dx_max, dy_max = boxs_default[i, 4:]
                d_start = (int(dx_min * w), int(dy_min * h))
                d_end = (int(dx_max * w), int(dy_max * h))
                cv2.rectangle(image4, d_start, d_end, colors[j], thickness=2)
    # combine four images into one
    h, w, _ = image1.shape
    image = np.zeros([h * 2, w * 2, 3], np.uint8)
    image[:h, :w] = image1
    image[:h, w:] = image2
    image[h:, :w] = image3
    image[h:, w:] = image4
    cv2.imshow(windowname + " [[gt_box,gt_dft],[pd_box,pd_dft]]", image)
    cv2.waitKey(0)
    # cv2.waitKey(1)
    # if you are using a server, you may not be able to display the image.
    # in that case, please save the image using cv2.imwrite and check the saved image for visualization.

def visualize_res(pred_confidence, pred_box, image_, boxs_default, threshold = 0.5):
    _, class_num = pred_confidence.shape
    class_num = class_num - 1  # Exclude the background class
    class_names = ["cat", "dog", "human"]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Colors for each class (Blue, Green, Red)

    # Prepare the input image
    image = np.transpose(image_, (1, 2, 0)).astype(np.uint8)  # Convert (C, H, W) to (H, W, C)
    h, w, _ = image.shape
    image1 = np.zeros(image.shape, np.uint8)
    image1[:] = image[:]
    # Draw predicted bounding boxes
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i, j] > threshold:  # Confidence threshold
                # Calculate predicted bounding box coordinates
                gt_x, gt_y = boxs_default[i, 2] * pred_box[i, 0] + boxs_default[i, 0], \
                             boxs_default[i, 3] * pred_box[i, 1] + boxs_default[i, 1]
                gt_w, gt_h = boxs_default[i, 2] * np.exp(pred_box[i, 2]), \
                             boxs_default[i, 3] * np.exp(pred_box[i, 3])
                gt_start = (int((gt_x - gt_w / 2.0) * w), int((gt_y - gt_h / 2.0) * h))
                gt_end = (int((gt_x + gt_w / 2.0) * w), int((gt_y + gt_h / 2.0) * h))

                # Draw rectangle for predicted bounding box
                cv2.rectangle(image1, gt_start, gt_end, colors[j], thickness=2)

                # Add class name and confidence as text
                label = f"{class_names[j]}: {pred_confidence[i, j]:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                text_start = (min(gt_start[0], w - text_width - 5), max(gt_start[1] - 10, 20))  # Position text above the box
                cv2.putText(image1, label, text_start, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, color=colors[j], thickness=2)

    # Display the result
    cv2.imshow(" Predicted Bounding Boxes", image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.1, threshold=0.5):
    # TODO: non maximum suppression
    # input:
    # confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    # box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    # boxs_default -- default bounding boxes, [num_of_boxes, 8]
    # overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    # threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.

    # output:
    # depends on your implementation.
    # if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    # you can also directly return the final bounding boxes and classes, and write a new visualization function for that.
    num_boxes, num_cls = confidence_.shape
    conf = np.copy(confidence_)
    reference_box = np.zeros_like(boxs_default)
    px, py, pw, ph = boxs_default[:, 0], boxs_default[:, 1], boxs_default[:, 2], boxs_default[:, 3]
    reference_box[:, 0] = pw * box_[:, 0] + px  # gx = pw*dx + px
    reference_box[:, 1] = ph * box_[:, 1] + py  # gy = ph*dy + py
    reference_box[:, 2] = pw * np.exp(box_[:, 2])  # gw = pw * exp(dw)
    reference_box[:, 3] = ph * np.exp(box_[:, 3])  # gh = ph * exp(dh)
    reference_box[:, 4] = reference_box[:, 0] - reference_box[:, 2] / 2  # min_x
    reference_box[:, 5] = reference_box[:, 1] - reference_box[:, 3] / 2  # min_y
    reference_box[:, 6] = reference_box[:, 0] + reference_box[:, 2] / 2  # max_x
    reference_box[:, 7] = reference_box[:, 1] + reference_box[:, 3] / 2  # max_y

    B_box = []
    # pick the one with the highest probability in classes
    max_conf = np.max(conf[:, 0:num_cls - 1], axis=1)
    # pick indices that has conf > threshold
    A_box = np.where(max_conf >= threshold)[0]
    if len(A_box) == 0:  # pick max if no indices > threshold
        B_box.append(np.argmax(max_conf))
    while len(A_box) > 0:
        max_idx = np.argmax(max_conf[A_box])
        x = A_box[max_idx]
        B_box.append(x)
        # remove from A
        A_box = np.delete(A_box, max_idx)
        A_box_ = reference_box[A_box]
        max_box_reference = reference_box[x]
        ious = iou(A_box_, max_box_reference[4], max_box_reference[5], max_box_reference[6], max_box_reference[7])
        overlaps = np.where(ious > overlap)[0]
        A_box = np.delete(A_box, overlaps)
    pred_confidence_, pred_box_ = np.zeros_like(confidence_), np.copy(box_)
    for idx in B_box:
        pred_confidence_[idx] = confidence_[idx]
    return np.array(pred_confidence_), pred_box_

def save_txt(pred_box, cat_id, image_id,boxs_default):
    img = cv2.imread("data/test/images/" + str(image_id).zfill(5) + ".jpg")
    if img is None:
        return
    h,w,c = img.shape
    outPath = "results/"
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    f = open(outPath + str(image_id).zfill(5) + ".txt", "w")
    if len(pred_box) == 0:
        print("No Object Detected.")
        f.close()
        return
    class_num = cat_id.shape[1]
    for i in range(len(cat_id)):
        for j in range(class_num - 1):
            if cat_id[i, j] > 0.6:  # if the network/ground_truth has high confidence on cell[i] with class[j]
                gt_x, gt_y = boxs_default[i, 2] * pred_box[i, 0] + boxs_default[i, 0], boxs_default[i, 3] * pred_box[
                    i, 1] + boxs_default[i, 1]
                gt_w, gt_h = boxs_default[i, 2] * np.exp(pred_box[i, 2]), boxs_default[i, 3] * np.exp(pred_box[i, 3])
                x_min, y_min = (int((gt_x - gt_w / 2.0) * w), int((gt_y - gt_h / 2.0) * h))
                b_w, b_h = gt_w * w, gt_h * h
                f.write(f"{j} {x_min} {y_min} {b_w} {b_h}\n")
    f.close()
# convert to actual coordinate on image
def convertToRealSize(ann_box,pred_box,ann_confidence,pred_confidence,boxs_default, thres = 0.6):
    w,h = 320, 320
    ann_box_real = ann_box.clone()
    pred_box_real = pred_box.clone()
    for i in range(len(ann_confidence)):
        for j in range(3):
            if ann_confidence[i, j] == 1:  # if the network/ground_truth has high confidence on cell[i] with class[j]
                # ann_box
                gt_x, gt_y = boxs_default[i, 2] * ann_box[i, 0] + boxs_default[i, 0], boxs_default[i, 3] * ann_box[
                    i, 1] + boxs_default[i, 1]
                gt_w, gt_h = boxs_default[i, 2] * np.exp(ann_box[i, 2]), boxs_default[i, 3] * np.exp(ann_box[i, 3])
                gt_start = (int((gt_x - gt_w / 2.0) * w), int((gt_y - gt_h / 2.0) * h))
                gt_end = (int((gt_x + gt_w / 2.0) * w), int((gt_y + gt_h / 2.0) * h))
                ann_box_real[i,:] = torch.tensor([gt_start[0], gt_start[1], gt_end[0], gt_end[1]], dtype=ann_box.dtype) # xmin, ymin, xmax, ymax

    for i in range(len(pred_confidence)):
        for j in range(3):
            if pred_confidence[i, j] > thres:
                # pred_box
                gt_x, gt_y = boxs_default[i, 2] * pred_box[i, 0] + boxs_default[i, 0], boxs_default[i, 3] * pred_box[
                    i, 1] + boxs_default[i, 1]
                gt_w, gt_h = boxs_default[i, 2] * np.exp(pred_box[i, 2]), boxs_default[i, 3] * np.exp(pred_box[i, 3])
                gt_start = (int((gt_x - gt_w / 2.0) * w), int((gt_y - gt_h / 2.0) * h))
                gt_end = (int((gt_x + gt_w / 2.0) * w), int((gt_y + gt_h / 2.0) * h))
                pred_box_real[i,:] = torch.tensor([gt_start[0], gt_start[1], gt_end[0], gt_end[1]], dtype=pred_box.dtype)
    return ann_box_real, pred_box_real

def generate_mAP(pred_confidence_, pred_box_, ann_confidence_, ann_box_, boxs_default, thres=0.6):
    """
    Calculate mAP for object detection using IoU function for overlap calculations.

    Input:
        pred_confidence_ (np.array): Predicted confidences for each bounding box.
        pred_box_ (np.array): Predicted bounding boxes.
        ann_confidence_ (np.array): Ground truth confidences for each bounding box.
        ann_box_ (np.array): Ground truth bounding boxes.
        thres (float): IOU threshold for considering a prediction as True Positive.

    Output:
        The mean Average Precision (mAP).
    """
    # Convert numpy arrays to tensors
    pred_confidence_tensor = torch.from_numpy(pred_confidence_)
    ann_confidence_tensor = torch.from_numpy(ann_confidence_)
    pred_box_tensor = torch.from_numpy(pred_box_)
    ann_box_tensor = torch.from_numpy(ann_box_)
    ann_box_tensor,pred_box_tensor = convertToRealSize(ann_box_tensor,pred_box_tensor,ann_confidence_,pred_confidence_,boxs_default,thres)

    # drop zeroes
    non_zero_idx = pred_confidence_tensor.sum(dim=1) != 0
    pred_box_tensor = pred_box_tensor[non_zero_idx]
    pred_confidence_tensor = pred_confidence_tensor[non_zero_idx]
    non_zero_idx = ann_confidence_tensor[:,-1] != 1
    ann_confidence_tensor = ann_confidence_tensor[non_zero_idx]
    ann_box_tensor = ann_box_tensor[non_zero_idx]

    ann_labels = torch.argmax(ann_confidence_tensor, dim=1)
    ann_labels = torch.unsqueeze(ann_labels, dim=1)
    concat_anns = torch.cat([ann_labels, ann_box_tensor],dim=1)
    unique_boxes = torch.unique(concat_anns, dim=0)

    label = torch.empty((unique_boxes.shape[0],), dtype=unique_boxes.dtype)
    box = torch.empty((unique_boxes.shape[0], 4), dtype=unique_boxes.dtype)
    if unique_boxes.shape[0] > 0:
        label = unique_boxes[:, 0].to(torch.int64)
        box= unique_boxes[:, 1:]

    # Extract labels by finding the maximum confidence score per box
    pred_labels = torch.argmax(pred_confidence_tensor, dim=1)
    pred_scores = pred_confidence_tensor.max(dim=1)[0] # Take max confidence per box


    # Prepare predictions and targets as lists of dictionaries
    preds = [{
        "boxes": pred_box_tensor,
        "scores": pred_scores,
        "labels": pred_labels,
    }]
    targets = [{
        "boxes": box,
        "labels": label,
    }]
    # Initialize MeanAveragePrecision metric and update with predictions and targets
    metric = MeanAveragePrecision(iou_type="bbox",extended_summary=True)
    metric.warn_on_many_detections = False
    metric.update(preds, targets)
    res = metric.compute()
    return res,pred_confidence_tensor[0,:3] if pred_confidence_tensor is not None else None,label[0] if label is not None else None

# plot precision-recall curve
def plot_precision_recall(precision, recall, cls_num):
    colors = ['red', 'blue', 'green']
    labels = ['Human', 'Cat', 'Dog']
    plt.figure(figsize=(8, 6))
    for i in range(cls_num):
        plt.plot(recall[i], precision[i], label=labels[i], color=colors[i])

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=16)
    plt.legend()
    plt.grid(True)

    plt.show()