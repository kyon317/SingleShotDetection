import numpy as np
import cv2
from dataset import iou


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use [blue green red] to represent different classes

def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)
    
    # image = np.transpose(image_, (1,2,0)).astype(np.uint8)
    image = np.transpose(image_.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    image = np.transpose(image, (1, 2, 0))
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]
    image4[:]=image[:]
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)

    w,h,_ = image.shape
    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i,j]==1: #if the network/ground_truth has high confidence on cell[i] with class[j]
                print("GT")
                #TODO:
                #image1: draw ground truth bounding boxes on image1
                # [x_center, y_center, w, h, x_min, y_min, x_max, y_max]
                gt_x, gt_y = boxs_default[i,2] * ann_box[i,0] + boxs_default[i,0], boxs_default[i,3] * ann_box[i,1] + boxs_default[i,1]
                gt_w, gt_h = boxs_default[i,2] * np.exp(ann_box[i,2]), boxs_default[i,3] * np.exp(ann_box[i,3])
                gt_start = (int((gt_x - gt_w / 2.0) * w), int((gt_y - gt_h / 2.0) * h))
                gt_end = (int((gt_x + gt_w / 2.0) * w), int((gt_y + gt_h / 2.0) * h))
                cv2.rectangle(image1, gt_start, gt_end, colors[j], thickness=2)
                # image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                dx_min,dy_min,dx_max,dy_max = boxs_default[i,4:]
                d_start = (int(dx_min * w), int(dy_min * h))
                d_end = (int(dx_max * w), int(dy_max * h))
                cv2.rectangle(image2, d_start, d_end, colors[j], thickness=2)
                print(f'{gt_x,gt_y,dx_min,dy_min,dx_max,dy_max}')
    
    #pred
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i,j]>0.5:
                print("PRED")
                #TODO:
                #image3: draw network-predicted bounding boxes on image3
                gt_x, gt_y = boxs_default[i, 2] * pred_box[i, 0] + boxs_default[i, 0], boxs_default[i, 3] * pred_box[
                    i, 1] + boxs_default[i, 1]
                gt_w, gt_h = boxs_default[i, 2] * np.exp(pred_box[i, 2]), boxs_default[i, 3] * np.exp(pred_box[i, 3])
                gt_start = (int((gt_x - gt_w / 2.0) * w), int((gt_y - gt_h / 2.0) * h))
                gt_end = (int((gt_x + gt_w / 2.0) * w), int((gt_y + gt_h / 2.0) * h))
                cv2.rectangle(image3, gt_start, gt_end, colors[j], thickness=2)
                #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                dx_min, dy_min, dx_max, dy_max = boxs_default[i, 4:]
                d_start = (int(dx_min * w), int(dy_min * h))
                d_end = (int(dx_max * w), int(dy_max * h))
                cv2.rectangle(image4, d_start, d_end, colors[j], thickness=2)
                print(f'{gt_x, gt_y, dx_min, dy_min, dx_max, dy_max}')
    #combine four images into one
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4
    cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    cv2.waitKey(0)
    # cv2.waitKey(1)
    #if you are using a server, you may not be able to display the image.
    #in that case, please save the image using cv2.imwrite and check the saved image for visualization.



def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.5, threshold=0.5):
    #TODO: non maximum suppression
    #input:
    #confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #boxs_default -- default bounding boxes, [num_of_boxes, 8]
    #overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    #threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.
    
    #output:
    #depends on your implementation.
    #if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    #you can also directly return the final bounding boxes and classes, and write a new visualization function for that.
    num_boxes, num_cls = confidence_.shape
    conf = np.copy(confidence_)
    reference_box = np.zeros_like(boxs_default)
    px, py, pw, ph = boxs_default[:,0], boxs_default[:,1], boxs_default[:,2], boxs_default[:,3]
    reference_box[:,0] = pw*box_[:,0] + px    # gx = pw*dx + px
    reference_box[:,1] = py*box_[:,1] + py    # gy = py*dy + py
    reference_box[:,2] = pw*np.exp(box_[:,2])   # gw = pw * exp(dw)
    reference_box[:,3] = ph*np.exp(box_[:,3])   # gh = ph * exp(dh)
    reference_box[:,4] = reference_box[:,0] - reference_box[:,2] / 2 # min_x
    reference_box[:,5] = reference_box[:,1] - reference_box[:,3] / 2 # min_y
    reference_box[:,6] = reference_box[:,0] + reference_box[:,2] / 2 # max_x
    reference_box[:,7] = reference_box[:,1] + reference_box[:,3] / 2 # max_y

    B_box = []
    # pick the one with the highest probability in classes
    max_conf = np.max(conf[:,0:num_cls-1], axis=1)
    # pick indices > threshold
    A_box = np.where(max_conf >= threshold)[0]
    if len(A_box) == 0: # pick max if no indices > threshold
        B_box.append(np.argmax(max_conf))
    while len(A_box) > 0:
        max_idx = np.argmax(max_conf[A_box])
        x = A_box[max_idx]
        B_box.append(x)
        # remove from A
        A_box = np.delete(A_box, max_idx)
        A_box_ = reference_box[A_box]
        max_box_reference = reference_box[x]
        ious = iou(A_box_, max_box_reference[4],max_box_reference[5],max_box_reference[6],max_box_reference[7])
        overlaps = np.where(ious > overlap)[0]
        A_box = np.delete(A_box, overlaps)
    pred_confidence_, pred_box_ = np.zeros_like(confidence_),np.copy(box_)
    for idx in B_box:
        pred_confidence_[idx] = confidence_[idx]
    return np.array(pred_confidence_),pred_box_

def generate_mAP():
    #TODO: Generate mAP
    pass








