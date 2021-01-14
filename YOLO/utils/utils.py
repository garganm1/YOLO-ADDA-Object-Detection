from __future__ import division
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """

    if not x1y1x2y2:
        # Transform from center and width to exact coordinates of top left and bottom right corners
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou





def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    prediction: (batch_size,10647,5+num_classes)

    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    # len(prediction()) give number of images

    output = [None for _ in range(len(prediction))] # List of length (= number of images) all filled with None
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold

        # image_pred:(10647,16) i.e. for each image
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze() #tensor of size 10647, True where pred obj prob > threshold
        
        image_pred = image_pred[conf_mask] # (n, 16) where n is the no. of bounding boxes whose pred obj prob > threshold
    
       
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1, keepdim=True)

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
     

        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()

        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]

            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)

            detections_class = detections_class[conf_sort_index]
            
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]
            
            max_detections = torch.cat(max_detections).data
            
            # Add max detections to outputs
            output[image_i] = (
                max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
            )

    
    # output is a list of length = number of images, where each elemnet is a tensor of shape (number of detections * 7)
    # 7 attr are (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
    return output


def build_targets(
    pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, num_classes, grid_size, ignore_thres, img_dim
):
    # targets if of shape (B,1,5). Here one represents total number of actual banned items present in a single image
    nB = target.size(0) # Number of images in the batch
    nA = num_anchors
    nC = num_classes
    nG = grid_size
    mask = torch.zeros(nB, nA, nG, nG)  # (B, 3, 13, 13)
    conf_mask = torch.ones(nB, nA, nG, nG)  # (B, 3, 13, 13)
    tx = torch.zeros(nB, nA, nG, nG)  # (B, 3, 13, 13)
    ty = torch.zeros(nB, nA, nG, nG)  # (B, 3, 13, 13)
    tw = torch.zeros(nB, nA, nG, nG)  # (B, 3, 13, 13)
    th = torch.zeros(nB, nA, nG, nG)  # (B, 3, 13, 13)
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)  # (B, 3, 13, 13)
    tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)  # (B, 3, 13, 13, num_classes)

    nGT = 0 # Keeps the track of total number of objects present in a single batch
    nCorrect = 0 # Keeps the track of true detections present in a single batch
    for b in range(nB):
        # For evry image in the batch
        for t in range(target.shape[1]):

            # For every detection in one image. In out case shape of target (B,1,5). Here 1 implies number of detections/image = 1 
            #What if there are different number of banned items in different images of a batch??????????

            # Case where no object lies in the image to be trained on
            
            # When there is no object to be detected in the actual image, statemnets foloowing doesn't run.
            # Conf mask for al the bbox stays at 1 which would force the model to have low obj conf (controlled by noobj loss)
            
            if target[b, t].sum() == 0:
                continue

            nGT += 1  # Increasing by 1 for every object present 
            # Convert to position relative to box

            # Scaling original bbox values to the current grid scale (13*13/26*26/52*52). x,y are now the coordinates from
            # the top left corner of the scaled image
            # gx,gy,gw,gh are integers
            gx = target[b, t, 1] * nG
            gy = target[b, t, 2] * nG
            gw = target[b, t, 3] * nG
            gh = target[b, t, 4] * nG
            # Get grid box indices which us responsible for detections
            gi = int(gx)
            gj = int(gy)

            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0) #(1,4)

            # Since only areas are being compared between true bbox (scaled) and anchor boxes,
            # top left coordinates are set to 0 and as a result bottom coordinates are height and width
            # This way, the best anchor box that matches with true bbox is obtained

            # tensor([[0.0000, 0.0000, 2.1911, 2.3981]]) represents # [x1,y1,x2,y2]

            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1)) 
            
        #     tensor([[ 0.0000,  0.0000,  3.6250,  2.8125], represents # [x1,y1,x2,y2] of one anchor box
        # [ 0.0000,  0.0000,  4.8750,  6.1875],
        # [ 0.0000,  0.0000, 11.6562, 10.1875]])

            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_iou(gt_box, anchor_shapes) # tensor of size 3 denoting iou of target bbox against each anchor box

            # Where the overlap is larger than threshold set mask to zero (ignore)
            # pred_boxes[b,:, gj, gi,:] represents the attributes of 3 bbox of the grid cell respomsible for detection
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0
            # print (conf_mask)
            # print (conf_mask.shape)
            # input()

            # Find the best matching anchor box
            # Index of best matching anchor box with target bbox stored here
            best_n = np.argmax(anch_ious)

            # Get ground truth box
            # gt_box contains true bbox dimensions transformed by the grid scale
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0) #(1,4)

            # Get the best prediction
            # Out of the three bbox predicted by the concerned grid cell, pick the one whose corresponding anchor has the max iou with true bbox
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)  # (1, 4)

            # Originaly the mask is of shape (B,3,13,13) and all set to 0. So mask[b, best_n, gj, gi] represents 
            # the value of mask for the bbox (out of 3 bbox), predicted my model, for which the cooresponding anchor box 
            # had the max iou with true bbox, for that image and the reponsible grid cell 
            mask[b, best_n, gj, gi] = 1

            # 
            conf_mask[b, best_n, gj, gi] = 1

            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi   # Normalized x-coordinate wrt to the grid cell's top left corner
            ty[b, best_n, gj, gi] = gy - gj   # Normalized y-coordinate wrt to the grid cell's top left corner
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
            # One-hot encoding of label
            target_label = int(target[b, t, 0])  # Obtaining target label of current image and object
            tcls[b, best_n, gj, gi, target_label] = 1  # Initializing value of anchor with best iou and for that target label in that image to 1
            tconf[b, best_n, gj, gi] = 1

            # Calculate iou between ground truth and best matching prediction (in grid scale size with dimensions of centre, width & height)
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)

            # Storing predicted label which is the label with maximum probability in the prediction corresponding to the best anchor box & grid cell
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])

            # Storing best objectness probability which is the one in the prediction corresponding to the best anchor box & grid cell
            score = pred_conf[b, best_n, gj, gi]

            # Check if iou of the predicted wrt to ground truth is > 0.5 or not
            # Check if the predicted label is the same as ground truth label
            # Check if the probability that an object exists computed by the model is > 0.5
            if iou > 0.5 and pred_label == target_label and score > 0.5:
                nCorrect += 1  # If all conditions are true, correct prediction is made. Increase count by 1

    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.from_numpy(np.eye(num_classes, dtype="uint8")[y])
