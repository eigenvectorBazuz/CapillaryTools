import torch
import torchvision.ops.boxes as bops
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from rbo import rbo

# Use torchvision for IoU calculation
def iou_torchvision(box1, box2):
    box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
    
    box1_tensor = torch.tensor([box1], dtype=torch.float32)
    box2_tensor = torch.tensor([box2], dtype=torch.float32)
    
    iou_value = bops.box_iou(box1_tensor, box2_tensor)
    return iou_value.item()

# Function to assign unique IDs based on IoU matching across two lists of bboxes
def assign_bbox_ids(L1, L2, iou=0.9):
    next_id = 1
    L1_ids = []
    L2_ids = [-1] * len(L2)  # Pre-fill L2 with placeholders
    matched_L2 = set()

    # Assign IDs to L1 and L2
    for i, box1 in enumerate(L1):
        matched = False
        for j, box2 in enumerate(L2):
            if j in matched_L2:
                continue
            if iou_torchvision(box1, box2) >= iou:
                L1_ids.append(next_id)
                L2_ids[j] = next_id  # Match L2[j] with L1[i]
                matched_L2.add(j)
                matched = True
                next_id += 1
                break
        if not matched:
            L1_ids.append(next_id)
            next_id += 1

    # Assign IDs to unmatched boxes in L2
    for j in range(len(L2)):
        if L2_ids[j] == -1:  # If still unmatched
            L2_ids[j] = next_id
            next_id += 1

    return L1_ids, L2_ids  # Return two separate lists

def compare_bboxes(gt_bboxes, det_bboxes, p=0.5, iou=0.8):
    print(iou)
    L1_ids, L2_ids = assign_bbox_ids(gt_bboxes, det_bboxes, iou=iou)
    r = rbo(L1_ids, L2_ids, p=p)
    return r['res']
  
