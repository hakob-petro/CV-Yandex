import os

import torch
from ssd_v2 import SSD300v2
from ssd_utils import BBoxUtility

# Using labels
VOC_CLASSES = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']

NUM_CLASSES = len(VOC_CLASSES) + 1

bbox_util = BBoxUtility(NUM_CLASSES)

# SSD model
input_shape = (300, 300, 3)
model = SSD300v2(input_shape, num_classes=NUM_CLASSES)

model.eval()
model.load_state_dict(torch.load('SSD300v2_torch.pth'))
