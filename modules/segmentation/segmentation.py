import cv2
import numpy as np
import torch
import torch.nn as nn

from models.yolov7.models.common import Conv
from models.yolov7.utils.general import non_max_suppression, scale_coords
from models.yolov7.models.yolo import Model
from ultralytics import SAM
from models.fastsam import FastSAM, FastSAMPrompt 


class FoodSegmentation():

    def __init__(self, config, device):
        self.device = device
        self.model = SAM('/mnt/data2/ai_food_pth/sam_b.pt')
       
    def __call__(self, img):
        results = self.model(img)
        return results

class FoodSegmentationFastSAM():

    def __init__(self, config, device):
        self.device = device
        self.model = FastSAM(config["segmentation"]["seg_checkpoint"])

       
    def __call__(self, img):
        results = self.model(
            img,
            device=self.device,
            retina_masks=True,
            imgsz=1024,
            conf=0.4,
            iou=0.9, 
        )
        return results
