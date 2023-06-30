import cv2
import numpy as np
import torch
import torch.nn as nn

from models.yolov7.common import Conv
from models.yolov7.general import non_max_suppression, scale_coords
from models.yolov7.yolo import Model


class FoodSegmentation():

    def __init__(self, config, device):
        self.device = device
       
    def __call__(self, img):
        
        return results
