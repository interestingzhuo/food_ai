import cv2
import numpy as np
import torch
import os
import torch.nn as nn
from ultralytics import SAM
from models.fastsam import FastSAM, FastSAMPrompt 
from models.yolov7.models.common import Conv
from models.yolov7.utils.general import non_max_suppression, scale_coords
from models.yolov7.models.yolo import Model
from models.FoodSAM.FoodSAM_tools.predict_semantic_mask import semantic_predict
from models.FoodSAM.FoodSAM_tools.enhance_semantic_masks import enhance_masks
from models.FoodSAM.semantic import write_masks_to_folder
from models.FoodSAM.segment_anything import SamAutomaticMaskGenerator,sam_model_registry
import sys
class FoodSegmentation():

    def __init__(self, config, device):
        self.device = device
        self.model = SAM('./sam_b.pt')
       
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
    
class FoodSegmentationFoodSAM():
    def __init__(self, config, device):
        self.device = device
        #self.model = FastSAM(config["segmentation"]["seg_checkpoint"])
        self.config = config["segmentation"]
        self.model = sam_model_registry['vit_b'](checkpoint=self.config["seg_checkpoint"])
        self.output_mode = "binary_mask"
        _ = self.model.to(device=self.device)
        #amg_kwargs = get_amg_kwargs(args)
        self.generator = SamAutomaticMaskGenerator(self.model, output_mode=self.output_mode)#, **amg_kwargs)
    def __call__(self, img, args):
        masks = self.generator.generate(img)
        output = os.path.join(self.config['seg_output'],args.image_name)
        write_masks_to_folder(masks, output)
        result = semantic_predict(
            self.config['seg_config'],
            self.config['aug_test'], 
            self.config['semantic_checkpoint'], 
            self.config['seg_output'], 
            self.config['semantic_color_list'], 
            args.image_path)
        enhanced_mask = enhance_masks(
            args.image_name,
            self.config['seg_output'], 
            args.image_path,
            self.config['category_txt_list'],
            self.config['semantic_color_list'],
            num_class=104, 
            area_thr=0, 
            ratio_thr=0.5, 
            top_k=80)
        return result,enhanced_mask
