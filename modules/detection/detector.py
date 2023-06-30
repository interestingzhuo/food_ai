import cv2
import numpy as np
import torch
import torch.nn as nn

from models.yolov7.common import Conv
from models.yolov7.general import non_max_suppression, scale_coords
from models.yolov7.yolo import Model


class FoodDetector():

    def __init__(self, config, device):
        self.device = device
        self.half = True

        self.det_model = Model(cfg=config["detector"]["det_config"])
        self.det_model.fuse()

        self.det_model.load_model(config["detector"]["det_checkpoint"], self.device)
        self.det_model.to(self.device)
        self.det_model.eval()

        if self.half:
            self.det_model.half()

        for m in self.det_model.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            elif type(m) is nn.Upsample:
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

        self.test_shape = (640, 640)
        self.stride = int(self.det_model.stride.max())
        self.conf_thres = 0.50
        self.iou_thres = 0.45
        self.agnostic_nms = False

        self.buffer_size_threshold = 3
        self.buffer_size = 0
        self.buffer_bbox = None

    def preprocessing(self, img):
        self.origin_img_shape = img.shape

        # Resize and pad image while meeting stride-multiple constraints
        self.origin_shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(self.test_shape[0] / self.origin_shape[0], self.test_shape[1] / self.origin_shape[1])

        # Compute padding
        new_unpad = int(round(self.origin_shape[1] * r)), int(round(self.origin_shape[0] * r))
        dw, dh = self.test_shape[1] - new_unpad[0], self.test_shape[0] - new_unpad[1]  # wh padding
        # minimum rectangle
        dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if self.origin_shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])  # add border

        # BGR2RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)[np.newaxis, ...]
        img = torch.from_numpy(np.ascontiguousarray(img)).to(torch.float32).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        self.test_shape = img.shape[2:]  # height, width

        return img

    def postprocessing(self, pred):
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=0, agnostic=self.agnostic_nms)

        max_area = 0
        bbox = None
        for det in pred:
            # Rescale boxes from img_size to origin img_size
            det[:, :4] = scale_coords(self.test_shape, det[:, :4], self.origin_shape).round()
            for *xyxy, _, _ in reversed(det):
                x1, y1, x2, y2 = [xy.cpu().numpy() for xy in xyxy]
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    bbox = np.array([[x1, y1, x2 - x1, y2 - y1]])

        if bbox is not None:
            self.buffer_bbox = bbox
            self.buffer_size = 0
        else:
            if self.buffer_size < self.buffer_size_threshold:
                bbox = self.buffer_bbox
                self.buffer_size += 1

        return bbox

    def __call__(self, img):
        img = self.preprocessing(img)
        pred = self.det_model(img)[0]
        results = self.postprocessing(pred)
        return results
