import sys
import argparse
import copy
import os
import cv2
#import mmcv
from PIL import Image
import torch
from numpy import random
from omegaconf import OmegaConf
from tqdm import tqdm
from util.profile.time import log_time_consuming, time_synchronized
from util.visualization.draw_bbox import draw_bbox
sys.path.insert(0,'./models/yolov7')
print(sys.path)
from models.yolov7.utils.plots import plot_one_box
from modules.segmentation.segmentation import FoodSegmentation,FoodSegmentationFoodSAM,FoodSegmentationFastSAM
from modules.detection.detector import FoodDetector
from modules.classification.classification import FoodClassifier
#from modules.depth.depth import Depth
#from modules.volume.volume import VolumeEstimation
#from modules.calories.calories import Calories

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path",
                        type=str,
                        default="./data/01.jpg",
                        help="iut image path")
    parser.add_argument("--output_path",
                        type=str,
                        default="output/",
                        help="output path")
    args = parser.parse_args()

    image_name = os.path.splitext(os.path.basename(args.image_path))[0]
    args.image_name = image_name
    config = OmegaConf.load("video_mocap.yml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # test_image = mmcv.imread(args.image_path)
    test_image = [cv2.imread(args.image_path)]


    #segment = FoodSegmentation(config, device)
    segment = FoodSegmentationFoodSAM(config, device)
    detector = FoodDetector(config, device)
    classifier = FoodClassifier(config, device)
    #depth_estimator = Depth(config, device)
    #estimation = VolumeEstimation(config, device)#立体匹配
    #calories = Calories(config, device)#建数据库

    detection_results = []
    segmentation_results = []
    classifier_results = []
    depth_results = []
    volume_results = []
    calories_results = []


    time_consuming = {
        "detection": [],
        "segmentation": [],
        "recognition": [],
        "Depth": [],
        "Volume": [],
        "Calories": [],
        "all": [],
    }

    with torch.no_grad():
        #for idx, image, calibration in tqdm(enumerate(test_image), total=len(test_image)): 会报错#
        for idx, image in tqdm(enumerate(test_image), total=len(test_image)): #, calibration
            # detection
            t1 = time_synchronized()
            detection_result = detector(image)
            #print('detection_result',detection_result)
            #break
            # segmentation
            t2 = time_synchronized()
            segmentation_result,enhanced_mask = segment(image, args)
            #rint('segmentation_result',segmentation_result)
            #print('enhanced_mask',enhanced_mask)
            # recognition
            t3 = time_synchronized()
            classifier_result = classifier(image, detection_result)
            #print('classifier_result',classifier_result)
            # Depth Estimation
            t4 = time_synchronized()
            #depth_result = depth_estimator(image, calibration)
            # Volume Estimation
            #t5 = time_synchronized()
            #volume_result = estimation(depth_result, segmentation_result, calibration)
            # Calories Estimation
            #t6 = time_synchronized()
            #calories_result = calories(volume_result, segmentation_result, detection_result)
            #t7 = time_synchronized()
           

            time_consuming["detection"].append(t2 - t1)
            time_consuming["segmentation"].append(t3 - t2)
            time_consuming["recognition"].append(t4 - t3)
            #time_consuming["Depth"].append(t5 - t4)
            #time_consuming["Volume"].append(t6 - t5)
            #time_consuming["Calories"].append(t7 - t6)
            time_consuming["all"].append(t4 - t1)

            #detection_results.append(detection_result)
            segmentation_results.append(segmentation_result)
            classifier_results.append(classifier_result)
            #depth_results.append(depth_result)
            #volume_results.append(volume_result)
            #calories_results.append(calories_result)
            result_img = cv2.imread(f'./runs/segment/output/{image_name}/enhance_vis.png')
            colors = [random.randint(0, 255) for _ in range(len(classifier_result))]
            torch.cuda.empty_cache()
            for index,bbox in enumerate(detection_result):
                #log_time_consuming(time_consuming)
                #img = draw_bbox(image, detection_result,classifier_result)
                plot_one_box(bbox,result_img, label=classifier_result[index],color=colors[index], line_thickness=2)
                cv2.imwrite(f'{classifier_result[0]}.jpg',result_img)
            #draw_segmentation(image, segmentation_results)
