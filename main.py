import sys
sys.path.insert(0,'/mnt/data/chaoran/food_ai_main/model/yolov7')
import argparse
import copy
import os
import cv2
import mmcv
from PIL import Image
import torch
from omegaconf import OmegaConf
from tqdm import tqdm



from modules.segmentation.segmentation import FoodSegmentation
from modules.detection.detector import FoodDetector
from modules.classification.classification import FoodClassifier
#from modules.depth.depth import Depth
#from modules.volume.volume import VolumeEstimation
#from modules.calories.calories import Calories

from profile.time import log_time_consuming, time_synchronized
from visualization.draw_bbox import draw_bbox

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path",
                        type=str,
                        default="/mnt/data/chaoran/food_ai_main/data/10030006.jpg",
                        help="iut image path")
    parser.add_argument("--output_path",
                        type=str,
                        default="output/",
                        help="output path")
    args = parser.parse_args()

    image_name = os.path.splitext(os.path.basename(args.image_path))[0]
    config = OmegaConf.load("video_mocap.yml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # test_image = mmcv.imread(args.image_path)
    test_image = [cv2.imread(args.image_path)]


    segment = FoodSegmentation(config, device)
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
            # segmentation
            t2 = time_synchronized()
            segmentation_result = segment(image)
            # recognition
            t3 = time_synchronized()
            classifier_result = classifier(image, detection_result)
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


            torch.cuda.empty_cache()

            #log_time_consuming(time_consuming)
            img = draw_bbox(image, detection_result,classifier_result)
            cv2.imwrite(f'{classifier_result[0]}.jpg',img)
            # draw_segmentation(test_image, segmentation_results)
