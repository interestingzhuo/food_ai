
from models.volume import volume

class VolumeEstimation():

    def __init__(self, config, device):
        self.device = device
        self.volume = volume()

    def __call__(self, depth_map, segmentation_result, calibration):
        label_mask = segmentation_result
        attitude = None
        calibration = calibration
        depth_map = depth_map
        area_volume_list =  self.volume(depth_map, calibration, attitude, label_mask)
        return area_volume_list
