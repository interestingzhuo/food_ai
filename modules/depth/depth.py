
from model.depth import __models__
import torch

class Depth():

    def __init__(self, config, device):
        self.device = device
        self.model = __models__[config['depth']['model']](config["depth"]["maxdisp"]).to(self.device)
        state_dict = torch.load(config['depth']['ckpt'])
        self.model.load_state_dict(state_dict['model'])
        self.model.eval()




    def depth(self, disparity_map, calibration):
        pass;

        


    def __call__(self, images, calibration):
        imgL, imgR = images
        imgL = imgL.to(self.device)
        imgR = imgR.to(self.device)

        disp_ests, _, _ = self.model(imgL, imgR)
        depth_result = self.depth(disp_ests, calibration)
        return depth_result
