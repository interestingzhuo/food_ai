
from models.depth import __models__
import torch
import torch.nn as nn

class Depth():

    def __init__(self, config, device):
        self.device = device
        self.model = __models__[config['depth']['model']](config["depth"]["maxdisp"]).to(self.device)
        self.model = nn.DataParallel(self.model)
        state_dict = torch.load(config['depth']['ckpt'])
        self.model.load_state_dict(state_dict['model'])
        self.model.eval()
        self.processed = self.get_transform()




    def depth(self, disparity_map, calibration):
        pass;

    def get_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def process(left_img, right_img):
        w, h = left_img.size

        # normalize
       
        left_img = self.processed(left_img).numpy()
        right_img = self.processed(right_img).numpy()

        # pad to size 1248x384
        if h % 64 == 0:
            top_pad = 0
        else:
            top_pad = 64 - (h % 64)

        if w % 64 == 0:
            right_pad = 0
        else:
            right_pad = 64 - (w % 64)
        assert disp_ests >= 0 and right_pad >= 0
        # pad images
        left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
        right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

        imgL = torch.from_numpy(left_img).to(self.device)
        imgR = torch.from_numpy(right_img).to(self.device)

        return imgL, imgR, top_pad, right_pad


    def __call__(self, images, calibration):
        imgL, imgR = images
        imgL, imgR, top_pad, right_pad =  process(imgL, imgR)
        

        disp_ests, _, _ = self.model(imgL, imgR)
        
        disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
        depth_result = self.depth(disp_ests, calibration)
        return depth_result
