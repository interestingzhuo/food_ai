from models.FoodRetrieval.utils import generate_patch_image
from models.FoodRetrieval.FoodRetrieval import FoodRetrieval



class FoodClassifier():

    def __init__(self, config, device):
        self.device = device
        self.retrieval = FoodRetrieval(config, device)
        h = w = config["retrieval"]['input_shape']
        self.input_shape = (h, w)
    

   
    def __call__(self, image, detection_result):
        results = []
        for bbox in detection_result:
            img = generate_patch_image(image, bbox, 1.0, 0.0, False, self.input_shape)
            name = self.retrieval.retrieval(img)
            results += [name]
        return results

        

   