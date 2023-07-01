from models.FoodRetrieval.utils import generate_patch_image
from model.FoodRetrieval.FoodRetrieval import FoodRetrieval



class FoodClassifier():

    def __init__(self, config, device):
        self.device = device
        self.retrieval = FoodRetrieval(config, device)
    

   
    def __call__(self, image, detection_result):
        results = []
        for bbox in detection_result:
            img = generate_patch_image(image, bbox)
            name = self.retrieval.retrieval(img)
            results += [name]
        return results

        

   