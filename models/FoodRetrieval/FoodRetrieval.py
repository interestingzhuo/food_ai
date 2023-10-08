import sys
sys.path.insert(0,'./models/FoodRetrieval')
import torchvision.transforms as transforms
from .dataset import *
import torch
from .utils import *
import time
import faiss
import numpy as np
from tqdm import tqdm
from .deit import Network
from PIL import Image
class FoodRetrieval():
    def __init__(self, config, device):
        self.device = device
        self.model = Network()
        checkpoint = torch.load(config["retrieval"]["retrieval_checkpoint"])
        new_checkpoint = {}

        for k,v in checkpoint['state_dict'].items():
            k = k.replace('module.model', 'model')
            new_checkpoint[k] = v
        self.model.load_state_dict(new_checkpoint)
        self.model.to(self.device)
        self.model.eval()

        self.dim = config["retrieval"]["dim"]
        self.batch_size = config["retrieval"]["batch_size"]
        self.gallery_list = config["retrieval"]["gallery_list"]
        self.top_k = config["retrieval"]["top_k"]

        self.f_norm = normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        self.normal_transform = []
        self.imsize = 224
        self.normal_transform.extend([transforms.Resize(config["retrieval"]["input_shape"]), transforms.CenterCrop(self.imsize)])
        self.normal_transform.extend([transforms.ToTensor(), normalize])
        self.normal_transform = transforms.Compose(self.normal_transform)
        self.labels = []
        self.conduct_gallery(config)
        
        print('#images in database:', self.gpu_index.ntotal)

    def conduct_gallery(self,config):
        

        test_dataset_g = ImagesForTest(self.gallery_list, transform=self.normal_transform)  
        test_loader_g = torch.utils.data.DataLoader(
            test_dataset_g, batch_size=self.batch_size, shuffle=False,
            num_workers=8, pin_memory=True, sampler=None,
        )
    

           
        cpu_index = faiss.IndexFlatL2(self.dim)
        self.gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
        st = time.time()
        for (x, label) in tqdm(test_loader_g):
            x = x.to(self.device)
            x = x.contiguous()
            with torch.no_grad():
                vec = self.model(x)
            self.gpu_index.add(vec.cpu().numpy())
            self.labels.extend(label)
        end = time.time()
        print("Conduct Time:", end-st)


    def retrieval(self, img):
        img = Image.fromarray((img * 1).astype(np.uint8)).convert('RGB')
        #image = Image.new("RGB", np.shape(img))
        #img = image.putdata(img)
        img = self.normal_transform(img)
        img = img.to(self.device)
        img = img.contiguous().unsqueeze(0)
        with torch.no_grad():
            vec = self.model(img)
        _, I = self.gpu_index.search(vec.cpu().numpy(), self.top_k)
        name = self.labels[I[0][0]]
        return name



   


    

