import torchvision.transforms as transforms
from dataset import *
import torch
from utils import *
import time
from imageretrievalnet import *
import time
import faiss
from tqdm import tqdm


class FoodRetrieval():
    def __init__(self, config, device):
        self.device = device
        self.model = image_net(config["retrieval"]["retrieval_config"])

        self.model.load_model(config["retrieval"]["retrieval_checkpoint"], self.device)
        self.model.to(self.device)
        self.model.eval()
        self.dim = config["retrieval"]["dim"]
        self.batch_size = config["retrieval"]["batch_size"]
        self.gallery_list = config["retrieval"]["gallery_list"]
        self.conduct_gallery(config)
        self.top_k = config["retrieval"]["top_k"]
        print('#images in database:', self.gpu_index.ntotal)
        self.f_norm = normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        self.normal_transform = []
        self.normal_transform.extend([transforms.Resize(256), transforms.CenterCrop(imsize)])
        self.normal_transform.extend([transforms.ToTensor(), normalize])
        self.normal_transform = transforms.Compose(self.normal_transform)
        self.names = []

    def conduct_gallery(self,config):
        

        test_dataset_g = ImagesForTest(self.gallery_list, transform=transform)  
        test_loader_g = torch.utils.data.DataLoader(
            test_dataset_g, batch_size=self.batch_size, shuffle=False,
            num_workers=8, pin_memory=True, sampler=None,
        )
    

           
        cpu_index = faiss.IndexFlatL2(self.dim)
        self.gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
        st = time.time()
        for (x, name) in tqdm(test_loader_g):
            x = x.to(self.device)
            x = x.contiguous()
            with torch.no_grad():
                vec, _ = self.model(x)
            self.gpu_index.add(vec.cpu().numpy())
            self.names += [name]


    def retrieval(self, img):
        img = self.normal_transform(img)
        img = img.to(self.device)
        img = img.contiguous()
        with torch.no_grad():
            vec, _ = self.model(img)
        _, I = self.gpu_index.search(vec, self.top_k)
        name = self.names[I[0]]
        return name



   


    

