import torchvision.transforms as transforms
from dataset import *
import torch
from utils import *
import time
from imageretrievalnet import *
import time
import faiss
from tqdm import tqdm
from deit import Network

class FoodRetrieval():
    def __init__(self, config, device):
        self.device = device
        self.model = Network()
        checkpoint = torch.load(config["retrieval"]["retrieval_checkpoint"])

        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.dim = config["retrieval"]["dim"]
        self.batch_size = config["retrieval"]["batch_size"]
        self.gallery_list = config["retrieval"]["gallery_list"]
        self.top_k = config["retrieval"]["top_k"]
        
        self.conduct_gallery(config)
        
        print('#images in database:', self.gpu_index.ntotal)
        self.f_norm = normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        self.normal_transform = []
        self.imsize = 224
        self.normal_transform.extend([transforms.Resize(256), transforms.CenterCrop(self.imsize)])
        self.normal_transform.extend([transforms.ToTensor(), normalize])
        self.normal_transform = transforms.Compose(self.normal_transform)
        self.labels = []

    def conduct_gallery(self,config):
        

        test_dataset_g = ImagesForTest(self.gallery_list, transform=transform)  
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
                vec, _ = self.model(x)
            self.gpu_index.add(vec.cpu().numpy())
            self.labels += [label]
        end = time.time()
        print("Conduct Time:", end-st)


    def retrieval(self, img):
        img = self.normal_transform(img)
        img = img.to(self.device)
        img = img.contiguous()
        with torch.no_grad():
            vec, _ = self.model(img)
        _, I = self.gpu_index.search(vec, self.top_k)
        name = self.labels[I[0]]
        return name



   


    

