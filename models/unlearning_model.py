import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F

def compute_KLdiv(l_1, l_2):
    dist_1 = F.log_softmax(l_1, dim=1)
    dist_2 = F.softmax(l_2, dim=1)
    return F.kl_div(dist_1, dist_2, reduction='batchmean')
    
    

class SiamUL(nn.Module):
    
    def __init__(self, backbone, num_classes, hidden_dim=64):
        super().__init__()
        
        self.backbone = backbone
        # build a 2-layer projector
        self.projector = nn.Sequential(nn.Linear(num_classes, hidden_dim, bias=False),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(hidden_dim, num_classes, bias=False)
                                        )
        
    def forward(self, x1, x2):
        z1 = self.backbone(x1) # NxC
        z2 = self.backbone(x2) # NxC

        p1 = self.projector(z1) # NxC
        p2 = self.projector(z2) # NxC

        return p1, p2, z1, z2, z1.detach(), z2.detach()    
        
class LossKVKC(nn.Module):
    def __init__(self, lam=1, forget=True):
        super().__init__()
        self.lam = lam
        self.L_1 = nn.CosineSimilarity(dim=1)
        self.L_2 = nn.CrossEntropyLoss()
        self.forget = forget
        
    def knowledge_vaporization(self, p1, p2, z1, z2, 
                                zd1, zd2, y1, y2):
        cos_similarity = 0.5 * (self.L_1(p1, zd2).mean() + self.L_1(p2, zd1).mean()) 
        sce = 0.5 * (self.L_2(z1, y1) + self.L_2(z2, y2))
        return cos_similarity + self.lam*sce, cos_similarity, sce
    
    def knowledge_concentration(self, p1, p2, z1, z2, 
                                zd1, zd2, y1, y2):
        cos_similarity = 0.5 * (self.L_1(p1, zd2).mean() + self.L_1(p2, zd1).mean()) 
        sce = 0.5 * (self.L_2(z1, y1) + self.L_2(z2, y2))
        return -1 * cos_similarity + self.lam*sce, cos_similarity, sce
    
    def forward(self, p1, p2, z1, z2, 
                zd1, zd2, y1, y2):
        if self.forget:
            return self.knowledge_vaporization(p1, p2, z1, z2, zd1, zd2, y1, y2)
        else:
            return self.knowledge_concentration(p1, p2, z1, z2, zd1, zd2, y1, y2)



class RandomResponse(nn.Module):
    def __init__(self, ps:list, num_classes:int):
        super().__init__()
        self.ps = ps
        self.k = num_classes

    def response(self, ele):
        idx = int(ele)
        tmp = np.random.uniform()
        if tmp > self.ps[idx]:
            return ele
        else:
            while True:
                resp = np.random.choice(self.k)
                if resp != ele:
                    return resp

    def forward(self, arr):
        l = arr.shape[0]
        re_arr = torch.zeros_like(arr)
        for i in range(l):
            re_arr[i] = self.response(arr[i])
        return re_arr
        