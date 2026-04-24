import torch
import numpy as np
from typing import Union
from .prior_depth_anything import PriorDepthAnything


class DAP():
    def __init__(self,
                 device,
                 mde_ckpt='/mnt/proj/0_Checkpoints/20_DepthAnythingv2/depth_anything_v2_vitb.pth',
                 dap_ckpt='/mnt/proj/0_Checkpoints/22_priorDA/prior_depth_anything_vitb.pth'):
        self.device = device
        fmde_dir = mde_ckpt[:str.rfind(mde_ckpt,'/')]
        ckpt_dir = dap_ckpt[:str.rfind(dap_ckpt,'/')]
        frozen_model_size = str.split(mde_ckpt,'/')[-1][-8:-4]
        conditioned_model_size = str.split(dap_ckpt,'/')[-1][-8:-4]
        self.model = PriorDepthAnything(device=device,
                             fmde_dir=fmde_dir,
                             cmde_dir=fmde_dir,
                             ckpt_dir=ckpt_dir,
                             frozen_model_size=frozen_model_size,
                             conditioned_model_size=conditioned_model_size)
    
    def to(self,device):
        self.device = device
        self.model.device = device
        self.model.to(self.device)
        self.model.sampler.device = self.device
        self.model.completion.set_device(self.device)
    
    @torch.no_grad()
    def __call__(self, 
                 image: Union[str, torch.Tensor, np.ndarray] = None, 
                 prior: Union[str, torch.Tensor, np.ndarray] = None,
                 visualize = False):
        output = self.model.infer_one_sample(image=image, prior=prior, visualize=visualize)
        return output
        
if __name__ == '__main__':
    from PIL import Image
    from utils import get_intrins_from_fov,dpt2xyz,visual_pcd
        
    tool = DAP('cuda')
    image_path = 'assets/sample-2/rgb.jpg'
    prior_path = 'assets/sample-2/prior_depth.png'
    image = np.array(Image.open(image_path))
    prior = np.array(Image.open(prior_path))/1000
    depth = tool(image,prior).cpu().numpy()
    intrinsic = get_intrins_from_fov(60,image.shape[0],image.shape[1])
    xyz = dpt2xyz(depth,intrinsic)
    visual_pcd(xyz)
    
    
