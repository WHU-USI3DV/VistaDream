import numpy as np
from tools.DAAnyPrior.DAP_command import DAP

class DepthAnyPrior_Tool():
    def __init__(self,
                 device='cuda',
                 mde_ckpt='/mnt/proj/0_Checkpoints/20_DepthAnythingv2/depth_anything_v2_vitb.pth',
                 dap_ckpt='/mnt/proj/0_Checkpoints/22_priorDA/prior_depth_anything_vitb.pth'):
        self.device = device
        self.tool = DAP(device,mde_ckpt,dap_ckpt)
    
    def to(self, device):
        self.device = device
        self.tool.to(self.device)
    
    def __call__(self, inpaint_image, rendered_depth, inpaint_mask):
        # inpaint_image must in range of 0-255
        if np.amax(inpaint_image) < 1.5:
            inpaint_image = inpaint_image * 255.
        # inpaint area should be 0
        rendered_depth[inpaint_mask] = 0.
        # get inpainted depth
        inpaint_depth = self.tool(image = inpaint_image,
                                  prior = rendered_depth,
                                  visualize=False)
        inpaint_depth = inpaint_depth.squeeze().cpu().numpy()
        return inpaint_depth
        
