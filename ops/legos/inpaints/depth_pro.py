import os,sys
currect = os.getcwd()
reference = f'{currect}/tools/DepthPro'
sys.path.append(reference)

from command_pro_dpt import apple_pro_depth

class Depth_Pro_Tool(apple_pro_depth):
    def __init__(self, device='cuda', ckpt='/mnt/proj/SOTAs/ml-depth-pro-main/checkpoints/depth_pro.pt'):
        super().__init__(device, ckpt)

    def __call__(self, image, f_px=None):
        return super().__call__(image, f_px)