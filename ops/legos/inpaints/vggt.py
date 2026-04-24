import os,sys
currect = os.getcwd()
reference = f'{currect}/tools/VGGT'
sys.path.append(reference)

from vggt_command import VGGT_MVR

class VGGT_Tool():
    '''
    input: numpy rgb BHW3, intrinsic = None/3*3 numpy
    output: numpy dpt BHW, intrinsic = 3*3 numpy
    '''
    def __init__(self,device='cpu',ckpt=f'/mnt/proj/0_Checkpoints/12_VGGT/model.pt'):
        self.device = device
        self.model = VGGT_MVR(device,ckpt)
    
    def to(self,device):
        self.device = device
        self.model.to(device)
        
    def __call__(self, images):
        '''
        input images should be [HW3] in numpy 0-1/0-255
        '''
        depths,dptconfs,intrinsics,extrinsics = self.model(images)
        return depths,dptconfs,intrinsics,extrinsics