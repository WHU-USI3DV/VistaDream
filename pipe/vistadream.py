'''
render using frames in GS
inpaint with fooocus
'''
import torch
from PIL import Image
from ops.utils.utils import *
from ops.utils.visual_check import Check
from pipe.stages.prepare import Prepare_Phase
from pipe.stages.loadtools import Load_Tools_Phase
from pipe.stages.scaffold import Scaffold_Phase
from pipe.stages.warpextend import WarpExtend_Phase
from pipe.stages.xmcsrefine import MCS_Phase

        
class Pipeline():
    def __init__(self,cfg) -> None:
        self.cfg = cfg
        self.checkor = Check()
        self.prepare = Prepare_Phase(self.cfg)()

    def __call__(self):
        rgb_fn = self.cfg.scene.input.rgb
        temp_dir = rgb_fn[:str.rfind(rgb_fn,'/')]
        # prepare stage
        if os.path.exists(f'{temp_dir}/scene.coarse.pth'):
            self.cfg.tools.preload = ['mcs']
            self.tools = Load_Tools_Phase(self.cfg,mcs=True)
            scene = torch.load(f'{temp_dir}/scene.coarse.pth',weights_only=False)
        else:
            self.cfg.tools.preload = ['llava','rgb_inpaint','dpt_inpaint','mcs']
            self.tools = Load_Tools_Phase(self.cfg,mcs=True)
            # scaffold stage
            rgb = Image.open(rgb_fn)
            scene = Scaffold_Phase(self.cfg,self.tools)(rgb)
            # coarse stage
            scene = WarpExtend_Phase(self.cfg,self.tools)(scene)
            torch.save(scene,f'{temp_dir}/scene.coarse.pth')
            self.checkor._render_video(scene,save_dir=f'{temp_dir}/coarse.')
        # refine stage
        scene = MCS_Phase(self.cfg,self.tools,scene,device='cuda')()
        torch.save(scene,f'{temp_dir}/scene.refine.pth')
        self.checkor._render_video(scene,save_dir=f'{temp_dir}/refine.')
