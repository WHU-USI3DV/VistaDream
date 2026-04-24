'''
render using frames in GS
inpaint with fooocus
'''
import torch
import numpy as np
from PIL import Image
from ops.utils.utils import *
from ops.utils.visual_check import Check
from ops.legos.inpaints.vggt import VGGT_Tool
from pipe.stages.prepare import Prepare_Phase
from pipe.stages.scaffold import Scaffold_Phase
from pipe.stages.loadtools import Load_Tools_Phase
from pipe.stages.warpextend import WarpExtend_Phase
from pipe.stages.xmcsrefine import MCS_Phase
from ops.gs.basic import Gaussian_Scene,Frame
from ops.gs.train import GS_Train_Tool

        
class Pipeline_Sparse():
    def __init__(self,cfg) -> None:
        cfg.scene.traj.traj_type = 'interp'
        self.cfg = cfg
        self.checkor = Check()
        self.prepare = Prepare_Phase(self.cfg)

    def _sparse_scaffold_(self,rgbs_fn):
        ''' We need to load vggt firstly. '''
        rgbs = [np.array(Image.open(rgb_fn))[...,0:3] for rgb_fn in rgbs_fn]
        # load VGGT
        vggt = VGGT_Tool(device='cuda',ckpt=self.cfg.model.dpt.vggt.ckpt)
        # estimate
        dpts,confs,intrinsics,extrinsics = vggt(images=rgbs)
        # describe
        scaffold_tool = Scaffold_Phase(self.cfg,self.tools)
        scaffold_tool._lvm_prompt_(rgbs[0])
        prompt = scaffold_tool.prompt
        # build scene
        scene = Gaussian_Scene(self.cfg)
        # add frame by frame       
        for i in range(len(rgbs)):
            rgb = rgbs[i]
            dpt = dpts[i]
            intrinsic = intrinsics[i]
            extrinsic = extrinsics[i]
            H,W = rgb.shape[0:2]
            frame = Frame(H,W,intrinsic=intrinsic,extrinsic=extrinsic)
            # get inpaint mask
            if len(scene.frames) > 0:
                frame = scene._render_for_inpaint(frame)
            else:
                frame.inpaint = np.ones_like(dpt) > .5
            # set others
            frame.rgb = rgb
            frame.dpt = dpt
            frame.prompt = prompt
            # process sky
            if len(scene.frames)<0: self.tools.dpt_inpaint.sky._set_sky_depth_(frame)
            frame = self.tools.dpt_inpaint.sky(frame)
            # add to scene
            scene._add_trainable_frame(frame)
            scene._require_grad_(True)
            GS_Train_Tool(scene,iters=self.cfg.scene.gaussian.opt_iters_per_frame)(scene.frames)
        # get dense trajectory
        scene = scaffold_tool._generate_traj(scene)
        return scene

    def __call__(self):
        rgbs_fn = self.cfg.scene.input.rgbs
        temp_dir = rgbs_fn[0][:str.rfind(rgbs_fn[0],'/')]
        # prepare stage
        for rgb_fn in rgbs_fn:
            self.prepare._resize_input(rgb_fn)
        # build coarse scene
        if os.path.exists(f'{temp_dir}/scene.coarse.pth'):
            self.cfg.tools.preload = ['mcs']
            self.tools = Load_Tools_Phase(self.cfg)
            scene = torch.load(f'{temp_dir}/scene.coarse.pth')
        else:
            self.cfg.tools.preload = ['llava','rgb_inpaint','dpt_inpaint','mcs']
            self.tools = Load_Tools_Phase(self.cfg)
            # scaffold stage
            scene = self._sparse_scaffold_(rgbs_fn)
            torch.cuda.empty_cache()
            # coarse stage
            scene = WarpExtend_Phase(self.cfg,self.tools)(scene)
            torch.save(scene,f'{temp_dir}/scene.coarse.pth')
            self.checkor._render_video(scene,save_dir=f'{temp_dir}/coarse.')
        # refine stage
        scene = MCS_Phase(self.cfg,self.tools,scene,device='cuda')()
        torch.save(scene,f'{temp_dir}/scene.refine.pth')
        self.checkor._render_video(scene,save_dir=f'{temp_dir}/refine.')


    
    
    