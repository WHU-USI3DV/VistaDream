"""MoGe-3 scaffold depth and LingBot-Depth novel-view completion."""
import torch
from tqdm import tqdm
from ops.utils.utils import *
from ops.gs.train import GS_Train_Tool
from ops.trajs import _generate_trajectory
from ops.gs.basic import Frame,Gaussian_Scene
# for SKY
from ops.legos.inpaints.sky import Sky_Tool
# For MDI
from ops.legos.inpaints.lingbot_depth import LingBotDepth_Tool
# For MDE
from ops.legos.inpaints.moge3 import MoGe3_Tool
from ops.legos.inpaints.connect import Smooth_Connect_Tool,Occlusion_Removal

class Reconstruct_Tool():
    def __init__(self,cfg,device='cpu') -> None:
        self.cfg = cfg
        self.device = device
        # load depth estimator
        self.recon_type = self.cfg.scene.reconstruction.type
        if self.recon_type == 'mdi':
            self._load_lingbot_()
        else:
            raise TypeError(f'Wrong reonstruction type: {self.recon_type}. Desired: mdi (MoGe-3 + LingBot-Depth)')

    def _load_moge3_(self):
        moge_cfg = self.cfg.model.dpt.get('moge3', {})
        self.moge3 = MoGe3_Tool(
            device=self.device,
            ckpt=moge_cfg.get('ckpt', 'Ruicheng/moge-3-vitl'),
            revision=moge_cfg.get('revision'),
            resolution_level=moge_cfg.get('resolution_level', 9),
            refine_steps=moge_cfg.get('refine_steps', 3),
            use_fp16=moge_cfg.get('use_fp16', True),
        )
        self.connector = Smooth_Connect_Tool()

    def _load_lingbot_(self):
        self._load_moge3_()
        lingbot_cfg = self.cfg.model.dpt.get('lingbot', {})
        self.lingbot = LingBotDepth_Tool(
            device=self.device,
            ckpt=lingbot_cfg.get(
                'ckpt',
                'robbyant/lingbot-depth-pretrain-vitl-14-v0.5',
            ),
            resolution_level=lingbot_cfg.get('resolution_level', 9),
            preserve_known=lingbot_cfg.get('preserve_known', True),
            offline=lingbot_cfg.get('offline', False),
            revision=lingbot_cfg.get('revision'),
        )


    # ------------- Trial-1: Use Monocular Depth Estimator + Connector for depth inpainting ------------------ #
    def _conduct_mde_(self,rgb,
                      intrinsic=None,
                      refer_dpt=None,
                      inpaint_msk=None,
                      sky=None):
        # conduct reconstruction
        print('Monocular_Depth[1/3] Move MoGe-3 to GPU...')
        self.moge3.to('cuda')
        print('Monocular_Depth[2/3] MoGe-3 estimation...')
        f_px = intrinsic[0,0] if intrinsic is not None else None
        metric_dpt,intrinsic = self.moge3(rgb,f_px=f_px)
        inpaint_msk = inpaint_msk & (~sky)
        metric_dpt[sky] = 0.
        if refer_dpt is not None:
            metric_dpt = self.connector._affine_dpt_to_GS(refer_dpt,metric_dpt,inpaint_msk)
        print('Monocular_Depth[3/3] Move MoGe-3 to CPU...')
        self.moge3.to(self.device)
        torch.cuda.empty_cache()
        return metric_dpt, intrinsic

    # ------------- Trial-2: Use Monocular Depth Inpaintor for depth inpainting ------------------ #
    def _conduct_mdi_(self, rgb,
                      intrinsic=None,
                      refer_dpt=None,
                      inpaint_msk=None,
                      sky=None):
        if refer_dpt is None:
            # use mde
            return self._conduct_mde_(rgb,intrinsic,refer_dpt,inpaint_msk,sky)
        else:
            print('Depth_Inpainting[1/3] Move LingBot-Depth to GPU...')
            self.lingbot.to('cuda')
            print('Depth_Inpainting[2/3] LingBot-Depth completion...')
            completion_mask = inpaint_msk | sky if sky is not None else inpaint_msk
            metric_dpt_inpaint = self.lingbot(inpaint_image=rgb,
                                              rendered_depth=refer_dpt,
                                              inpaint_mask=completion_mask)
            print('Depth_Inpainting[3/3] Move LingBot-Depth to CPU...')
            self.lingbot.to(self.device)
            torch.cuda.empty_cache()
        return metric_dpt_inpaint, intrinsic


class Depth_Inpaint_Tool():
    def __init__(self,cfg,device='cpu') -> None:
        self.cfg = cfg
        # for 3D estimation
        self.recon_type = self.cfg.scene.reconstruction.type
        self.recons_tool = Reconstruct_Tool(self.cfg, device)
        # for sky, edge and noise removal
        self.sky = Sky_Tool(cfg)
        self.remover = Occlusion_Removal(self.cfg.scene.reconstruction.keep_occlu)
        self.edge_filter_times = self.cfg.scene.reconstruction.edge_filter_times
        # for add frame to scene
        self.gs_opt_iterations = self.cfg.scene.gaussian.opt_iters_per_frame

    def _inpaint_by_mdi_(self,scene:Gaussian_Scene,frame:Frame):
        frame.sky = self.sky._segment_sky_(frame.rgb)
        depth,intrinsic = self.recons_tool._conduct_mdi_(frame.rgb,
                                                         frame.intrinsic,
                                                         frame.dpt,
                                                         frame.inpaint,
                                                         frame.sky)
        frame.dpt = depth
        frame.intrinsic = intrinsic
        # remove block area
        scene,frame = self.remover(scene,frame)
        # remove sky
        if len(scene.frames)<1: self.sky._set_sky_depth_(frame)
        frame = self.sky(frame)
        # remove edge areas
        edge_mask = edge_filter(depth,frame.sky,times=self.edge_filter_times)
        if frame.inpaint is None: frame.inpaint = np.ones_like(edge_mask)>.5
        frame.inpaint = (frame.inpaint) & (~edge_mask)
        # add frame to the scene
        scene._add_trainable_frame(frame,require_grad=True)
        scene = GS_Train_Tool(scene,iters=self.gs_opt_iterations)(target_frames=[frame],only_inpainted_area=True)
        return scene



    def __call__(self, scene, frame, gs_opt_iterations=None):
        self.gs_opt_iterations = gs_opt_iterations if gs_opt_iterations is not None \
                                        else self.cfg.scene.gaussian.opt_iters_per_frame
        if self.recon_type == 'mdi':
            scene = self._inpaint_by_mdi_(scene,frame)
        else:
            raise TypeError(f'Wrong reonstruction type: {self.recon_type}. Desired: mdi (MoGe-3 + LingBot-Depth)')
        return scene
