'''
Dust3R reconstrucion
GeoWizard Estimation
Smooth Projection
'''
import torch
from tqdm import tqdm
from ops.utils.utils import *
from ops.gs.train import GS_Train_Tool
from ops.trajs import _generate_trajectory
from ops.gs.basic import Frame,Gaussian_Scene
# for SKY
from ops.legos.inpaints.sky import Sky_Tool
# For MVR
from ops.legos.inpaints.vggt import VGGT_Tool
# For MDI
from ops.legos.inpaints.daanyprior import DepthAnyPrior_Tool
# For MDE
from ops.legos.inpaints.depth_pro import Depth_Pro_Tool
from ops.legos.inpaints.connect import Smooth_Connect_Tool,Occlusion_Removal

class Reconstruct_Tool():
    def __init__(self,cfg,device='cpu') -> None:
        self.cfg = cfg
        self.device = device
        # load depth estimator
        self.recon_type = self.cfg.scene.reconstruction.type
        if self.recon_type == 'mde':
            self._load_depth_pro_()
        elif self.recon_type == 'mdi':
            self._load_daprior_()
        elif self.recon_type == 'mvr':
            self._load_vggt_()
        else:
            raise TypeError(f'Wrong reonstruction type: {self.recon_type}. Desired: mde/mdi/mvr')
    
    def _load_depth_pro_(self):
        self.dpt_pro = Depth_Pro_Tool(device=self.device,ckpt=self.cfg.model.dpt.dpt_pro.ckpt)
        self.connector = Smooth_Connect_Tool()
    
    def _load_daprior_(self):
        self.dpt_pro = Depth_Pro_Tool(device=self.device,ckpt=self.cfg.model.dpt.dpt_pro.ckpt)
        self.daprior = DepthAnyPrior_Tool(device=self.device,mde_ckpt=self.cfg.model.dpt.daprior.mde,dap_ckpt=self.cfg.model.dpt.daprior.dap)
        
    def _load_vggt_(self):
        self.vggt = VGGT_Tool(device=self.device,ckpt=self.cfg.model.dpt.vggt.ckpt)
    
    # ------------- Trial-1: Use Monocular Depth Estimator + Connector for depth inpainting ------------------ #
    def _conduct_mde_(self,rgb, 
                      intrinsic=None, 
                      refer_dpt=None, 
                      inpaint_msk=None,
                      sky=None):
        # conduct reconstruction
        print('Pro_dpt[1/3] Move Pro_dpt.model to GPU...')
        self.dpt_pro.to('cuda')
        print('Pro_dpt[2/3] Pro_dpt Estimation...')
        f_px = intrinsic[0,0] if intrinsic is not None else None
        metric_dpt,intrinsic = self.dpt_pro(rgb,f_px=f_px)
        inpaint_msk = inpaint_msk & (~sky)
        metric_dpt[sky] = 0.
        if refer_dpt is not None:
            metric_dpt = self.connector._affine_dpt_to_GS(refer_dpt,metric_dpt,inpaint_msk)
        print('Pro_dpt[3/3] Move Pro_dpt.model to CPU...')
        self.dpt_pro.to(self.device)
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
            print('Depth_Inpainting[1/3] Move DAPrior.model to GPU...')
            self.daprior.to('cuda')
            # donnot affect the real depth  
            print('Depth_Inpainting[2/3] DAPrior Estimation...')
            metric_dpt_inpaint = self.daprior(inpaint_image=rgb,
                                              rendered_depth=refer_dpt,
                                              inpaint_mask=inpaint_msk)
            print('Depth_Inpainting[3/3] Move DAPrior.model to CPU...')
            self.daprior.to(self.device)
            torch.cuda.empty_cache()
        return metric_dpt_inpaint, intrinsic
    
    # ------------- Trial-3: Use Multiview Reconstruction for re-generation ------------------ #
    def _conduct_mvr_(self, rgbs, intrinsic=None):
        # here we will re-generate the scene
        # all rgbs should be the same size and intrinsic
        print('MV-Reconstruction[1/3] Move VGGT.model to GPU...')
        self.vggt.to('cuda')
        print('MV-Reconstruction[2/3] VGGT Estimation...')
        dpts,confs,intrinsic,extrinsics = self.vggt(images=rgbs)
        print('MV-Reconstruction[3/3] Move VGGT.model to CPU...')
        self.vggt.to(self.device)
        torch.cuda.empty_cache()
        return dpts,intrinsic,extrinsics

class Depth_Inpaint_Tool():
    def __init__(self,cfg,device='cpu') -> None:
        self.cfg = cfg
        # for 3D estimation
        self.recon_type = self.cfg.scene.reconstruction.type
        self.recons_tool = Reconstruct_Tool(self.cfg,device)
        # for sky, edge and noise removal
        self.sky = Sky_Tool(cfg)
        self.remover = Occlusion_Removal(self.cfg.scene.reconstruction.keep_occlu)
        self.edge_filter_times = self.cfg.scene.reconstruction.edge_filter_times
        # for add frame to scene
        self.gs_opt_iterations = self.cfg.scene.gaussian.opt_iters_per_frame
    
    def _inpaint_by_mde_(self,scene:Gaussian_Scene,frame:Frame):
        frame.sky = self.sky._segment_sky_(frame.rgb)
        depth,intrinsic = self.recons_tool._conduct_mde_(frame.rgb,
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
        scene = GS_Train_Tool(scene,iters=self.gs_opt_iterations)(target_frames=[frame])
        return scene
        
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

    def _inpaint_by_mvr_w_scaffold_(self,scene:Gaussian_Scene,frame:Frame):
        '''
        re-estimate depths + (shared intrinsic) of all rgbs
        re-generate inpaint masks
        re-training for all rgbs
        '''
        # estimate depths for all images
        frames = scene.frames + [frame]
        rgbs = [f.rgb for f in frames]
        dpts,intrinsics,extrinsics = self.recons_tool._conduct_mvr_(rgbs)
        # force
        if len(intrinsics)>1: # force the scaffold thing
            intrinsics[0][0, 0] = intrinsics[1][0, 0]
            intrinsics[0][1, 1] = intrinsics[1][1, 1]
            intrinsics[0][0,-1] = intrinsics[1][0,-1] - (dpts[1].shape[1]-dpts[0].shape[1])/2.
            intrinsics[0][1,-1] = intrinsics[1][1,-1] - (dpts[1].shape[0]-dpts[0].shape[0])/2.
            extrinsics[0] = extrinsics[1]
        # extrinsic align to scaffold
        base_inv_extrinsic = np.linalg.inv(deepcopy(extrinsics[1]))
        for i in range(len(extrinsics)): extrinsics[i] =  extrinsics[i] @ base_inv_extrinsic
        # re-set a scene
        rescene = Gaussian_Scene(self.cfg)
        rescene.traj_type = scene.traj_type
        # process scaffold frame: modify intrinsic, dpt, and edge(inpaint mask)
        outpaint_frame = deepcopy(scene.frames[1])
        outpaint_frame.intrinsic = intrinsics[1]
        outpaint_frame.extrinsic = np.eye(4)
        outpaint_frame.dpt = dpts[1]
        outpaint_frame.sky = self.sky._segment_sky_(outpaint_frame.rgb)
        # 1. reset sky value
        self.sky._set_sky_depth_(outpaint_frame)
        outpaint_frame = self.sky(outpaint_frame) # sphere
        # 2. remove edge areas
        edge_mask = edge_filter(outpaint_frame.dpt,outpaint_frame.sky,times=self.edge_filter_times)
        outpaint_frame.inpaint = np.ones_like(outpaint_frame.inpaint)>.5
        outpaint_frame.inpaint = (outpaint_frame.inpaint) & (~edge_mask)
        # 3. get input frame back
        input_frame = deepcopy(scene.frames[0])
        input_frame.intrinsic = intrinsics[0]
        input_frame.extrinsic = np.eye(4)
        # 4. set dpt, sky, edge of input frame from outpaint frame    
        H,W = input_frame.H, input_frame.W
        begin_H, begin_W = (outpaint_frame.H-H)//2, (outpaint_frame.W-W)//2
        input_area = np.zeros_like(outpaint_frame.inpaint)
        input_area[begin_H:begin_H+H,begin_W:begin_W+W] = 1.
        input_area = input_area > .5
        input_frame.dpt = outpaint_frame.dpt[input_area].reshape(H,W)
        input_frame.inpaint = outpaint_frame.inpaint[input_area].reshape(H,W)
        # 5. feed them to the scene
        rescene._add_trainable_frame(input_frame,require_grad=True)
        rescene._add_trainable_frame(outpaint_frame,require_grad=True)
        rescene = GS_Train_Tool(rescene,iters=self.gs_opt_iterations)(rescene.frames,show_progress=False)   
        # 6. other frames     
        for i in tqdm(range(2,len(rgbs))):
            rgb = rgbs[i]
            dpt = dpts[i]
            H,W = rgb.shape[:2]
            prompt = frames[i].prompt
            intrinsic = intrinsics[i]
            extrinsic = extrinsics[i] # force the first one is always identity
            f = Frame(H,W,
                      prompt=prompt,
                      intrinsic=intrinsic,
                      extrinsic=extrinsic)
            f.keep = frames[i].keep
            f.anchor = frames[i].anchor
            # for inpaint mask
            f = rescene._render_for_inpaint(f)
            f.rgb = rgb
            f.dpt = dpt
            f.sky = self.sky._segment_sky_(frame.rgb)
            # reset sky to sphere
            f = self.sky(f)
            # remove edge areas
            edge_mask = edge_filter(f.dpt,f.sky,times=self.edge_filter_times)
            f.inpaint = (f.inpaint) & (~edge_mask)
            # remove occuleded area
            rescene,f = self.remover(rescene,f)
            # add frame to the scene
            rescene._add_trainable_frame(f,require_grad=True)
            rescene = GS_Train_Tool(rescene,iters=self.gs_opt_iterations)(target_frames=[f],show_progress=False)
        # generate a dense trajectory
        rescene.dense_trajs = _generate_trajectory(self.cfg,rescene)
        return rescene
        
    def _inpaint_by_mvr_(self,scene:Gaussian_Scene,frame:Frame):
        '''
        re-estimate depths + (shared intrinsic) of all rgbs
        re-generate inpaint masks
        re-training for all rgbs
        '''
        # estimate depths for all images
        frames = scene.frames + [frame]
        rgbs = [f.rgb for f in frames]
        dpts,intrinsics,extrinsics = self.recons_tool._conduct_mvr_(rgbs)
        # force
        base_inv_extrinsic = np.linalg.inv(deepcopy(extrinsics[0]))
        if len(intrinsics)>1: # force the scaffold thing
            intrinsics[1][0, 0] = intrinsics[0][0, 0]
            intrinsics[1][1, 1] = intrinsics[0][1, 1]
            intrinsics[1][0,-1] = intrinsics[0][0,-1] + (dpts[1].shape[1]-dpts[0].shape[1])/2.
            intrinsics[1][1,-1] = intrinsics[0][1,-1] + (dpts[1].shape[0]-dpts[0].shape[0])/2.
            extrinsics[1] = extrinsics[0]
        # re-set a scene
        rescene = Gaussian_Scene(self.cfg)
        rescene.traj_type = scene.traj_type
        for i in range(len(rgbs)):
            rgb = rgbs[i]
            dpt = dpts[i]
            H,W = rgb.shape[:2]
            prompt = frames[i].prompt
            intrinsic = intrinsics[i]
            extrinsic = extrinsics[i] @ base_inv_extrinsic# force the first one is always identity
            f = Frame(H,W,
                      prompt=prompt,
                      intrinsic=intrinsic,
                      extrinsic=extrinsic)
            f.keep = frames[i].keep
            f.anchor = frames[i].anchor
            # for inpaint mask
            if len(rescene.frames) > 1:
                f = rescene._render_for_inpaint(f)
            else:
                f.inpaint = np.ones((H,W)) > .5
            f.rgb = rgb
            f.dpt = dpt
            f.sky = self.sky._segment_sky_(f.rgb)
            # reset sky to sphere
            if len(rescene.frames)<1: self.sky._set_sky_depth_(f)
            f = self.sky(f)
            # remove edge areas
            edge_mask = edge_filter(f.dpt,f.sky,times=self.edge_filter_times)
            f.inpaint = (f.inpaint) & (~edge_mask)
            # remove occuleded area
            rescene,f = self.remover(rescene,f)
            # add frame to the scene
            rescene._add_trainable_frame(f,require_grad=True)
            rescene = GS_Train_Tool(rescene,iters=self.gs_opt_iterations)(target_frames=[f],show_progress=False)
        # generate a dense trajectory
        rescene.dense_trajs = _generate_trajectory(self.cfg,rescene)
        return rescene
        
    def __call__(self, scene, frame, gs_opt_iterations=None):
        self.gs_opt_iterations = gs_opt_iterations if gs_opt_iterations is not None \
                                        else self.cfg.scene.gaussian.opt_iters_per_frame
        if self.recon_type == 'mde':
            scene = self._inpaint_by_mde_(scene,frame)
        elif self.recon_type == 'mdi':
            scene = self._inpaint_by_mdi_(scene,frame)
        elif self.recon_type == 'mvr':
            if len(scene.frames) < 1:
                # in scaffold stage
                scene = self._inpaint_by_mvr_(scene,frame)
            else:
                scene = self._inpaint_by_mvr_w_scaffold_(scene,frame)
        else:
            raise TypeError(f'Wrong reonstruction type: {self.recon_type}. Desired: mde/mdi/mvr')
        return scene
        
        
        