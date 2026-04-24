
'''
render using frames in GS
inpaint with fooocus
'''
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from ops.utils.utils import *
from ops.gs.train import GS_Train_Tool
from ops.gs.basic import Frame,Gaussian_Scene

class Post_Optimize():
    def __init__(self):
        self.iters = 10 #self.cfg.scene.mcs.pre_hole_inpainting.nviews
        self.steps = 64 #self.cfg.scene.mcs.pre_hole_inpainting.nsteps
        self.alpha_thres = 0.7
    
    def __call__(self, scene:Gaussian_Scene):
        trajs = scene.dense_trajs
        print('[Optimization before MCS]...')
        for i in tqdm(range(self.iters)):
            ratio = []
            for traj in trajs:
                idx = np.random.randint(1,len(trajs)-1)
                frame = Frame(H=scene.frames[0].H,
                            W=scene.frames[0].W,
                            intrinsic=deepcopy(scene.frames[0].intrinsic),
                            extrinsic=np.linalg.inv(traj))
                _,_,alpha = scene._render_RGBD(frame)
                alpha = alpha.detach().cpu().numpy()
                ratio.append(np.mean(alpha<self.alpha_thres))
            ratio = np.array(ratio)
            idx = np.argmax(ratio)
            frame = Frame(H=scene.frames[0].H,
                          W=scene.frames[0].W,
                          intrinsic=deepcopy(scene.frames[0].intrinsic),
                          extrinsic=np.linalg.inv(trajs[idx]))
            rgb,dpt,alpha = scene._render_RGBD(frame)
            rgb = rgb.detach().cpu().numpy()
            dpt = dpt.detach().cpu().numpy()
            alpha = alpha.detach().squeeze().cpu().numpy()
            inpaint_msk = alpha < self.alpha_thres
            if np.sum(inpaint_msk) < 50: break
            rgb,dpt = fill_mask_with_nearest([rgb,dpt],inpaint_msk.astype(np.float64))
            frame.rgb = rgb
            frame.dpt = dpt
            frame.inpaint = inpaint_msk
            frame.hole_painting = True
            scene._add_trainable_frame(frame,require_grad=True)
            trainer = GS_Train_Tool(scene,self.steps)
            scene = trainer([frame],only_inpainted_area=True,show_progress=False)
        return scene

class WarpExtend_Phase():
    def __init__(self,cfg,tools) -> None:
        self.device = 'cuda'
        self.cfg = cfg
        self.tools = tools
        self.next_extend_margin = 32
        self.n_sample = self.cfg.scene.traj.n_sample
        self.opt_iters_per_frame = self.cfg.scene.gaussian.opt_iters_per_frame
        self.coarse_interval_rgb_fn = None
        # post optimizer
        self.post_optimizer = Post_Optimize()
    
    def _set_coarse_fn_(self):
        rgb_fn = self.cfg.scene.input.rgb
        if os.path.exists(rgb_fn):
            dir = rgb_fn[:str.rfind(rgb_fn,'/')]
            self.coarse_interval_rgb_fn = f'{dir}/temp.coarse.interval.png'
    
    def _pose_to_frame(self,scene:Gaussian_Scene,pose):
        extrinsic = np.linalg.inv(pose)
        H = scene.frames[0].H + self.next_extend_margin
        W = scene.frames[0].W + self.next_extend_margin
        prompt = scene.frames[-1].prompt
        intrinsic = deepcopy(scene.frames[0].intrinsic)
        intrinsic[0,-1], intrinsic[1,-1] = W/2, H/2
        frame = Frame(H=H,W=W,intrinsic=intrinsic,extrinsic=extrinsic,prompt=prompt)
        frame = scene._render_for_inpaint(frame)  
        return frame
      
    def _next_frame(self,scene:Gaussian_Scene):
        # select the frame with largest holes but less than 60% 
        inpaint_area_ratio = []
        for pose in scene.dense_trajs:
            temp_frame = self._pose_to_frame(scene,pose)
            inpaint_mask = temp_frame.inpaint 
            inpaint_area_ratio.append(np.mean(inpaint_mask))
        inpaint_area_ratio = np.array(inpaint_area_ratio)
        inpaint_area_ratio[inpaint_area_ratio > 0.6] = 0.
        # remove adjustancy frames
        for s in self.select_frames:
            inpaint_area_ratio[s] = 0.
            if s-1>-1:
                inpaint_area_ratio[s-1] = 0.
            if s+1<len(scene.dense_trajs):
                inpaint_area_ratio[s+1] = 0.
        # select the largest ones
        select = np.argmax(inpaint_area_ratio)
        if inpaint_area_ratio[select] < 0.03: return None
        self.select_frames.append(select)
        pose = scene.dense_trajs[select]
        frame = self._pose_to_frame(scene,pose)
        return frame   

    def _inpaint_next_frame(self,scene:Gaussian_Scene,frame:Frame):
        self._set_coarse_fn_()
        # inpaint the rgb
        if self.coarse_interval_rgb_fn is not None:
            save_pic(frame.rgb,self.coarse_interval_rgb_fn)
        frame = self.tools.rgb_inpaint(frame)
        if self.coarse_interval_rgb_fn is not None:
            save_pic(frame.rgb,self.coarse_interval_rgb_fn)
        torch.cuda.empty_cache()
        # inpaint the dpt
        scene = self.tools.dpt_inpaint(scene,frame)
        # temp visualization
        return scene

    def _warp_and_inpaint(self,scene:Gaussian_Scene):
        self.select_frames = []
        warp_samples = self.n_sample - len(scene.frames) + 1 # exclude input view
        for i in range(warp_samples):
            print(f'Processing {i+1}/{warp_samples} frame...')
            next_frame = self._next_frame(scene)
            if next_frame is None: break
            # here this frame is auto, not user-defined, not anchor
            next_frame.anchor = False
            scene = self._inpaint_next_frame(scene,next_frame)
        return scene

    def _inpaint_holes(self,scene:Gaussian_Scene):
        scene = self.post_optimizer(scene)
        return scene

    def __call__(self,scene:Gaussian_Scene):
        self._set_coarse_fn_()
        # warp and inpaint
        scene = self._warp_and_inpaint(scene)
        scene = self._inpaint_holes(scene)
        return scene
        