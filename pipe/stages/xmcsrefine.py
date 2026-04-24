'''
Coarse Gaussian Rendering -- RGB-D as init
RGB-D add noise (MV init)
Cycling:
    denoise to x0 and d0 -- optimize Gaussian
    re-rendering RGB-D
    render RGB-D to rectified noise
    noise rectification
    step denoise with rectified noise
-- Finally the Gaussian
'''
import cv2
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from ops.utils.utils import *
from ops.gs.train import GS_Train_Tool
from ops.gs.basic import Frame,Gaussian_Scene

class MCS_Phase():
    def __init__(self,
                 cfg,
                 tools,
                 scene:Gaussian_Scene,
                 device = 'cuda',
                 steps = None,
                 views = None,
                 rectw = None) -> None:
        # input coarse GS
        self.cfg = cfg
        self.coarse_GS = scene
        self.RGB_LCM = tools.mcs_refiner
        # refine frames to be refined; here we refine frames rather than gaussian paras
        self.refine_interval_rgb_fn = None
        self.ad_steps = steps if steps is not None else self.RGB_LCM.denoise_steps
        self.ac_steps = 0
        # leave some anchor steps
        self.steps = self.ad_steps - self.ac_steps
        # others
        self.rect_w = rectw if rectw is not None else self.cfg.scene.mcs.rect_w
        self.n_gsopt_iters = self.cfg.scene.mcs.gsopt_iters
        # refiner
        self.refine_frames: list[Frame] = []
        # hyperparameters total is 50 steps and here is the last N steps
        self.process_res = 512
        self.device = device
        # models
        self.RGB_LCM.denoise_steps = self.ad_steps
        self.RGB_LCM.to(device) 
        # prompt for diffusion
        prompt = self.coarse_GS.frames[0].prompt
        self.rgb_prompt_latent = self.RGB_LCM.model._encode_text_prompt(prompt)
        # sample and blurrness
        self.blurriness = False
        self.ratio_sample = True
        self.n_view = views if views is not None else self.cfg.scene.mcs.n_view
        
    def _set_refine_fn_(self):
        rgb_fn = self.cfg.scene.input.rgb
        if os.path.exists(rgb_fn):
            dir = rgb_fn[:str.rfind(rgb_fn,'/')]
            self.refine_interval_rgb_fn = f'{dir}/temp.refine.interval.png' 
    
    def _sample_next_view_(self,
                           scene:Gaussian_Scene,
                           full_scene:Gaussian_Scene,
                           trajs,sampled,target_H,target_W,intrinsic):
        # render temp_scene to all trajs
        ratios,frames = [],[]
        for i, pose in enumerate(trajs):
            frame = Frame()
            frame.H = target_H
            frame.W = target_W
            frame.extrinsic = np.linalg.inv(pose)
            frame.intrinsic = deepcopy(intrinsic)
            frame = scene._render_for_inpaint(frame)
            ratios.append(np.mean(frame.inpaint))
            frames.append(frame)
        # sample the view with max holes / inpaint area
        for t in sampled:
            ratios[t] = 0.
        idx = int(np.argmax(np.array(ratios)))
        next_view = frames[idx]
        inpaint_mask = deepcopy(next_view.inpaint)
        next_view = full_scene._render_for_inpaint(next_view)
        next_view.inpaint = inpaint_mask
        # add frame back
        scene._add_trainable_frame(next_view,require_grad=False)
        return scene, idx
    
    def _pre_process(self): 
        # determine the diffusion target shape
        strict_times = 16
        # pre-optimize
        origin_H = self.coarse_GS.frames[0].H
        origin_W = self.coarse_GS.frames[0].W
        self.target_H,self.target_W = self.process_res,self.process_res
        # reshape to the same (target) shape for rendering and denoising
        intrinsic = deepcopy(self.coarse_GS.frames[0].intrinsic)
        H_ratio, W_ratio = self.target_H/origin_H, self.target_W/origin_W
        intrinsic[0] *= W_ratio
        intrinsic[1] *= H_ratio 
        target_H, target_W = self.target_H+2*strict_times, self.target_W+2*strict_times
        intrinsic[0,-1] += strict_times
        intrinsic[1,-1] += strict_times
        # generate a set of cameras
        
        if self.ratio_sample:
            sample_trajs = []
            trajs = self.coarse_GS.dense_trajs
            sample_inits = np.arange(0,len(trajs),3)
            trajs = [trajs[int(i)] for i in sample_inits]
            # determine inpaint mask -- no use now
            temp_scene = Gaussian_Scene()
            # anchor first
            for frame in self.coarse_GS.frames:
                if frame.keep:
                    temp_scene._add_trainable_frame(frame,require_grad=False)
            sampled = []
            for i in range(self.n_view):
                temp_scene,next_pose_idx = self._sample_next_view_(temp_scene,
                                                               self.coarse_GS,
                                                               trajs,sampled,
                                                               self.coarse_GS.frames[0].H,
                                                               self.coarse_GS.frames[0].W,
                                                               self.coarse_GS.frames[0].intrinsic,)
                sampled.append(next_pose_idx)
                sample_trajs.append(trajs[next_pose_idx])
            del temp_scene
        else:
            trajs = self.coarse_GS.dense_trajs
            sample_idx = np.linspace(2,len(trajs)-1,self.n_view)
            sample_trajs = [trajs[int(idx)] for idx in sample_idx]
        
        # refine frames
        for i, pose in enumerate(sample_trajs):
            fine_frame = Frame()
            fine_frame.H = target_H
            fine_frame.W = target_W
            fine_frame.extrinsic = np.linalg.inv(pose)
            fine_frame.intrinsic = deepcopy(intrinsic)
            fine_frame.prompt  = self.coarse_GS.frames[-1].prompt
            self.refine_frames.append(fine_frame) 
            
    def _mv_init(self):
        rgbs = []
        # only for inpainted images
        for i,frame in enumerate(self.refine_frames):
            # rendering at now; all in the same shape
            render_rgb,render_dpt,render_alpha=self.coarse_GS._render_RGBD(frame)
            rgbs.append(render_rgb.permute(2,0,1)[None])
        self.rgbs = torch.cat(rgbs,dim=0)
        self.RGB_LCM._encode_mv_init_images(self.rgbs)

    def _to_cuda(self,tensor):
        tensor = torch.from_numpy(tensor.astype(np.float32)).to('cuda')
        return tensor

    def _x0_rectification_by_GS_(self, denoise_rgb, iters):
        # gaussian initialization
        CGS = deepcopy(self.coarse_GS)
        for gf in CGS.gaussian_frames:
            gf._require_grad(True)
        self.refine_GS = GS_Train_Tool(CGS)
        # rectification
        for iter in tqdm(range(iters)):
            loss = 0.
            # supervise on input view
            for i,keep_frame in enumerate(self.coarse_GS.frames):
                if not keep_frame.keep: continue
                loss_rgb = self.refine_GS._get_rgb_loss_(deepcopy(keep_frame),only_inpainted_area=False) # the same as keep
                loss += loss_rgb * len(self.refine_frames) 
            # optimization
            loss.backward()  
            self.refine_GS.optimizer.step()
            self.refine_GS.optimizer.zero_grad()
            loss = 0.
            # then multiview supervision
            for i,frame in enumerate(self.refine_frames):
                render_rgb,_,_ = self.refine_GS._render(frame)
                loss_rgb_item = self.refine_GS.rgb_lossfunc(denoise_rgb[i],render_rgb)
                loss += loss_rgb_item 
            # optimization
            loss.backward()  
            self.refine_GS.optimizer.step()
            self.refine_GS.optimizer.zero_grad()
        self.refine_GS.GS._require_grad_(False)

    def _step_gaussian_optimization(self,step,iters=None):
        # denoise to x0 and d0
        with torch.no_grad():
            # we left the last 2 steps for stronger guidances
            rgb_noise_pr,rgb_denoise,rgb_t = self.RGB_LCM._denoise_to_x0(step,self.rgb_prompt_latent)
            rgb_denoise = rgb_denoise.permute(0,2,3,1)
        
        if self.blurriness:
            if (step+1)%100 == 0:
            # we need to make refinement sharp
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) 
                rgb_denoise = rgb_denoise.detach().cpu().numpy()
                for t in range(1):
                    rgb_denoise = [cv2.filter2D(rgb, -1, kernel=kernel) for rgb in rgb_denoise]
                rgb_denoise = np.stack(rgb_denoise)
                rgb_denoise = torch.from_numpy(rgb_denoise.astype(np.float32)).cuda()
                
        # rendering each frames and weight-able refinement
        iters = self.n_gsopt_iters if iters is None else iters
        self._x0_rectification_by_GS_(rgb_denoise,iters)    
        torch.cuda.empty_cache()
        return rgb_t, rgb_noise_pr

    def _step_diffusion_rectification(self, rgb_t, rgb_noise_pr):
        # re-rendering RGB
        with torch.no_grad():
            x0_rect = []
            for i,frame in enumerate(self.refine_frames):
                re_render_rgb,_,re_render_alpha= self.refine_GS._render(frame)
                
                if self.blurriness:
                    # avoid rasterization holes yield more block holes and more
                    re_render_rgb = re_render_rgb.detach().cpu().numpy()
                    re_render_alpha = re_render_alpha.squeeze().detach().cpu().numpy()
                    re_render_rgb = inpaint_tiny_holes(re_render_rgb,re_render_alpha,0.9)
                    re_render_rgb = torch.from_numpy(re_render_rgb).to(rgb_noise_pr)
                
                x0_rect.append(re_render_rgb.permute(2,0,1)[None])
            x0_rect = torch.cat(x0_rect,dim=0)
        # rectification
        self._visual_check_()
        self.RGB_LCM._step_denoise(rgb_t,rgb_noise_pr,x0_rect,rect_w=self.rect_w) 

    def _visual_check_(self):
        # randomly Visualization
        if self.refine_interval_rgb_fn is not None:
            random_frame = 3
            random_frame = self.refine_frames[random_frame]
            rgb,_,_ = self.refine_GS._render(random_frame)
            rgb = rgb.detach().cpu().numpy()
            save_pic(rgb,self.refine_interval_rgb_fn)      
                      
    def __call__(self):
        self._set_refine_fn_()
        # warmup
        self._pre_process()
        self._mv_init()
        for step in tqdm(range(self.steps-1)):
            iters = self.n_gsopt_iters
            rgb_t, rgb_noise_pr = self._step_gaussian_optimization(step,iters=iters)
            self._step_diffusion_rectification(rgb_t, rgb_noise_pr)
        # final optimization
        self._step_gaussian_optimization(self.steps-1,iters=self.n_gsopt_iters)
        scene = self.refine_GS.GS
        for gf in scene.gaussian_frames:
            gf._require_grad(False)
        self.RGB_LCM.to('cpu')
        return scene