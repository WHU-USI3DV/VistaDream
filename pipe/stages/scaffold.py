'''
only valid when using mde/mdi for depth inpainting
'''
import torch,os
import numpy as np
from copy import deepcopy
from ops.utils.utils import *
from ops.gs.train import GS_Train_Tool
from ops.trajs import _generate_trajectory
from ops.gs.basic import Frame,Gaussian_Scene
        
class Scaffold_Phase():
    '''
    yield prompt
    build scaffold
    create trajectory
    '''
    
    def __init__(self,cfg,tools) -> None:
        self.cfg = cfg
        self.tools = tools
        self.coarse_interval_rgb_fn = None

    def _set_coarse_fn_(self):
        rgb_fn = self.cfg.scene.input.rgb
        if os.path.exists(rgb_fn):
            dir = rgb_fn[:str.rfind(rgb_fn,'/')]
            self.coarse_interval_rgb_fn = f'{dir}/temp.coarse.interval.png'

    def _lvm_prompt_(self,rgb,user_prompt=None):
        if user_prompt is not None and len(str(user_prompt).strip()) > 0:
            self.prompt = str(user_prompt).strip()
            print('[INFO] Use user-provided text prompt. Skip LLaVA captioning.')
            return

        if self.cfg.scene.outpaint.lvm_prompt:
            if hasattr(self.tools, '_ensure_llava_'):
                self.tools._ensure_llava_()
            if (not hasattr(self.tools, 'llava')) or (self.tools.llava is None):
                print('[WARN] LLaVA is unavailable. Fallback to default prompt.')
                self.prompt = 'A beautiful scene with clear objects and coherent layouts.'
                return

            query = '<image>\n \
                    USER: Detaily imagine and describe the scene this image taken from? \
                    \n ASSISTANT: This image is taken from a scene of ' 
            try:
                print('Inpaint-Caption[1/3] Move llava.model to GPU...')
                self.tools.llava.model.to('cuda')
                print('Inpaint-Caption[2/3] Llava inpainting instruction:')
                prompt = self.tools.llava(rgb,query)
                split  = str.rfind(prompt,'ASSISTANT: This image is taken from a scene of ') \
                                + len(f'ASSISTANT: This image is taken from a scene of ')
                prompt = prompt[split:]
                print("Prompt:",prompt)
                prompt = 'This is a real scene but not a photo, no frame, no rim. This is a scene of ' + prompt
                self.prompt = prompt
            except Exception as e:
                print(f'[WARN] LLaVA prompt generation failed: {e}')
                self.prompt = 'A beautiful scene with clear objects and coherent layouts.'
            finally:
                try:
                    print('Inpaint-Caption[3/3] Move llava.model to CPU...')
                    self.tools.llava.model.to('cpu')
                except Exception:
                    pass
                torch.cuda.empty_cache()
        else:
            self.tools.llava = None
            self.prompt = 'A beautiful scene with clear objects and coherent layouts.'

    def _wo_zoom_out_scaffold(self,rgb):
        H,W = rgb.shape[0:2]
        scene = Gaussian_Scene(self.cfg)
        frame = Frame(H=H,W=W,rgb=rgb,prompt=self.prompt)
        frame.keep = True
        frame.inpaint = np.ones((H,W)) > 0.5
        scene = self.tools.dpt_inpaint(scene,frame)
        # split into two frames
        return scene

    def _zoom_out_scaffold_(self,rgb,intrinsic=None):
        # conduct outpainting on rgb and change cu,cv
        outpaint_frame :Frame = self.tools.rgb_inpaint(Frame(rgb=rgb,prompt=self.prompt,intrinsic=deepcopy(intrinsic),extrinsic=np.eye(4)),
                                                       outpaint_selections=self.cfg.scene.outpaint.outpaint_selections,
                                                       outpaint_extend_times=self.cfg.scene.outpaint.outpaint_extend_times)
        outpaint_area = deepcopy(outpaint_frame.inpaint)
        outpaint_frame.inpaint = np.ones_like(outpaint_frame.inpaint)>.5
        # estimate depth for the outpainted rgb
        scene = Gaussian_Scene(self.cfg)
        scene = self.tools.dpt_inpaint(scene,outpaint_frame,gs_opt_iterations=0)
        # split the out-paint frame into original one and outpaint one
        # outpaint frame
        outpaint_frame = scene.frames[0]
        outpaint_edges = deepcopy(outpaint_frame.inpaint)
        outpaint_frame.inpaint = outpaint_edges & outpaint_area
        # input frame
        H,W = rgb.shape[0:2]
        intrinsic = deepcopy(outpaint_frame.intrinsic)
        intrinsic[0,-1] = rgb.shape[1]/2
        intrinsic[1,-1] = rgb.shape[0]/2
        input_frame = Frame(H=H,
                            W=W,
                            rgb=rgb,
                            dpt=outpaint_frame.dpt[~outpaint_area].reshape(H,W),
                            sky=outpaint_frame.sky[~outpaint_area].reshape(H,W),
                            inpaint = outpaint_edges[~outpaint_area].reshape(H,W),
                            intrinsic=intrinsic,
                            extrinsic=np.eye(4),
                            prompt=outpaint_frame.prompt)
        input_frame.keep = True
        outpaint_frame.keep = True
        # the scene with scaffold
        scene = Gaussian_Scene(self.cfg)
        scene._add_trainable_frame(input_frame,require_grad=True)
        scene._add_trainable_frame(outpaint_frame,require_grad=True)
        scene = GS_Train_Tool(scene,iters=self.cfg.scene.gaussian.opt_iters_per_frame)(scene.frames) 
        return scene
    
    def _generate_traj(self,scene:Gaussian_Scene):
        dense_trajs = _generate_trajectory(self.cfg,scene)
        scene.dense_trajs = dense_trajs
        return scene
    
    def __call__(self, rgb, intrinsic = None, user_prompt = None):
        rgb = np.array(rgb)[...,0:3]
        # describe
        self._lvm_prompt_(rgb,user_prompt=user_prompt)
        self.recon_type = self.cfg.scene.reconstruction.type
        if not self.cfg.scene.outpaint.sign:
            scene = self._wo_zoom_out_scaffold(rgb)
        else:
            scene = self._zoom_out_scaffold_(rgb,intrinsic)
        self._set_coarse_fn_()
        save_pic(scene.frames[-1].rgb,self.coarse_interval_rgb_fn)
        scene = self._generate_traj(scene)
        return scene
            