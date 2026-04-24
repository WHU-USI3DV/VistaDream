import PIL
import torch
import numpy as np
import gsplat as gs
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from dataclasses import dataclass
from ops.utils.utils import (
    dpt2xyz,
    visual_pcd,
    alpha_inpaint_mask,
    transform_points,
    numpy_normalize,
    numpy_quaternion_from_matrix
)

@dataclass
class Frame():
    '''
    rgb: in shape of H*W*3, in range of 0-1
    dpt: in shape of H*W, real depth
    inpaint: bool mask in shape of H*W for inpainting
    intrinsic: 3*3
    extrinsic: array in shape of 4*4
    
    As a class for:
    initialize camera
    accept rendering result
    accept inpainting result
    All at 2D-domain
    '''
    def __init__(self,
                 H: int = None,
                 W: int = None,
                 rgb: np.array = None,
                 dpt: np.array = None,
                 sky: np.array = None,
                 inpaint: np.array = None,
                 intrinsic: np.array = None,
                 extrinsic: np.array = None,
                 prompt: str = None) -> None:
        self.H = H
        self.W = W
        self.rgb = rgb
        self.dpt = dpt
        self.sky = sky
        self.prompt = prompt
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self._rgb_rect()
        self._extr_rect()
        # for inpainting 
        self.inpaint = inpaint
        # for keep supervision
        self.keep = False
        self.modify_mask = None
        # for generate trajectories when using spline/interp trajectory
        self.anchor = False
        self.hole_painting = False

    def _rgb_rect(self):
        if self.rgb is not None:
            if isinstance(self.rgb, PIL.PngImagePlugin.PngImageFile):
                self.rgb = np.array(self.rgb)
            if isinstance(self.rgb, PIL.JpegImagePlugin.JpegImageFile):
                self.rgb = np.array(self.rgb)
            if np.amax(self.rgb) > 1.1:
                self.rgb = self.rgb / 255
            
    def _extr_rect(self):
        if self.extrinsic is None: self.extrinsic = np.eye(4)
        self.inv_extrinsic = np.linalg.inv(self.extrinsic)

@dataclass
class Gaussian_Frame():
    '''
    In-frame-frustrum
    Gaussians from a single RGBD frame
    As a class for:
    accept information from initialized/inpainting+geo-estimated frame
    saving pixelsplat properties including rgb, xyz, scale, rotation, opacity; note here, we made a modification to xyz;
    we first project depth to xyz
    then we tune a scale map(initialized to ones) and a shift map(initialized to zeros), they are optimized and add to the original xyz when rendering
    '''
    # as pixelsplat guassian
    rgb:       torch.Tensor = None,
    scale:     torch.Tensor = None,
    opacity:   torch.Tensor = None,
    rotation:  torch.Tensor = None,
    # gaussian center
    dpt:       torch.Tensor = None,
    xyz:       torch.Tensor = None,
    # as a frame
    H:         int = 480,
    W:         int = 640,

    def __init__(self, frame: Frame, device = 'cuda'):
        '''after inpainting'''
        # de-active functions
        self.rgbs_deact    = torch.logit
        self.scales_deact  = torch.log
        self.opacity_deact = torch.logit
        self.device =  device
        self.ds_times = 1
        # for gaussian initialization
        self._set_property_from_frame(frame)

    def _to_3d(self):
        # inv intrinsic
        xyz = dpt2xyz(self.dpt,self.intrinsic)
        inv_extrinsic = np.linalg.inv(self.extrinsic)
        xyz = transform_points(xyz,inv_extrinsic)
        return xyz

    def _paint_filter(self,paint_mask):
        # down sample along h and w direction
        if self.ds_times > 1:
            H_sample = np.arange(0,paint_mask.shape[0],self.ds_times).astype(np.int64)
            W_sample = np.arange(0,paint_mask.shape[1],self.ds_times).astype(np.int64)
            sample_mask = np.zeros([paint_mask.shape[0],paint_mask.shape[1]])
            sample_mask[H_sample,:] += 1
            sample_mask[:,W_sample] += 1
            sample_mask = sample_mask > 1.5
        else:
            sample_mask = np.ones([paint_mask.shape[0],paint_mask.shape[1]]) > .5
        if np.sum(paint_mask)<3:
            paint_mask = np.zeros((self.H,self.W))
            paint_mask[0:1] = 1
        paint_mask = (paint_mask>.5) & sample_mask 
        self.rgb = self.rgb[paint_mask]
        self.xyz = self.xyz[paint_mask]
        self.scale = self.scale[paint_mask]
        self.opacity = self.opacity[paint_mask]
        self.rotation = self.rotation[paint_mask]
            
    def _to_cuda(self):
        self.rgb        = torch.from_numpy(self.rgb.astype(np.float32)).to(self.device)
        self.xyz        = torch.from_numpy(self.xyz.astype(np.float32)).to(self.device)
        self.scale      = torch.from_numpy(self.scale.astype(np.float32)).to(self.device)
        self.opacity    = torch.from_numpy(self.opacity.astype(np.float32)).to(self.device)
        self.rotation   = torch.from_numpy(self.rotation.astype(np.float32)).to(self.device)

    def _fine_init_scale_rotations(self):
        # from https://arxiv.org/pdf/2406.09394
        """ Compute rotation matrices that align z-axis with given normal vectors using matrix operations. """
        up_axis = np.array([0,1,0])
        nml = self.nml @ self.extrinsic[0:3,0:3]
        qz = numpy_normalize(nml)
        qx = np.cross(up_axis,qz)
        qx = numpy_normalize(qx)
        qy = np.cross(qz,qx)
        qy = numpy_normalize(qy)
        rot = np.concatenate([qx[...,None],qy[...,None],qz[...,None]],axis=-1)
        self.rotation = numpy_quaternion_from_matrix(rot)
        # scale
        safe_nml = deepcopy(self.nml)
        safe_nml[safe_nml[:,:,-1]<0.2,-1] = .2
        normal_xoz = deepcopy(safe_nml)
        normal_yoz = deepcopy(safe_nml)
        normal_xoz[...,1] = 0.
        normal_yoz[...,0] = 0.
        normal_xoz = numpy_normalize(normal_xoz)
        normal_yoz = numpy_normalize(normal_yoz)
        cos_theta_x = np.abs(normal_xoz[...,2])
        cos_theta_y = np.abs(normal_yoz[...,2])
        scale_basic = self.dpt / self.intrinsic[0,0] / np.sqrt(2)
        scale_x = scale_basic / cos_theta_x
        scale_y = scale_basic / cos_theta_y
        scale_z = (scale_x + scale_y) / 10.
        self.scale = np.concatenate([scale_x[...,None],
                                     scale_y[...,None],
                                     scale_z[...,None]],axis=-1) * self.ds_times

    def _coarse_init_scale_rotations(self):
        # gaussian property -- HW3 scale
        self.scale = self.dpt / self.intrinsic[0,0] / np.sqrt(2) * self.ds_times
        self.scale = self.scale[:,:,None].repeat(3,-1)
        # gaussian property -- HW4 rotation
        self.rotation = np.zeros((self.H,self.W,4))
        self.rotation[:,:,0] = 1.
    
    def _set_property_from_frame(self,frame: Frame):
        '''frame here is a complete init/inpainted frame'''
        # basic frame-level property
        self.H = frame.H
        self.W = frame.W
        self.dpt = frame.dpt
        self.intrinsic = frame.intrinsic
        self.extrinsic = frame.extrinsic
        # gaussian property -- xyz with train-able pixel-aligned scale and shift
        self.xyz = self._to_3d()
        # gaussian property -- HW3 rgb
        self.rgb = frame.rgb
        # gaussian property -- HW4 rotation HW3 scale 
        self._coarse_init_scale_rotations()
        # gaussian property -- HW opacity
        self.opacity = np.ones((self.H,self.W,1)) #* 0.8
        # to cuda
        self._paint_filter(frame.inpaint)
        self._to_cuda()
        # de-activate
        self.rgb = self.rgbs_deact(self.rgb)
        self.scale = self.scales_deact(self.scale)
        self.opacity = self.opacity_deact(self.opacity)
        # to torch parameters
        self.rgb = nn.Parameter(self.rgb,requires_grad=False)
        self.xyz = nn.Parameter(self.xyz,requires_grad=False)
        self.scale = nn.Parameter(self.scale,requires_grad=False)
        self.opacity = nn.Parameter(self.opacity,requires_grad=False)
        self.rotation = nn.Parameter(self.rotation,requires_grad=False)

    def _require_grad(self,sign=True):
        self.rgb = self.rgb.requires_grad_(sign)
        self.xyz = self.xyz.requires_grad_(sign)
        self.scale = self.scale.requires_grad_(sign)
        self.opacity = self.opacity.requires_grad_(sign)
        self.rotation = self.rotation.requires_grad_(sign)
    
class Gaussian_Scene():  
    def __init__(self,cfg=None):
        # frames initialing the frame
        self.frames = []
        self.gaussian_frames: list[Gaussian_Frame] = [] # gaussian frame require training at this optimization
        # activate fuctions
        self.rgbs_act    = torch.sigmoid
        self.scales_act  = torch.exp
        self.opacity_act = torch.sigmoid
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # for traj generation
        self.traj_type   = 'spiral' if cfg is None else cfg.scene.traj.traj_type
        self.alpha_thres = 0.7 if cfg is None else cfg.scene.inpaint.alpha_thres
        if cfg is not None:
            self.traj_min_percentage = cfg.scene.traj.near_percentage
            self.traj_max_percentage = cfg.scene.traj.far_percentage
            self.traj_forward_ratio  = cfg.scene.traj.traj_forward_ratio
            self.traj_backward_ratio = cfg.scene.traj.traj_backward_ratio
        else:
            self.traj_min_percentage,self.traj_max_percentage,self.traj_forward_ratio,self.traj_backward_ratio = 5, 50, 0.3, 0.4
        self.dense_trajs = []
        
    # basic operations
    def _render_RGBD(self,frame,background_color='black'):
        '''
        :intinsic: tensor of [fu,fv,cu,cv] 4-dimension
        :extinsic: tensor 4*4-dimension
        :out: tensor H*W*3-dimension
        '''
        background = None
        if background_color =='white':
            background = torch.ones(1,3,device=self.device)
            background[:,-1] = 1. # for depth
        # aligned untrainable xyz and unaligned trainable xyz
        # others
        xyz       = torch.cat([gf.xyz.reshape(-1,3) for gf in self.gaussian_frames],dim=0)
        rgb       = torch.cat([gf.rgb.reshape(-1,3) for gf in self.gaussian_frames],dim=0)
        scale     = torch.cat([gf.scale.reshape(-1,3) for gf in self.gaussian_frames],dim=0)
        opacity   = torch.cat([gf.opacity.reshape(-1) for gf in self.gaussian_frames],dim=0)
        rotation  = torch.cat([gf.rotation.reshape(-1,4) for gf in self.gaussian_frames],dim=0)
        # activate
        rgb       = self.rgbs_act(rgb)
        scale     = self.scales_act(scale)
        rotation  = F.normalize(rotation,dim=1)
        opacity   = self.opacity_act(opacity)
        # property
        H,W = frame.H, frame.W
        intrinsic = torch.from_numpy(frame.intrinsic.astype(np.float32)).to(self.device)
        extrinsic = torch.from_numpy(frame.extrinsic.astype(np.float32)).to(self.device)
        # render
        render_out,render_alpha,_ = gs.rendering.rasterization(means = xyz,
                                                scales    = scale,
                                                quats     = rotation,
                                                opacities = opacity,
                                                colors    = rgb,
                                                Ks        = intrinsic[None],
                                                viewmats  = extrinsic[None],
                                                width     = W, 
                                                height    = H, 
                                                packed    = False,
                                                near_plane= 0.01,
                                                render_mode="RGB+ED",
                                                backgrounds=background) # render: 1*H*W*(3+1)
        render_out  = render_out.squeeze() # result: H*W*(3+1)
        render_rgb  = render_out[:,:,0:3]
        render_dpt  = render_out[:,:,-1]
        return render_rgb, render_dpt, render_alpha
    
    @torch.no_grad()
    def _render_for_inpaint(self,frame):
        # first render
        render_rgb, render_dpt, render_alpha = self._render_RGBD(frame)
        render_msk = alpha_inpaint_mask(render_alpha,self.alpha_thres)
        # to numpy
        render_rgb = render_rgb.detach().cpu().numpy()
        render_dpt = render_dpt.detach().cpu().numpy()
        render_alpha = render_alpha.detach().cpu().numpy()
        # assign back
        frame.rgb = render_rgb
        frame.dpt = render_dpt
        frame.inpaint = render_msk
        return frame
    
    def _add_trainable_frame(self,frame:Frame,require_grad=True):
        # temp: make sky unpaint region -- no gaussian generation, no depth alignment -- no rgb supervision
        # for the init frame, we keep all pixels for finetuning
        self.frames.append(frame)
        gf = Gaussian_Frame(frame, self.device)
        gf._require_grad(require_grad)
        self.gaussian_frames.append(gf)

    def _require_grad_(self,sign=False):
        for gf in self.gaussian_frames:
            gf._require_grad(sign)
            
    def _visualize_(self):
        rgbs = []
        xyzs = []
        # visualize point cloud
        for gf in self.gaussian_frames:
            rgb = self.rgbs_act(gf.rgb).detach().cpu().numpy()
            xyz = gf.xyz.detach().cpu().numpy()
            # sample
            idx = np.random.permutation(len(xyz))[0:50000]
            rgbs.append(rgb[idx])
            xyzs.append(xyz[idx])
        rgbs = np.concatenate(rgbs,axis = 0)
        xyzs = np.concatenate(xyzs,axis = 0)
        visual_pcd(xyzs,rgbs)
            