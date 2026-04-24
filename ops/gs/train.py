import cv2
import tqdm
import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from ops.utils.utils import save_pic
from ops.gs.basic import Gaussian_Scene,Frame
from torchmetrics.image import StructuralSimilarityIndexMeasure

class RGB_Loss():
    def __init__(self,w_lpips=0.2,w_ssim=0.2):
        self.rgb_loss = F.smooth_l1_loss
        # self.lpips_alex = lpips.LPIPS(net='alex').to('cuda')
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to('cuda')
        self.w_ssim = w_ssim
        self.w_lpips = w_lpips
        
    def __call__(self,pr,gt,valid_mask=None):
        pr = torch.nan_to_num(pr)
        gt = torch.nan_to_num(gt)
        if len(pr.shape) < 3: pr = pr[:,:,None].repeat(1,1,3)
        if len(gt.shape) < 3: gt = gt[:,:,None].repeat(1,1,3)
        # size align
        h,w = gt.shape[0:2]
        pr = F.interpolate(pr.permute(2,0,1)[None],(h,w))[0].permute(1,2,0)
        # calculate
        pr_valid = pr[valid_mask] if valid_mask is not None else pr.reshape(-1,pr.shape[-1])
        gt_valid = gt[valid_mask] if valid_mask is not None else gt.reshape(-1,gt.shape[-1])
        l_rgb = self.rgb_loss(pr_valid,gt_valid)
        l_ssim = 1.0 - self.ssim(pr[None].permute(0, 3, 1, 2), gt[None].permute(0, 3, 1, 2))
        return l_rgb + self.w_ssim * l_ssim

class Scale_Loss():
    def __init__(self,optimize_frames):
        self.optimize_frames = optimize_frames
        self.basic = torch.mean(torch.cat([gf.scale for gf in self.optimize_frames]).detach())
    
    def __call__(self):
        # do not derive too much
        scales = torch.cat([gf.scale for gf in self.optimize_frames])
        scales_disp = (scales - self.basic)**2
        scales_loss = torch.mean(scales_disp)
        return scales_loss    
    
class Smooth_Loss():
    def __init__(self):
        pass    
    
    def _knn_idx_(self,xyz):
        # query and database
        query_idx = np.random.permutation(len(xyz))[0:2000]
        bases_idx = np.random.permutation(len(xyz))[0:50000]
        query_idx = torch.from_numpy(query_idx).to(xyz).long()
        bases_idx = torch.from_numpy(bases_idx).to(xyz).long()
        # xyzs
        query_xyz = xyz[query_idx]
        bases_xyz = xyz[bases_idx]
        # knn
        query_extend = query_xyz[:,None].repeat(1,len(bases_idx),1)
        bases_extend = bases_xyz[None,:].repeat(len(query_idx),1,1)
        qbd = torch.square(query_extend - bases_extend).sum(-1)
        qbd = torch.argsort(qbd,dim=-1)[...,0:20]
        query_knn_idx = bases_idx[qbd.reshape(-1)].reshape(len(query_idx),-1)
        return query_idx,query_knn_idx
    
    def _svd_(self,center,knns):
        # center: n*3, knns: n*k*3
        knns_center = knns - center[:,None,:]
        knns_covera = torch.einsum('nkt,nkh->nth',knns_center,knns_center)
        # torch svd
        S = torch.linalg.svdvals(knns_covera)
        S = torch.amin(S,dim=-1)
        return S     
    
    def _xyz_smooth_(self,xyz):
        query_idx, query_knn_idx = self._knn_idx_(xyz)
        query_xyz, query_knn_xyz = xyz[query_idx],\
                                   xyz[query_knn_idx.reshape(-1)].reshape(len(query_idx),-1,3)
        S = self._svd_(query_xyz,query_knn_xyz)
        l = torch.mean(torch.exp(S)-1)
        return l
    
    def __call__(self, optimize_frames):
        xyz = torch.cat([gf.xyz for gf in optimize_frames])
        l = self._xyz_smooth_(xyz)
        return l

class GS_Train_Tool():
    '''
    Frames and well-trained gaussians are kept, refine the trainable gaussians
    The supervision comes from the Frames of GS_Scene
    '''
    def __init__(self,
                 GS:Gaussian_Scene,
                 iters = 100) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # hyperparameters for prune, densify, and update
        self.lr_factor = 1.00
        self.lr_update = 0.99
        # learning rate
        self.rgb_lr = 0.0005
        self.xyz_lr = 0.0001
        self.scale_lr = 0.005
        self.opacity_lr = 0.05
        self.rotation_lr = 0.001
        # GSs for training
        self.GS = GS
        # hyperparameters for training
        self.iters = iters
        self._init_optimizer()
        self.rgb_lossfunc = RGB_Loss(w_ssim=0.2)
        self.scale_lossfunc = Scale_Loss(self.optimize_frames)
        self.smooth_lossfunc = Smooth_Loss()
    
    def _init_optimizer(self):
        self.optimize_frames = [gf for gf in self.GS.gaussian_frames if gf.rgb.requires_grad]
        # following https://github.com/pointrix-project/msplat
        self.optimizer = torch.optim.Adam([
            {'params': [gf.xyz for gf in self.optimize_frames],      'lr': self.xyz_lr},
            {'params': [gf.rgb for gf in self.optimize_frames],      'lr': self.rgb_lr},
            {'params': [gf.scale for gf in self.optimize_frames],    'lr': self.scale_lr},
            {'params': [gf.opacity for gf in self.optimize_frames],  'lr': self.opacity_lr},
            {'params': [gf.rotation for gf in self.optimize_frames], 'lr': self.rotation_lr}
        ]) 

    def _render(self,frame):
        rgb,dpt,alpha = self.GS._render_RGBD(frame)
        return rgb,dpt,alpha
    
    def _to_cuda(self,tensor):
        tensor = torch.from_numpy(tensor.astype(np.float32)).to('cuda')
        return tensor
    
    def _get_rgb_loss_(self,frame,only_inpainted_area=True):
        # render
        render_rgb,_,_=self._render(frame)
        # anchor
        anchor_rgb = deepcopy(self._to_cuda(frame.rgb))
        # valid msk
        # check modify_mask # only for keep frames and only for keep occlusion
        if frame.modify_mask is None:
            frame.modify_mask = np.zeros((frame.H,frame.W))
        # refine
        valid_mask = deepcopy(frame.modify_mask)
        kernel = np.ones((5,5),np.uint8)
        valid_mask = cv2.erode(valid_mask,kernel,iterations=1)
        valid_mask = cv2.dilate(valid_mask,kernel,iterations=2)
        valid_mask = valid_mask < 5.
        # loss rgb
        if only_inpainted_area:
            valid_mask = valid_mask & frame.inpaint
        valid_mask = torch.from_numpy(valid_mask).to(render_rgb.device).bool()
        render_rgb[~valid_mask] *= 0.
        anchor_rgb[~valid_mask] *= 0.
        # calculate loss
        loss_rgb = self.rgb_lossfunc(render_rgb,anchor_rgb,valid_mask=valid_mask)
        return loss_rgb
    
    def _get_scale_loss_(self):
        return self.scale_lossfunc()
    
    def _get_smooth_loss_(self):
        return self.smooth_lossfunc(self.optimize_frames)
    
    def __call__(self,target_frames=None,only_inpainted_area=True,show_progress=True,no_grad_after=True):
        target_frames = self.GS.frames if target_frames is None else target_frames
        if show_progress:
            bar = tqdm.tqdm(range(self.iters))
        else:
            bar = range(self.iters)
        for iter in bar:
            # randomly sample on frame
            frame_idx = np.random.randint(0,len(target_frames))
            frame :Frame = target_frames[frame_idx]
            # rgb loss
            loss_rgb = self._get_rgb_loss_(frame,only_inpainted_area)
            # supervise the scale not to be too large
            loss_scale = self._get_scale_loss_()
            # optimization 
            loss = loss_rgb + loss_scale
            loss.backward()  
            self.optimizer.step()
            self.optimizer.zero_grad()
        refined_scene = self.GS
        if no_grad_after:
            refined_scene._require_grad_(False)
        return refined_scene
    
