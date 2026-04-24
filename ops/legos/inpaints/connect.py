import cv2
import numpy as np
from copy import deepcopy
from ops.utils.utils import dpt2xyz,transform_points,save_pic

class Connect_Tool():
    def __init__(self) -> None:
        pass
        
    def _align_scale_shift_numpy(self, pred: np.array, target: np.array):
        mask = (target > 0) & (pred < 199)
        target_mask = target[mask]
        pred_mask = pred[mask]
        if np.sum(mask) > 10:
            scale, shift = np.polyfit(pred_mask, target_mask, deg=1)
            if scale < 0:
                scale = np.median(target[mask]) / (np.median(pred[mask]) + 1e-8)
                shift = 0
        else:
            scale = 1
            shift = 0
        return scale,shift
        
    def __call__(self, render_dpt, inpaint_dpt, inpaint_msk):
        if np.sum(inpaint_msk > 0.5) < 1.: return render_dpt
        # get areas need to be aligned
        render_dpt_valid  = render_dpt[~inpaint_msk]
        inpaint_dpt_valid = inpaint_dpt[~inpaint_msk]
        # rectify
        scale,shift = self._align_scale_shift_numpy(inpaint_dpt_valid,render_dpt_valid)
        inpaint_dpt = inpaint_dpt*scale + shift
        return inpaint_dpt

class Smooth_Connect_Tool():
    def __init__(self) -> None:
        self.coarse_align = Connect_Tool()
    
    def _coarse_alignment(self, render_dpt, ipaint_dpt, ipaint_msk):
        # determine the scale and shift of inpaint_dpt to coarsely align it to render_dpt
        inpaint_dpt = self.coarse_align(render_dpt,ipaint_dpt,ipaint_msk)
        return inpaint_dpt
    
    def _refine_movements(self, render_dpt, ipaint_dpt, ipaint_msk):
        '''
        Follow https://arxiv.org/pdf/2311.13384
        '''
        # Determine the adjustment of un-inpainted area
        ipaint_msk = ipaint_msk>.5
        H, W = ipaint_msk.shape[0:2]
        U = np.arange(W)[None,:].repeat(H,axis=0)
        V = np.arange(H)[:,None].repeat(W,axis=1)
        # on kept areas
        keep_render_dpt = render_dpt[~ipaint_msk]
        keep_ipaint_dpt = ipaint_dpt[~ipaint_msk]
        keep_adjust_dpt = keep_render_dpt - keep_ipaint_dpt
        # iterative refinement
        complete_adjust = np.zeros_like(ipaint_dpt)
        for i in range(100):
            complete_adjust[~ipaint_msk] = keep_adjust_dpt
            complete_adjust = cv2.blur(complete_adjust,(15,15))
        # complete_adjust[~ipaint_msk] = keep_adjust_dpt
        ipaint_dpt = ipaint_dpt + complete_adjust        
        return ipaint_dpt
           
    def _affine_dpt_to_GS(self, render_dpt, inpaint_dpt, inpaint_msk):
        if np.sum(inpaint_msk > 0.5) < 1.: return render_dpt
        inpaint_dpt = self._coarse_alignment(render_dpt,inpaint_dpt,inpaint_msk)
        inpaint_dpt = self._refine_movements(render_dpt,inpaint_dpt,inpaint_msk)
        return inpaint_dpt
                     
    def _scale_dpt_to_GS(self, render_dpt, inpaint_dpt, inpaint_msk):
        if np.sum(inpaint_msk > 0.5) < 1.: return render_dpt
        inpaint_dpt = self._refine_movements(render_dpt,inpaint_dpt,inpaint_msk)
        return inpaint_dpt
    
class Occlusion_Removal():
    '''
    Remove or keep
    # remove:
        # inpainted gaussians -- projection to keep (just keep is ok, for we only supervise the scene always be the same as keep) -- depth check (note edge areas and sky has large value); and remove the gaussians
    # keep
        # inpainted gaussians -- projection to keep (all former?) -- keep frame owns a modify-mask, modify the mask w/o remove the gaussians
    # modify
        # inpainted gaussians -- projection to keep (all former?) -- move along the depth direction?
    '''
    
    def _save_check_(self,keep_frame,uv,rgb,d,fn):
        temp = np.zeros((keep_frame.H,keep_frame.W,3))
        temp[uv[:,1],uv[:,0],:] = rgb
        save_pic(temp,f'{fn}.rgb.png')
        temp = np.zeros((keep_frame.H,keep_frame.W))
        temp[uv[:,1],uv[:,0]] = d
        save_pic(temp,f'{fn}.dpt.png')
    
    def __init__(self,keep_occlu=False) -> None:
        self.keep_occlu = keep_occlu
    
    def _to_3d_(self,frame):
        # first get xyz of the newly added frame
        xyz = dpt2xyz(frame.dpt,frame.intrinsic)
        # we only check newly added areas
        xyz = xyz[frame.inpaint]
        # move these xyzs to world coor system
        inv_extrinsic = np.linalg.inv(frame.extrinsic)
        xyz = transform_points(xyz,inv_extrinsic)
        return xyz
        
    def _proj_(self,keep_frame,xyz):
        ''' project to the keep frames '''
        # xyz in camera frustrum
        xyz_camera = transform_points(deepcopy(xyz),keep_frame.extrinsic)
        # uvz in camera frustrum
        uvz_camera = np.einsum(f'ab,pb->pa',keep_frame.intrinsic,xyz_camera)
        # uv and d in camra frustrum
        uv,d = uvz_camera[...,:2]/uvz_camera[...,-1:], uvz_camera[...,-1]
        # in-frusturm pixels
        valid_msk = (uv[...,0]>0) & (uv[...,0]<keep_frame.W) & (uv[...,1]>0) & (uv[...,1]<keep_frame.H) & (d>1e-2)
        valid_idx = np.where(valid_msk)[0]
        return uv, d, valid_idx
    
    def _keep_(self,scene,frame):
        xyz = self._to_3d_(frame)
        for anchor_frame in scene.frames:
            if anchor_frame.modify_mask is None:
                anchor_frame.modify_mask = np.zeros((anchor_frame.H,anchor_frame.W))
            if anchor_frame.keep:
                uv,d,valid_idx = self._proj_(anchor_frame,xyz)
                uv,d = uv[valid_idx].astype(np.uint32),d[valid_idx] 
                # occluded pixels
                # compared depth comes from original keep depth and should be closer
                compard_d = deepcopy(anchor_frame.dpt)
                compare_d = compard_d[uv[:,1],uv[:,0]]
                occlu_msk = (compare_d-d)>(d+compare_d)/30.
                occlu_uvs = uv[occlu_msk].astype(np.int64)
                anchor_frame.modify_mask[occlu_uvs[:,1],occlu_uvs[:,0]] = 1.
        # no change happens to frame
        return scene,frame
    
    def _remove_(self,scene,frame):
        xyz = self._to_3d_(frame)
        msk = np.ones_like(xyz[...,0])
        for anchor_frame in scene.frames:
            if anchor_frame.modify_mask is None:
                anchor_frame.modify_mask = np.zeros((anchor_frame.H,anchor_frame.W))
            if anchor_frame.keep:
                uv,d,valid_idx = self._proj_(anchor_frame,xyz)
                uv,d = uv[valid_idx].astype(np.uint32),d[valid_idx]       
                # self._save_check_(anchor_frame,uv,rgb[valid_idx],d,'proj')
                # occluded pixels
                # compared depth comes from original keep depth and should be closer
                compard_d = deepcopy(anchor_frame.dpt)
                compare_d = compard_d[uv[:,1],uv[:,0]]
                occlu_msk = (compare_d-d)>(d+compare_d)/30.
                # this pixels will be removed
                invalid_idx = valid_idx[occlu_msk]
                # self._save_check_(anchor_frame,uv[occlu_msk],rgb[invalid_idx],d[occlu_msk],'removed')
                msk[invalid_idx] = 0.
        # USE indexes rather than [][]
        inpaint_idx_v,inpaint_idx_u = np.where(frame.inpaint)
        inpaint_idx_v = inpaint_idx_v[msk<.5]
        inpaint_idx_u = inpaint_idx_u[msk<.5]
        frame.inpaint[inpaint_idx_v,inpaint_idx_u] = False 
        # no change happens to the scene
        return scene,frame
    
    def __call__(self,scene,frame):
        if len(scene.frames) < 1:return scene,frame
        if self.keep_occlu:
            scene,frame = self._keep_(scene,frame)
        else:
            scene,frame = self._remove_(scene,frame)
        return scene,frame