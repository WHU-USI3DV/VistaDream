import cv2
import torch
import numpy as np
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    
class VGGT_MVR():
    def __init__(self,device='cpu',ckpt=f'/mnt/proj/0_Checkpoints/12_VGGT/model.pt'):
        self.ckpt = ckpt
        self.device = device
        self._load_model_()
        self.adjusted_H,self.adjusted_W = 518,518
        
    def _load_model_(self):
        self.model = VGGT()
        state_dict = torch.load(self.ckpt)
        self.model.load_state_dict(state_dict,strict=True)
        self.model.eval().to(self.device)
    
    def to(self,device):
        self.device = device
        self.model.to(self.device)
    
    def _preprocess_images_(self,images):
        # input images is numpy at [HW3], not to be the same size or intrinsic
        # following: The function ensures width=518px while maintaining aspect ratio
        # and height is center-cropped if larger than 518px
        # Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
        self.original_size = [img.shape[0:2] for img in images]
        # reshape the images
        images = [img/255. if np.amax(img)>2 else img for img in images]
        adjust_images = [cv2.resize(img,(self.adjusted_W,self.adjusted_H))[None] for img in images]
        adjust_images = np.concatenate(adjust_images,axis=0)
        # to tensor
        adjust_images = torch.from_numpy(adjust_images.astype(np.float32)).permute(0,3,1,2)
        return adjust_images
    
    def _resize_back_(self,dpts,depth_conf,intrinsics,extrinsics):
        output_dpts,output_confs,output_intrinsics,output_extrinsics = [],[],[],[]
        for i in range(len(dpts)):
            original_H, original_W = self.original_size[i]
            # dpt
            dpt = dpts[i]
            conf = depth_conf[i]
            dpt = cv2.resize(dpt,(original_W,original_H),cv2.INTER_NEAREST)
            conf = cv2.resize(conf,(original_W,original_H),cv2.INTER_NEAREST)
            # INTRINSIC
            intrinsic = intrinsics[i]
            intrinsic[0] *= (original_W / self.adjusted_W)
            intrinsic[1] *= (original_H / self.adjusted_H)
            # EXTRINSIC
            extrinsic = np.eye(4)
            extrinsic[0:3] = extrinsics[i]
            # output
            output_dpts.append(dpt)
            output_confs.append(conf)
            output_intrinsics.append(intrinsic)
            output_extrinsics.append(extrinsic)
        return output_dpts,output_confs,output_intrinsics,output_extrinsics
            
    def __call__(self,images):
        batch = self._preprocess_images_(images)
        batch = batch.to(self.device)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = self.model(batch)
        depths = predictions["depth"]
        pose_enc = predictions['pose_enc']
        depth_conf = predictions['depth_conf']
        extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, batch.shape[-2:])    
        # depth reshape back
        b = len(images)
        depths = depths.squeeze()
        depth_conf = depth_conf.squeeze()
        intrinsics = intrinsics.squeeze()
        extrinsics = extrinsics.squeeze()
        if b<2:
            depths = depths[None]
            depth_conf = depth_conf[None]
            intrinsics = intrinsics[None]
            extrinsics = extrinsics[None]
        # to desired type
        depths = depths.detach().cpu().numpy()
        depth_conf = depth_conf.detach().cpu().numpy()
        intrinsics = intrinsics.detach().cpu().numpy()
        extrinsics = extrinsics.detach().cpu().numpy()
        depths,dptconfs,intrinsics,extrinsics = self._resize_back_(depths,depth_conf,intrinsics,extrinsics)
        # return
        return depths,dptconfs,intrinsics,extrinsics

if __name__ == '__main__':
    import open3d as o3d
    from glob import glob
    from PIL import Image
    from tqdm import tqdm
    from copy import deepcopy
    
    def save_pic(input_pic:np.array,save_fn,normalize=True):
        # avoid replace
        pic = deepcopy(input_pic).astype(np.float32)
        pic = np.nan_to_num(pic)
        if normalize:
            vmin = np.percentile(pic, 2)
            vmax = np.percentile(pic, 98)
            pic = (pic - vmin) / (vmax - vmin)
        pic = (pic * 255.0).clip(0, 255)
        if save_fn is not None:
            pic_save = Image.fromarray(pic.astype(np.uint8))
            pic_save.save(save_fn)
        return pic
    
    def dpt2xyz(dpt,intrinsic):
        # get grid
        height, width = dpt.shape[0:2]
        grid_u = np.arange(width)[None,:].repeat(height,axis=0)
        grid_v = np.arange(height)[:,None].repeat(width,axis=1)
        grid = np.concatenate([grid_u[:,:,None],grid_v[:,:,None],np.ones_like(grid_v)[:,:,None]],axis=-1)
        uvz = grid * dpt[:,:,None]
        # inv intrinsic
        inv_intrinsic = np.linalg.inv(intrinsic)
        xyz = np.einsum(f'ab,hwb->hwa',inv_intrinsic,uvz)
        return xyz
    
    def visual_pcd(xyz, color=None, normal = True):
        if hasattr(xyz,'ndim'):
            xyz_norm = np.mean(np.sqrt(np.sum(np.square(xyz),axis=1)))
            xyz = xyz / xyz_norm
            xyz = xyz.reshape(-1,3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
        else: pcd = xyz
        if color is not None:
            color = color.reshape(-1,3)
            pcd.colors = o3d.utility.Vector3dVector(color)
        if normal:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.2, 20))
        o3d.visualization.draw_geometries([pcd])
    
    scenes = glob(f'/mnt/proj/5_VistaDream/VistaDream+_v1/data/main/*')
    for scene in tqdm(scenes):
        rgbs = glob(f'{scene}/color.*')
        rgbs = [np.array(Image.open(rgb))[None][...,0:3] for rgb in rgbs]
        rgbs = np.concatenate(rgbs,axis=0)
        tool = VGGT_MVR(device='cuda')
        dpts,confs,intrinsics,extrinsics = tool(rgbs)
        xyz = dpt2xyz(dpts[0],intrinsics[0])
        visual_pcd(xyz,rgbs[0]/255.)
    