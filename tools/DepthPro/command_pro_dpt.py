import torch
import depth_pro
import numpy as np
from PIL import Image
from depth_pro.depth_pro import DepthProConfig

class apple_pro_depth():
    def __init__(self,device='cuda',ckpt = '/mnt/proj/0_Checkpoints/1_DepthPro/depth_pro.pt'):
        self.ckpt = ckpt
        self.device = device
        self._load_model()
        
    def _load_model(self):
        cfg = DepthProConfig(
            patch_encoder_preset="dinov2l16_384",
            image_encoder_preset="dinov2l16_384",
            checkpoint_uri=self.ckpt,
            decoder_features=256,
            use_fov_head=True,
            fov_encoder_preset="dinov2l16_384",
        )
        self.model, self.transform = depth_pro.create_model_and_transforms(config=cfg,device=self.device)
        self.model.eval()
        
    def get_intrins(self, f, H, W):
        new_cu = (W / 2.0) - 0.5
        new_cv = (H / 2.0) - 0.5
        intrins = np.array([
            [f,         0,     new_cu  ],
            [0,         f,     new_cv  ],
            [0,         0,     1       ]
        ])
        return intrins
    
    def to(self,device):
        self.device = device
        self.model.to(device)
    
    @torch.no_grad()
    def __call__(self, image,f_px=None):
        if type(image) is np.ndarray:
            if np.amax(image) < 1.1:
                image = image*255
            image = Image.fromarray(image.astype(np.uint8))
        # trans
        image = self.transform(image).to(self.device)
        # predict
        prediction = self.model.infer(image, f_px=f_px)
        depth = prediction["depth"]  # Depth in [m].
        focallength_px = prediction["focallength_px"]  # Focal length in pixels.
        # output
        H,W = depth.shape[0:2]
        depth = depth.detach().cpu().numpy()
        focallength_px = focallength_px.detach().cpu().numpy() if f_px is None else f_px
        intrinsic = self.get_intrins(focallength_px,H,W)
        return depth, intrinsic


if __name__ == '__main__':
    import open3d as o3d
    from glob import glob
    from PIL import Image
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
    for scene in scenes[4:]:
        image = glob(f'{scene}/color.*')[0]
        rgb = np.array(Image.open(image))[...,:3]
        tool = apple_pro_depth(device='cuda')
        dpt,intrinsic = tool(rgb)
        torch.cuda.empty_cache()
        xyz = dpt2xyz(dpt,intrinsic)
        visual_pcd(xyz,rgb/255.)


