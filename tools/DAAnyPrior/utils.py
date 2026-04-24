import torch
import numpy as np
import open3d as o3d
from copy import deepcopy
from omegaconf import OmegaConf

def get_pca_color(input, brightness=1.25, center=True):
    if input.ndim == 4:
        b,c,h,w = input.shape
        feat = deepcopy(input).permute(0,2,3,1).reshape(b,h*w,c)
    else:
        feat = deepcopy(input)
    # feature should be the dimension of bnc
    u, s, v = torch.pca_lowrank(feat, center=center, niter=5)
    projection = feat @ v
    projection = projection[..., :3] * 0.6 + projection[..., 3:6] * 0.4
    min_val = projection.min(dim=-2, keepdim=True)[0]
    max_val = projection.max(dim=-2, keepdim=True)[0]
    div = torch.clamp(max_val - min_val, min=1e-6)
    color = (projection - min_val) / div * brightness
    color = color.clamp(0.0, 1.0)
    # reshape back
    if input.ndim == 4:
        b,c,h,w = input.shape
        color = color.permute(0,2,1).reshape(b,-1,h,w)
    return color

def get_config(cfg_path):
    return OmegaConf.load(cfg_path)

class nn_match():
    def __init__(self) -> None:
        pass

    def pdist(self, A, B, dist_type='L2'):
          if dist_type == 'L2':
              D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
              return torch.sqrt(D2 + 1e-7)
          elif dist_type == 'SquareL2':
              return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
          else:
              raise NotImplementedError('Not implemented')

    def find_nn_gpu(self, F0, F1, nn_max_n=-1, return_distance=False, dist_type='SquareL2'):
        # F0 and F1 should be tensor
        # Too much memory if F0 or F1 large. Divide the F0
        if nn_max_n > 1:
            N = len(F0)
            C = int(np.ceil(N / nn_max_n))
            stride = nn_max_n
            dists, inds = [], []
            for i in range(C):
                dist = self.pdist(F0[i * stride:(i + 1) * stride], F1, dist_type=dist_type)
                min_dist, ind = dist.min(dim=1)
                dists.append(min_dist.detach().unsqueeze(1).cpu())
                inds.append(ind.cpu())

            if C * stride < N:
                dist = self.pdist(F0[C * stride:], F1, dist_type=dist_type)
                min_dist, ind = dist.min(dim=1)
                dists.append(min_dist.detach().unsqueeze(1).cpu())
                inds.append(ind.cpu())

            dists = torch.cat(dists)
            inds = torch.cat(inds)
            assert len(inds) == N
        else:
            dist = self.pdist(F0, F1, dist_type=dist_type)
            min_dist, inds = dist.min(dim=1)
            dists = min_dist.detach().unsqueeze(1).cpu()
            inds = inds.cpu()
        if return_distance:
            return inds, dists
        else:
            return inds
        
    def find_corr(self, F0, F1, subsample_size=-1, mutual = True, nn_max_n = 500):
        #init
        # F0 and F1 should be tensor
        inds0, inds1 = np.arange(F0.shape[0]), np.arange(F1.shape[0])
        if subsample_size > 0:
            N0 = min(len(F0), subsample_size)
            N1 = min(len(F1), subsample_size)
            inds0 = np.random.choice(len(F0), N0, replace=False)
            inds1 = np.random.choice(len(F1), N1, replace=False)
            F0 = F0[inds0]
            F1 = F1[inds1]
        # Compute the nn
        nn_inds_in1 = self.find_nn_gpu(F0, F1, nn_max_n=nn_max_n)
        if not mutual:
          inds1 = inds1[nn_inds_in1]
        else:
          matches = []
          nn_inds_in0 = self.find_nn_gpu(F1, F0, nn_max_n=nn_max_n)
          for i in range(len(nn_inds_in1)):
              if i == nn_inds_in0[nn_inds_in1[i]]:
                matches.append((i, nn_inds_in1[i]))
          matches = np.array(matches).astype(np.int32)
          inds0 = inds0[matches[:,0]]
          inds1 = inds1[matches[:,1]]
        return inds0, inds1

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

def visual_pcds(xyzs, normal = True):
    pcds = []
    for xyz in xyzs:
        if hasattr(xyz,'ndim'):
            # xyz_norm = np.mean(np.sqrt(np.sum(np.square(xyz),axis=1)))
            # xyz = xyz / xyz_norm
            xyz = xyz.reshape(-1,3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.paint_uniform_color(np.random.rand(3))
        else: pcd = xyz
        if normal:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.2, 20))
        pcds.append(pcd)
    o3d.visualization.draw_geometries(pcds)

def to_cuda(data):
    if type(data)==list:
        results = []
        for i, item in enumerate(data):
            if type(item).__name__ == "Tensor":
                results.append(item.cuda())
            elif type(item).__name__ == 'list':
                tensor_list = []
                for tensor in item:
                    tensor_list.append(tensor.cuda())
                results.append(tensor_list)
            else:
                raise NotImplementedError
        return results
    elif type(data)==dict:
        results={}
        for k,v in data.items():
            if type(v).__name__ == "Tensor":
                results[k]=v.cuda()
            else:
                results[k] = to_cuda(v)
            # elif type(v).__name__ == 'list':
            #     tensor_list = []
            #     for tensor in v:
            #         tensor_list.append(tensor.cuda())
            #     results[k]=tensor_list
            # else:
            #     raise NotImplementedError
        return results
    else:
        raise NotImplementedError

def get_focal_from_fov(new_fov, H, W):
    # NOTE: top-left pixel should be (0,0)
    if W >= H:
        f = (W / 2.0) / np.tan(np.deg2rad(new_fov / 2.0))
    else:
        f = (H / 2.0) / np.tan(np.deg2rad(new_fov / 2.0))
    return f

def get_intrins_from_fov(new_fov, H, W):
    # NOTE: top-left pixel should be (0,0)
    f = get_focal_from_fov(new_fov,H,W)

    new_cu = (W / 2.0) - 0.5
    new_cv = (H / 2.0) - 0.5

    new_intrins = np.array([
        [f,         0,     new_cu  ],
        [0,         f,     new_cv  ],
        [0,         0,     1       ]
    ])

    return new_intrins

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