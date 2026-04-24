import os
import cv2
import torch
import imageio
import matplotlib
import numpy as np
import open3d as o3d
from PIL import Image
from copy import deepcopy
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

def gen_config(cfg_path):
    return OmegaConf.load(cfg_path)

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

def dpt2xyz_torch(dpt,intrinsic):
    # get grid
    height, width = dpt.shape[0:2]
    grid_u = torch.arange(width)[None,:].repeat(height,1)
    grid_v = torch.arange(height)[:,None].repeat(1,width)
    grid = torch.concatenate([grid_u[:,:,None],grid_v[:,:,None],torch.ones_like(grid_v)[:,:,None]],axis=-1).to(dpt)
    uvz = grid * dpt[:,:,None]
    # inv intrinsic
    inv_intrinsic = torch.linalg.inv(intrinsic)
    xyz = torch.einsum(f'ab,hwb->hwa',inv_intrinsic,uvz)
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

def save_video(video,save_fn):
    video = (video * 255).astype(np.uint8)
    imageio.mimwrite(save_fn,video,fps=20)

def depth_colorize(dpt,sky_mask=None):
    cm = matplotlib.colormaps["Spectral"]
    depth = dpt_normalize(dpt,sky_mask)
    img_colored_np = cm(depth, bytes=False)[:, :, 0:3]  # value from 0 to 1
    return img_colored_np

def dpt_normalize(dpt, sky_mask = None):
    if sky_mask is not None:
        pic = dpt[~sky_mask]
    else:
        pic = dpt
    vmin = np.percentile(pic, 2)
    vmax = np.percentile(pic, 98)
    dpt = (deepcopy(dpt) - vmin) / (vmax - vmin)
    if sky_mask is not None:
        dpt[sky_mask] = 1.
    return dpt

def transform_points(pts,transform):
    h,w=transform.shape
    if h==3 and w==3:
        return pts @ transform.T
    if h==3 and w==4:
        return pts @ transform[:,:3].T + transform[:,3:].T
    elif h==4 and w==4:
        return pts @ transform[0:3,:3].T + transform[0:3,3:].T
    else: raise NotImplementedError

def get_nml_from_quant(quant):
    '''
    input N*4
    outut N*3
    follow https://arxiv.org/pdf/2404.17774
    '''
    w=quant[:,0]
    x=quant[:,1]
    y=quant[:,2]
    z=quant[:,3]
    n0 = 2*x*z+2*y*w
    n1 = 2*y*z-2*x*w
    n2 = 1-2*x*x-2*y*y
    nml = torch.cat((n0[:,None],n1[:,None],n2[:,None]),dim=1)
    return nml

def quaternion_from_matrix(M):
    m00 = M[..., 0, 0]
    m01 = M[..., 0, 1]
    m02 = M[..., 0, 2]
    m10 = M[..., 1, 0]
    m11 = M[..., 1, 1]
    m12 = M[..., 1, 2]
    m20 = M[..., 2, 0]
    m21 = M[..., 2, 1]
    m22 = M[..., 2, 2]   
    K = torch.zeros((len(M),4,4)).to(M)
    K[:,0,0] = m00 - m11 - m22
    K[:,1,0] = m01 + m10
    K[:,1,1] = m11 - m00 - m22
    K[:,2,0] = m02 + m20
    K[:,2,1] = m12 + m21
    K[:,2,2] = m22 - m00 - m11
    K[:,3,0] = m21 - m12
    K[:,3,1] = m02 - m20
    K[:,3,2] = m10 - m01
    K[:,3,3] = m00 + m11 + m22
    K = K/3
    # quaternion is eigenvector of K that corresponds to largest eigenvalue
    w, V = torch.linalg.eigh(K)
    q = V[torch.arange(len(V)),:,torch.argmax(w,dim=1)]
    q = q[:,[3, 0, 1, 2]]
    for i in range(len(q)):
        if q[i,0]<0.:
            q[i] = -q[i]
    return q

def numpy_quaternion_from_matrix(M):
    H,W = M.shape[0:2]
    M = M.reshape(-1,3,3)
    m00 = M[..., 0, 0]
    m01 = M[..., 0, 1]
    m02 = M[..., 0, 2]
    m10 = M[..., 1, 0]
    m11 = M[..., 1, 1]
    m12 = M[..., 1, 2]
    m20 = M[..., 2, 0]
    m21 = M[..., 2, 1]
    m22 = M[..., 2, 2]   
    K = np.zeros((len(M),4,4))
    K[...,0,0] = m00 - m11 - m22
    K[...,1,0] = m01 + m10
    K[...,1,1] = m11 - m00 - m22
    K[...,2,0] = m02 + m20
    K[...,2,1] = m12 + m21
    K[...,2,2] = m22 - m00 - m11
    K[...,3,0] = m21 - m12
    K[...,3,1] = m02 - m20
    K[...,3,2] = m10 - m01
    K[...,3,3] = m00 + m11 + m22
    K = K/3
    # quaternion is eigenvector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    q = V[np.arange(len(V)),:,np.argmax(w,axis=1)]
    q = q[...,[3, 0, 1, 2]]
    for i in range(len(q)):
        if q[i,0]<0.:
            q[i] = -q[i]
    q = q.reshape(H,W,4)
    return q

def numpy_normalize(input):
    input = input / (np.sqrt(np.sum(np.square(input),axis=-1,keepdims=True))+1e-5)
    return input

class suppress_stdout_stderr(object):
    '''
    Avoid terminal output of diffusion processings!
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

import torch.nn.functional as F
def nei_delta(input,pad=2):
    if not type(input) is torch.Tensor:
        input = torch.from_numpy(input.astype(np.float32))
    if len(input.shape) < 3:
        input = input[:,:,None]
    h,w,c = input.shape
    # reshape
    input = input.permute(2,0,1)[None]
    input = F.pad(input, pad=(pad,pad,pad,pad), mode='replicate')
    kernel = 2*pad + 1
    input = F.unfold(input,[kernel,kernel],padding=0)
    input = input.reshape(c,-1,h,w).permute(2,3,0,1).squeeze() # hw(3)*25
    return torch.amax(input,dim=-1),torch.amin(input,dim=-1),input

def edge_filter(metric_dpt,sky=None,times=0.1):
    sky = np.zeros_like(metric_dpt,bool) if sky is None else sky
    _max = np.percentile(metric_dpt[~sky],95)
    _min = np.percentile(metric_dpt[~sky], 5)
    _range = _max - _min
    nei_max,nei_min,_ = nei_delta(metric_dpt)
    delta = (nei_max-nei_min).numpy()
    edge = delta > times*_range
    return edge

def alpha_inpaint_mask(render_alpha,alpha_thres=0.7):
    render_alpha = render_alpha.detach().squeeze().cpu().numpy()
    paint_mask = 1. - (render_alpha > alpha_thres).astype(np.float64)
    # slightly clean mask
    kernel = np.ones((5,5),np.uint8)
    paint_mask = cv2.erode(paint_mask,kernel,iterations=1)
    paint_mask = cv2.dilate(paint_mask,kernel,iterations=3)
    paint_mask = paint_mask > 0.5
    return paint_mask

def fill_mask_with_nearest(imgs, mask):
    # mask and un-mask pixel coors
    mask_coords = np.column_stack(np.where(mask > .5))
    non_mask_coords = np.column_stack(np.where(mask < .5))
    # kd-tree on un-masked pixels
    tree = cKDTree(non_mask_coords)
    # nn search of masked pixels
    _, idxs = tree.query(mask_coords)
    # replace and fill
    for i, coord in enumerate(mask_coords):
        nearest_coord = non_mask_coords[idxs[i]]
        for img in imgs:
            img[coord[0], coord[1]] = img[nearest_coord[0], nearest_coord[1]]
    return imgs

def inpaint_tiny_holes(rgb,alpha,alpha_thres=0.9):
    ipaint_msk = alpha < alpha_thres
    output = deepcopy(rgb)
    for i in range(5):
        output = cv2.blur(output,(10,10))
        output[~ipaint_msk] = rgb[~ipaint_msk]
    # output = cv2.blur(output,(5,5))
    return output

def edge_rectify(metric_dpt,rgb,sky=None):
    edge = edge_filter(metric_dpt,sky)
    process_rgb = deepcopy(rgb)
    metric_dpt,process_rgb = fill_mask_with_nearest([metric_dpt,process_rgb],edge)
    return metric_dpt,process_rgb

from plyfile import PlyData, PlyElement
def color2feat(color):
    max_sh_degree = 3
    fused_color = (color-0.5)/0.28209479177387814
    features = np.zeros((fused_color.shape[0], 3, (max_sh_degree + 1) ** 2))
    features = torch.from_numpy(features.astype(np.float32))
    features[:, :3, 0 ] = fused_color
    features[:, 3:, 1:] = 0.0
    features_dc   = features[:,:,0:1]
    features_rest = features[:,:,1: ]
    return features_dc,features_rest

def construct_list_of_attributes(features_dc,features_rest,scale,rotation):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(features_dc.shape[1]*features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(features_rest.shape[1]*features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(scale.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l

def save_ply(scene,path):
    xyz       = torch.cat([gf.xyz.reshape(-1,3) for gf in scene.gaussian_frames],dim=0).detach().cpu().numpy()
    scale     = torch.cat([gf.scale.reshape(-1,3) for gf in scene.gaussian_frames],dim=0).detach().cpu().numpy()
    opacities = torch.cat([gf.opacity.reshape(-1) for gf in scene.gaussian_frames],dim=0)[:,None].detach().cpu().numpy()
    rotation  = torch.cat([gf.rotation.reshape(-1,4) for gf in scene.gaussian_frames],dim=0).detach().cpu().numpy()
    rgb       = torch.sigmoid(torch.cat([gf.rgb.reshape(-1,3) for gf in scene.gaussian_frames],dim=0))
    # rgb    
    features_dc, features_rest = color2feat(rgb)
    f_dc = features_dc.flatten(start_dim=1).detach().cpu().numpy()
    f_rest = features_rest.flatten(start_dim=1).detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    # save
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(features_dc,features_rest,scale,rotation)]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
    

from sklearn.neighbors import NearestNeighbors
def knn(x, K: int = 4):
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x)
    distances, _ = model.kneighbors(x)
    return distances

def load_video(video_fn):
    capturer = cv2.VideoCapture(video_fn)
    frames = []
    while True:
        ret,frame = capturer.read()
        if ret == False or frame is None: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame.astype(np.uint8))
        frames.append(frame)
    return frames
