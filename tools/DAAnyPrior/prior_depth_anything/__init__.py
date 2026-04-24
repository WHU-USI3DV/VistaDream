import os
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from huggingface_hub import hf_hub_download
from datetime import datetime
from PIL import Image
import glob
from typing import Union
import time

from .depth_anything_v2 import build_backbone
from .depth_completion import DepthCompletion
from .sparse_sampler import SparseSampler
from .utils import (
    log_img,
    depth2disparity, 
    disparity2depth,
    Arguments
)

class PriorDepthAnything(nn.Module):
    def __init__(self, 
            device='cuda:0', 
            fmde_dir=None,
            cmde_dir=None,
            ckpt_dir=None,
            frozen_model_size=None, 
            conditioned_model_size=None,
            coarse_only=False
        ):
        super(PriorDepthAnything, self).__init__()
        
        self.args = Arguments()
        self.device = device
        """ 
        For inference stability, we set the output coarse/fine globally. 
        TODO : You can easily modify the code to specify the model to output coarse/fine depth sample-wisely.
        """
        self.coarse_only = coarse_only
        if frozen_model_size:
            self.args.frozen_model_size = frozen_model_size
        if conditioned_model_size:
            self.args.conditioned_model_size = conditioned_model_size
        
        ## Frozon MDE loading.
        if self.args.frozen_model_size in ['vitg']:
            raise ValueError(f'{self.args.frozen_model_size} coming soon...')
        fmde_name = f'depth_anything_v2_{self.args.frozen_model_size}.pth' # Download model checkpoints
        if fmde_dir is None:
            fmde_path = hf_hub_download(repo_id=self.args.repo_name, filename=fmde_name)
        else:
            fmde_path = os.path.join(fmde_dir, fmde_name)
        print(f"Loading pretrained fmde from {fmde_path}...")
        
        # Initialize Frozon-MDE.
        self.completion = DepthCompletion.build(args=self.args, fmde_path=fmde_path, device=device)
        
        ## Conditioned MDE loading.
        if not coarse_only:
            if self.args.conditioned_model_size in ['vitl', 'vitg']:
                raise ValueError(f'{self.args.conditioned_model_size} coming soon...')
            cmde_name = f'depth_anything_v2_{self.args.conditioned_model_size}.pth' # Download model checkpoints
            if cmde_dir is None:
                cmde_path = hf_hub_download(repo_id=self.args.repo_name, filename=cmde_name)
            else:
                cmde_path = os.path.join(cmde_dir, cmde_name)
            print(f"Loading pretrained cmde from {cmde_path}...")
        
            # Initialize and load preptrained `prior-depth-anything` models. JUST FOR INITIALIZATION
            model = build_backbone(
                depth_size=self.args.conditioned_model_size, 
                encoder_cond_dim=3, model_path=cmde_path
            ).eval()
            self.model = self.load_checkpoints(model, ckpt_dir, self.device)
            
        self.sampler = SparseSampler(device=device)
    
    def load_checkpoints(self, model, ckpt_dir, device='cuda:0'):
        ckpt_name = f'prior_depth_anything_{self.args.conditioned_model_size}.pth'
        if ckpt_dir is None:
            ckpt_path = hf_hub_download(repo_id=self.args.repo_name, filename=ckpt_name)
        else:
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        print(f"Loading checkpoint from {ckpt_path}...")
        
        state_dict = torch.load(ckpt_path, map_location='cpu')
        
        new_state_dict = OrderedDict()
        for key, value in state_dict['model'].items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict)
        model = model.to(device)
        return model
        
    def forward(self, images, sparse_depths, sparse_masks, cover_masks=None, prior_depths=None, geometric_depths=None, pattern=None):
        """ To facilitate further research, we batchify the forward process. """
        ##### Coarse stage. #####
        completed_maps = self.completion(
            images=images, sparse_depths=sparse_depths, 
            sparse_masks=sparse_masks, cover_masks=cover_masks, 
            prior_depths=prior_depths, pattern=pattern,
            geometric_depths=geometric_depths
        )
        
        # knn-aligned depths
        comp_cond = completed_maps['scaled_preds'].unsqueeze(1)
        if self.coarse_only:
            coarse_depths = disparity2depth(comp_cond)
            return coarse_depths
        # Global Scale-Shift aligned depths.
        global_cond = completed_maps['global_preds'].unsqueeze(1)
        
        ##### Fine stage. #####
        if self.args.normalize_depth:
            # Obtain the value of norm params.
            masked_min, denom = self.zero_one_normalize(sparse_depths, sparse_masks, affine_only=True)
            
            global_depths = (disparity2depth(global_cond) - masked_min) / denom
            global_cond = depth2disparity(global_depths)
            
            comp_depths = (disparity2depth(comp_cond) - masked_min) / denom
            comp_cond = depth2disparity(comp_depths)
        condition = torch.cat([global_cond, comp_cond], dim=1)
        
        if self.args.err_condition:
            uctns = completed_maps['uncertainties'].unsqueeze(1)
            condition = torch.cat([uctns, condition], dim=1)
            
        # heit = sparse_depths.shape[-2] // 14 * 14
        heit = 518
        if hasattr(self, "timer"):
            torch.cuda.synchronize()
            t0 = time.time()
        metric_disparities = self.model(images, heit, condition=condition, device=self.device)
        if hasattr(self, "timer"):
            torch.cuda.synchronize()
            t1 = time.time()
            self.timer.append(t1 - t0)
            
        metric_depths = disparity2depth(metric_disparities)
        if self.args.normalize_depth:
            metric_depths = metric_depths * denom + masked_min
        return metric_depths
    
    def zero_one_normalize(self, depth_maps, valid_masks=None, affine_only=False):
        
        if valid_masks is not None:
            masked_min = depth_maps.masked_fill(~valid_masks, float('inf')).min(dim=-1).values.min(dim=-1).values  # (B, 1)
            masked_max = depth_maps.masked_fill(~valid_masks, float('-inf')).max(dim=-1).values.max(dim=-1).values  # (B, 1)
        else:
            masked_min = depth_maps.min(dim=-1).values.min(dim=-1).values  # (B, 1)
            masked_max = depth_maps.max(dim=-1).values.max(dim=-1).values  # (B, 1)
        
        denom = masked_max - masked_min
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        masked_min = masked_min.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        denom = denom.view(-1, 1, 1, 1)
        
        if not affine_only:
            normalized = (depth_maps - masked_min) / denom
            return normalized, (masked_min, denom)
        else:
            return masked_min, denom
    
    def analyze_results(self, prior_depth, pred_depth, sparse_depth, log_dir, dir_name):
        """
        We visualize depth prior here, (gt_depth or prior_depth may be stored in uint16). 
            1. If there is ground-truth depth, we visualize ground-truth. 
            2. If the provided depth map is sampled depth prior, 
                we visualize the prior depth map.
        """
        
        print("Saving visual results...")
        target = prior_depth
        scale, shift = prior_depth.max() - prior_depth.min(), prior_depth.min()
        
        gt_name = os.path.join(dir_name, 'gt_depth.*')
        gt_path = glob.glob(gt_name, recursive=False)
        if gt_path:
            gt_path = gt_path[0]
            if os.path.exists(gt_path):
                if gt_path.split('.')[-1] in ['png', 'jpg']:
                    gt_depth = np.asarray(Image.open(gt_path)).astype(np.float32)
                elif gt_path.endswith('npy'):
                    gt_depth = np.load(gt_path)
                else:
                    raise NotImplementedError
                
                target = gt_depth
                scale, shift = gt_depth.max() - gt_depth.min(), gt_depth.min()
                
                log_img(
                    gt_depth.squeeze(),
                    os.path.join(log_dir, 'gt_norm.png'),
                    valids = gt_depth > 0.0001,
                    scale=scale, shift=shift
                )
        
        prior_depth = prior_depth.squeeze().cpu().numpy()
        log_img(
            prior_depth,
            os.path.join(log_dir, 'prior_norm.png'),
            valids=prior_depth > 0.0001,
            scale=scale, shift=shift
        )
        
        log_img(
            pred_depth.squeeze().cpu().numpy(),
            os.path.join(log_dir, 'pred_depth.png'), 
            scale=scale, shift=shift
        )
        
        sparse_depth = sparse_depth.squeeze().cpu().numpy()
        log_img(
            sparse_depth,
            os.path.join(log_dir, 'sparse_depth.png'),
            valids = sparse_depth > 0.0001,
            scale=scale, shift=shift
        )
        
    @torch.no_grad()
    def infer_one_sample(self, 
        image: Union[str, torch.Tensor, np.ndarray] = None, 
        prior: Union[str, torch.Tensor, np.ndarray] = None, 
        geometric: Union[str, torch.Tensor, np.ndarray] = None,
        pattern: str = None, double_global=False, 
        prior_cover=False, visualize=False
    ) -> torch.Tensor:
        """ Perform inference. Return the refined/completed depth.
        
        Args:
            image: 
                1. RGB in 'np.ndarray' or 'torch.Tensor' [H, W]
                2. Image path of RGB
            prior:
                1. Prior depth in 'np.ndarray' or 'torch.Tensor' [H, W]
                2. Path of prior depth map. (with scale)
            geometric:
                1. Geometric depth in 'np.ndarray' or 'torch.Tensor' [H, W]
                2. Path of geometric depth map. (with geometry)
            pattern: The mode of prior-based additional sampling. It could be None.
            double_global: Whether to condition with two estimated depths or estimated + knn-map.
            prior_cover: Whether to keep all prior areas in knn-map, it functions when 'pattern' is not None.
            visualize: Save results. 
            
            
            Example1:
                >>> import torch
                >>> from prior_depth_anything import PriorDepthAnything
                >>> device = "cuda" if torch.cuda.is_available() else "cpu"
                >>> priorda = PriorDepthAnything(device=device)
                >>> image_path = 'assets/sample-2/rgb.jpg'
                >>> prior_path = 'assets/sample-2/prior_depth.png'
                >>> output = priorda.infer_one_sample(image=image_path, prior=prior_path, visualize=True)
                
            Example2:
                >>> import torch
                >>> from prior_depth_anything import PriorDepthAnything
                >>> device = "cuda" if torch.cuda.is_available() else "cpu"
                >>> priorda = PriorDepthAnything(device=device)
                >>> image_path = 'assets/sample-6/rgb.npy'
                >>> prior_path = 'assets/sample-6/prior_depth.npy'
                >>> output = priorda.infer_one_sample(image=image_path, prior=prior_path, visualize=True)
        """
        
        # For each inference, params below should be reset.
        self.args.double_global = double_global
        assert image is not None and prior is not None
        
        ### Load and preprocess example images
        # We implement preprocess with batch size of 1, but our model works for multi-images naturally.
        data = self.sampler(
            image=image, prior=prior,
            geometric=geometric,
            pattern=pattern, K=self.args.K,
            prior_cover=prior_cover
        )
        rgb, prior_depth, sparse_depth = data['rgb'], data['prior_depth'], data['sparse_depth'] # Shape: [B, C, H, W]
        cover_mask, sparse_mask = data['cover_mask'], data['sparse_mask'] # Shape: [B, 1, H, W]
        geometric_depth = data['geometric_depth'] if geometric is not None else None
        if (sparse_mask.view(sparse_mask.shape[0], -1).sum(dim=1) < self.args.K).any():
            raise ValueError("There are not enough known points in at least one of samples")

        ### The core inference stage.
        """ If you want to input multiple samples at once, just stack samples at dim=0, s.t. [B, C, H, W] """
        pred_depth = self.forward(
            images=rgb, sparse_depths=sparse_depth, prior_depths=prior_depth,
            sparse_masks=sparse_mask, cover_masks=cover_mask, pattern=pattern,
            geometric_depths=geometric_depth
        ) # (B, 1, H, W)
        
        ### Visualize the results.
        if visualize:
            # 'dir_name' is the path that stores Ground-Truth RGB and depth.
            if isinstance(image, str):
                dir_name = os.path.dirname(image)
            else:
                dir_name = '.'
                
            parent = datetime.now().strftime("%Y-%m-%d %H:%M")
            log_dir = os.path.join(self.args.log_dir, parent)
            os.makedirs(log_dir, exist_ok=True)
            self.analyze_results(prior_depth, pred_depth, sparse_depth, log_dir, dir_name)
        
        return pred_depth.squeeze()