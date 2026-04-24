import torch
import numpy as np
from dataclasses import dataclass, field
from PIL import Image
from typing import Tuple
import matplotlib

@dataclass
class Arguments:
    K: int = field(default=5, metadata={"help": "K value of KNN"})
    conditioned_model_size: str = field(
        default="vitb", metadata={"help": "Size of conditioned model."})
    frozen_model_size: str = field(
        default="vitb", metadata={"help": "Size of frozen model."})
    normalize_depth: bool = field(
        default=True, metadata={"help": "Whether to normalize depth."})
    normalize_confidence: bool = field(
        default=True, metadata={"help": "Whether to normalize confidence."})
    err_condition: bool = field(
        default=True, metadata={"help": "Whether to use confidence/error as condition."})
    double_global: bool = field(
        default=False, metadata={"help": "Whether to use double globally-aligned conditions."})

    repo_name: str = field(
        default='Rain729/Prior-Depth-Anything', metadata={"help": "Name of hf-repo."})
    log_dir: str = field(
        default='output', metadata={"help": "The root path to save visualization results."})
    
    
# ******************** disparity space ********************
# Adapted from Marigold, available at https://github.com/prs-eth/Marigold
def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity

def disparity2depth(disparity, **kwargs):
    return depth2disparity(disparity, **kwargs)
# ************************* end ****************************
    
    
def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc

def colorize_depth_maps(
        depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
    ):
        """
        Colorize depth maps.
        """
        assert len(depth_map.shape) >= 2, "Invalid dimension"

        if isinstance(depth_map, torch.Tensor):
            depth = depth_map.detach().clone().squeeze().numpy()
        elif isinstance(depth_map, np.ndarray):
            depth = depth_map.copy().squeeze()
        # reshape to [ (B,) H, W ]
        if depth.ndim < 3:
            depth = depth[np.newaxis, :, :]

        # colorize
        cm = matplotlib.colormaps[cmap]
        depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
        img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
        img_colored_np = np.rollaxis(img_colored_np, 3, 1)

        if valid_mask is not None:
            if isinstance(depth_map, torch.Tensor):
                valid_mask = valid_mask.detach().numpy()
            valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
            if valid_mask.ndim < 3:
                valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
            else:
                valid_mask = valid_mask[:, np.newaxis, :, :]
            valid_mask = np.repeat(valid_mask, 3, axis=1)
            img_colored_np[~valid_mask] = 0

        if isinstance(depth_map, torch.Tensor):
            img_colored = torch.from_numpy(img_colored_np).float()
        elif isinstance(depth_map, np.ndarray):
            img_colored = img_colored_np

        return img_colored
        
def log_img(image, path, valids=None, scale=None, shift=None):
    if valids is not None:
        invalids = ~valids
        image[invalids] = 0
        
    if scale is None:
        scale, shift = image.max() - image.min(), image.min()
    
    normalized_value = (image - shift) / scale
    if "error" in path: normalized_value = 1 - normalized_value
    value_colored = colorize_depth_maps(
        normalized_value, 0, 1, cmap="Spectral"
    ).squeeze()
    
    if valids is not None:
        invalids = np.repeat(~valids[None, ...], 3, axis=0)
        value_colored[invalids] = 0
    value_colored = (value_colored * 255).astype(np.uint8)
    value_colored = Image.fromarray(chw2hwc(value_colored))
    value_colored.save(path)
