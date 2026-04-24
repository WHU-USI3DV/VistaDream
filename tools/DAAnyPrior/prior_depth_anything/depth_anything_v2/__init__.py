from .dpt import DepthAnythingV2
import torch
import os

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

def build_backbone(depth_size='vitb', encoder_cond_dim=-1, model_path=None):
    core = DepthAnythingV2(**model_configs[depth_size], encoder_cond_dim=encoder_cond_dim)

    state_dict = torch.load(model_path, map_location='cpu')
    core.init_state_dict(state_dict=state_dict)
    
    return core