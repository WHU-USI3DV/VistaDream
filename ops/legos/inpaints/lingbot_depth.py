"""LingBot-Depth adapter for VistaDream metric depth completion."""

import os
from types import MethodType

import numpy as np
import torch


DEFAULT_MODEL = "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"


def _enable_single_image_sdpa(model):
    """Run LingBot's one-item nested tensors without an xFormers dependency."""
    from mdm.model.dinov2_rgbd.layers.block import Block, NestedTensorBlock

    def forward_single(block, value):
        if isinstance(value, torch.Tensor):
            return Block.forward(block, value)
        if isinstance(value, list) and len(value) == 1:
            return [Block.forward(block, value[0])]
        raise RuntimeError(
            "LingBot native-SDPA fallback supports one image per inference call."
        )

    for module in model.encoder.backbone.modules():
        if isinstance(module, NestedTensorBlock):
            module.forward = MethodType(forward_single, module)


class LingBotDepth_Tool:
    def __init__(
        self,
        device="cpu",
        ckpt=DEFAULT_MODEL,
        resolution_level=9,
        preserve_known=True,
        offline=False,
        revision=None,
    ):
        # The upstream package falls back to native SDPA when xFormers is disabled.
        # Its pinned xFormers build is incompatible with the RTX 5090 environment.
        os.environ.setdefault("XFORMERS_DISABLED", "1")
        from mdm.model.v2 import MDMModel

        self.device = torch.device(device)
        self.resolution_level = int(resolution_level)
        self.preserve_known = bool(preserve_known)
        if not 0 <= self.resolution_level <= 9:
            raise ValueError("LingBot resolution_level must be between 0 and 9.")
        self.model = MDMModel.from_pretrained(
            ckpt,
            local_files_only=bool(offline),
            revision=revision,
        ).eval()
        _enable_single_image_sdpa(self.model)
        self.model.to(self.device)

    def to(self, device):
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    @torch.inference_mode()
    def __call__(self, inpaint_image, rendered_depth, inpaint_mask):
        image = np.asarray(inpaint_image)
        if image.ndim != 3 or image.shape[-1] < 3:
            raise ValueError(f"Expected an HxWx3 RGB image, got {image.shape}.")
        image = image[..., :3].astype(np.float32, copy=False)
        if image.max(initial=0.0) > 1.5:
            image = image / 255.0
        image = np.clip(image, 0.0, 1.0)

        depth = np.asarray(rendered_depth, dtype=np.float32).copy()
        mask = np.asarray(inpaint_mask, dtype=bool)
        if depth.shape != image.shape[:2] or mask.shape != depth.shape:
            raise ValueError(
                "LingBot inputs must share one spatial shape: "
                f"image={image.shape[:2]}, depth={depth.shape}, mask={mask.shape}."
            )

        valid_input = np.isfinite(depth) & (depth > 0.0)
        completion_mask = mask | ~valid_input
        depth_input = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth_input[completion_mask] = 0.0

        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(image))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        depth_tensor = torch.from_numpy(depth_input).to(self.device)
        output = self.model.infer(
            image_tensor,
            depth_in=depth_tensor,
            resolution_level=self.resolution_level,
            apply_mask=False,
            use_fp16=self.device.type == "cuda",
        )
        completed = output["depth"].squeeze().float().cpu().numpy()
        completed = completed.astype(np.float32, copy=False)

        invalid_output = ~np.isfinite(completed) | (completed <= 0.0)
        if np.any(invalid_output & completion_mask):
            count = int(np.count_nonzero(invalid_output & completion_mask))
            raise RuntimeError(
                f"LingBot-Depth returned {count} invalid values in completion regions."
            )
        if self.preserve_known:
            completed[~completion_mask] = depth[~completion_mask]
        return completed
