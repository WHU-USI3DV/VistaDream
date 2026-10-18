"""MoGe-3 adapter for monocular metric depth and camera intrinsics."""

import os
import sys
from pathlib import Path

import numpy as np
import torch


DEFAULT_MODEL = "Ruicheng/moge-3-vitl"


def _configure_triton_compiler():
    bin_dir = Path(sys.prefix) / "bin"
    candidates = {
        "CC": ("x86_64-conda-linux-gnu-cc", "gcc", "cc"),
        "CXX": ("x86_64-conda-linux-gnu-c++", "g++", "c++"),
    }
    for variable, names in candidates.items():
        if os.environ.get(variable):
            continue
        for name in names:
            executable = bin_dir / name
            if executable.exists():
                os.environ[variable] = str(executable)
                break


def _configure_triton_compatibility():
    # FlexGEMM's pinned Triton kernels use the pre-3.3 dtype.itemsize API.
    import triton.language as tl
    from triton.runtime.jit import KernelParam

    if not hasattr(tl.dtype, "itemsize"):
        tl.dtype.itemsize = property(lambda value: value.primitive_bitwidth // 8)
    if not getattr(KernelParam, "_vistadream_annotation_fix", False):
        def annotation_type(parameter):
            annotation = parameter.annotation
            for prefix, shorthand in (("uint", "u"), ("int", "i")):
                if annotation.startswith(prefix):
                    width = annotation[len(prefix):]
                    if width.isdigit():
                        return f"{shorthand}{width}"
            return "u1" if annotation == "bool" else ""

        KernelParam.annotation_type = property(annotation_type)
        KernelParam._vistadream_annotation_fix = True


class MoGe3_Tool:
    def __init__(
        self,
        device="cuda",
        ckpt=DEFAULT_MODEL,
        revision=None,
        resolution_level=9,
        refine_steps=3,
        use_fp16=True,
    ):
        _configure_triton_compiler()
        _configure_triton_compatibility()
        from moge.model.v3 import MoGeModel

        self.device = torch.device(device)
        self.resolution_level = int(resolution_level)
        self.refine_steps = int(refine_steps)
        self.use_fp16 = bool(use_fp16)
        if not 0 <= self.resolution_level <= 9:
            raise ValueError("MoGe-3 resolution_level must be between 0 and 9.")
        if self.refine_steps < 0:
            raise ValueError("MoGe-3 refine_steps cannot be negative.")
        self.model = MoGeModel.from_pretrained(
            ckpt,
            revision=revision,
        ).eval().to(self.device)

    def to(self, device):
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    @staticmethod
    def _pixel_intrinsics(normalized_intrinsics, height, width):
        intrinsic = normalized_intrinsics.astype(np.float32, copy=True)
        intrinsic[0] *= width
        intrinsic[1] *= height
        return intrinsic

    @torch.inference_mode()
    def __call__(self, image, f_px=None):
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] < 3:
            raise ValueError(f"Expected an HxWx3 RGB image, got {image.shape}.")
        image = image[..., :3].astype(np.float32, copy=False)
        if image.max(initial=0.0) > 1.5:
            image = image / 255.0
        image = np.clip(image, 0.0, 1.0)
        height, width = image.shape[:2]

        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(image))
            .permute(2, 0, 1)
            .to(self.device)
        )
        fov_x = None
        if f_px is not None:
            fov_x = float(np.degrees(2.0 * np.arctan(width / (2.0 * float(f_px)))))
        output = self.model.infer(
            image_tensor,
            resolution_level=self.resolution_level,
            refine_steps=self.refine_steps,
            fov_x=fov_x,
            apply_mask=False,
            use_fp16=self.use_fp16 and self.device.type == "cuda",
        )
        depth = output["depth"].float().cpu().numpy().astype(np.float32, copy=False)
        intrinsic = output["intrinsics"].float().cpu().numpy()
        intrinsic = self._pixel_intrinsics(intrinsic, height, width)

        invalid = ~np.isfinite(depth) | (depth <= 0.0)
        if np.any(invalid):
            raise RuntimeError(
                f"MoGe-3 returned {int(np.count_nonzero(invalid))} invalid depth values."
            )
        return depth, intrinsic
