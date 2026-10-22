# A VistaDreamer Interface for Interactive Reconstruction

> **VistaDreamer:  Zero-shot Single-view Scene Reconstruction by Multi-view Consistency Sampling**<br/>
> [Haiping Wang](https://hpwang-whu.github.io/), [Yuan Liu](https://liuyuan-pal.github.io/), [Ziwei Liu](https://liuziwei7.github.io/), [Yulan Guo](https://www.yulanguo.cn/), [Yu-Shen Liu](https://yushen-liu.github.io/), [Wenping Wang](https://www.cs.hku.hk/people/academic-staff/wenping), [Zhen Dong](https://dongzhenwhu.github.io/index.html), [Bisheng Yang](https://3s.whu.edu.cn/info/1025/1415.htm)<br/>
> *TPAMI 2026*<br/>

This release reconstructs and refines a Gaussian scene from one image through a
Gradio interface. **MoGe-3** estimates the scaffold's metric depth and camera
intrinsics; **LingBot-Depth** completes rendered depth for newly inpainted views,
preserving valid known depth. Fooocus handles RGB outpainting/inpainting,
Mask2Former segments sky, and Stable Diffusion 1.5 with LCM-LoRA provides MCS
refinement.

## Install with Agents
To avoid issues caused by installation errors or version conflicts, we strongly recommend using an agent such as Codex to handle the environment setup process. 
> Note: if you have previously installed VistaDream, we recommend asking the agent to replace the depth estimation module with MoGe3 and the depth completion module with Lingbot-Depth, which can significantly improve performance.

## Install by hand
> Note: Because these dependency files were generated from tests on an RTX 5090 GPU, it is still unclear whether migration will cause compatibility issues.

Linux x86_64, an NVIDIA CUDA GPU (tested on 4090 and 5090), and Conda/Miniforge are required. The supplied
environment uses Python 3.10, PyTorch 2.7.1 and CUDA 12.8. 

```bash
conda env create -f environment.yaml
conda activate vdplus
bash scripts/install_runtime.sh
bash download_weights.sh
```

## Lunch the interface

- Use ```python DESCRIBE.py --image <png/jpg file>``` to get the description of the image.

- Lunch the interface for interactive generation with 
```
python INTERFACE.py
```
- Workflow
<p align="center">
  <img src="data/assert/interface/interf.gif" alt="Interactive interface demo" width="90%">
</p>

1. Upload an image, enter the above image description, set the seed, and click **Build Scaffold**.
2. Move the camera in **GS-Viewer**. Click the viewer to focus it; drag to orbit,
   right/middle-drag to pan, and scroll to zoom. Use WASD to move, Space/C for
   vertical movement, and Shift to accelerate.
3. Use **Preview the view**, then **Inpaint the view** to add it. Keep the anchor
   checkbox enabled for camera views you want in the final trajectory.
   **Re-select the view** undoes the last added frame.
4. Select a few anchor views, then click **Generate a coarse scene** to fill the
   trajectory. Click **Refine the scene by MCS** for optional refinement.
5. Use **Rendering RGB and Depth videos** to export the current saved result.

## Configuration and outputs

Other hyperparameters can be edited in [pipe/cfgs/basic.yaml](pipe/cfgs/basic.yaml) before generating the scene.

Without `--output-dir`, each launch creates `temp/<timestamp>/`. Files include:

| File | Contents |
| --- | --- |
| `color.png` | Input image |
| `scene.coarse.pth` | Coarse scene and configuration |
| `scene.refine.pth` | MCS-refined scene |
| `temp.splat` | Downsampled browser preview |
| `temp.video_rgb.mp4`, `temp.video_dpt.mp4` | Rendered videos |
