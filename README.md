# VistaDreamer — Interactive Reconstruction

> **VistaDreamer:  Zero-shot Single-view Scene Reconstruction by Multi-view Consistency Sampling**<br/>
> [Haiping Wang](https://hpwang-whu.github.io/), [Yuan Liu](https://liuyuan-pal.github.io/), [Ziwei Liu](https://liuziwei7.github.io/), [Yulan Guo](https://www.yulanguo.cn/), [Yu-Shen Liu](https://yushen-liu.github.io/), [Wenping Wang](https://www.cs.hku.hk/people/academic-staff/wenping), [Zhen Dong](https://dongzhenwhu.github.io/index.html), [Bisheng Yang](https://3s.whu.edu.cn/info/1025/1415.htm)<br/>
> *TPAMI 2026*<br/>


An INTERFACE-only release of VistaDreamer.

This release reconstructs and refines a Gaussian scene from one image through a
Gradio interface. **MoGe-3** estimates the scaffold's metric depth and camera
intrinsics; **LingBot-Depth** completes rendered depth for newly inpainted views,
preserving valid known depth. Fooocus handles RGB outpainting/inpainting,
Mask2Former segments sky, and Stable Diffusion 1.5 with LCM-LoRA provides MCS
refinement. This is an updated depth backend, rather than the original paper's
exact model configuration.

## Work with Agents
To avoid issues caused by installation errors or version conflicts, we strongly recommend using an agent such as Codex to handle the environment setup process. Likewise, if you have previously installed VistaDream, we recommend asking the agent to replace the depth estimation module with MoGe3 and the depth completion module with Lingbot-Depth, which can significantly improve performance.

## Install and launch

Linux x86_64, an NVIDIA CUDA GPU (tested on 4090 and 5090), and Conda/Miniforge are required. The supplied
environment uses Python 3.10, PyTorch 2.7.1 and CUDA 12.8. See
[INSTALL.md](INSTALL.md) for compiler setup and verification.

```bash
conda env create -f environment.yaml
conda activate vdplus
bash scripts/install_runtime.sh
bash download_weights.sh
python INTERFACE.py
```

Open `http://127.0.0.1:7860`. Models load on the first reconstruction, so launching
the interface alone does not download weights. The depth model code and weights
are pinned to specific revisions. LLaVA captioning is disabled by default; provide
a scene prompt or leave it blank for a default prompt.

## Workflow

1. Upload an image, enter an optional scene prompt, set the seed, and click **Build Scaffold**.
2. Move the camera in **GS-Viewer**. Click the viewer to focus it; drag to orbit,
   right/middle-drag to pan, and scroll to zoom. Use WASD to move, Space/C for
   vertical movement, and Shift to accelerate.
3. Use **Preview the view**, then **Inpaint the view** to add it. Keep the anchor
   checkbox enabled for camera views you want in the final trajectory.
   **Re-select the view** undoes the last added frame.
4. Select a few anchor views, then click **Generate a coarse scene** to fill the
   trajectory. Click **Refine the scene by MCS** for optional refinement.
5. Use **Rendering RGB and Depth videos** to export the current saved result.

**End-to-end VistaDream** builds a scene automatically along a Spiral trajectory,
then refines it with MCS and exports the videos. MCS can take substantial time;
the view count, sampling steps, and Gaussian optimization iterations are editable
in the interface.

Each server process owns one shared scene; operate one reconstruction at a time.
All callbacks share the same queue lock. Run separate processes/output directories
for independent sessions.

## Configuration and outputs

Edit [pipe/cfgs/basic.yaml](pipe/cfgs/basic.yaml), or pass a copy with `--config`.
The `mdi` backend uses **both MoGe-3 and LingBot-Depth** and is required in this
release. Fooocus's model directory is resolved relative to the repository root;
Hugging Face model IDs can be replaced with local checkpoints. For MoGe-3 and
LingBot-Depth, a local checkpoint must be the actual `model.pt` file.

```bash
python INTERFACE.py --output-dir temp/my_scene --traj-type interp
# Restart using the same directory, then click ReLoad:
python INTERFACE.py --output-dir temp/my_scene
# Network binding is opt-in:
python INTERFACE.py --host 0.0.0.0 --port 7860
```

Without `--output-dir`, each launch creates `temp/<timestamp>/`. Files include:

| File | Contents |
| --- | --- |
| `color.png` | Input image |
| `scene.coarse.pth` | Coarse scene and configuration |
| `scene.refine.pth` | MCS-refined scene |
| `temp.splat` | Downsampled browser preview |
| `temp.video_rgb.mp4`, `temp.video_dpt.mp4` | Rendered videos |

**ReSet** clears the in-memory scene and viewer; saved files remain available to
**ReLoad**, which restores the coarse scene. Adding/undoing a view or regenerating
the coarse scene invalidates older refined results. The scene format serializes
Python objects; use your own saved scenes with this version of the code.

