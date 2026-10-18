# Installation

## Requirements

- Linux x86_64 and Conda/Miniforge.
- An NVIDIA GPU and driver compatible with CUDA 12.8.
- CUDA compilation support for gsplat and MoGe-3; this is included in the Conda environment.
- Adequate host RAM and disk for the SDXL, depth, segmentation and SD1.5 models.

The GPU checks use an RTX 5090 (32 GB VRAM). Smaller GPUs are not validated.
The supplied stack is Python 3.10 / PyTorch 2.7.1 / CUDA 12.8. This supports
Blackwell GPUs and avoids an incompatible upstream xFormers build. There is
no CPU-only reconstruction path.

## Create the environment

From this repository's root:

```bash
conda env create -f environment.yaml
conda activate vdplus
bash scripts/install_runtime.sh
```

The installation script installs the pinned MoGe-3, utils3d-moge, FlexGEMM,
LingBot-Depth and custom viewer. It applies the included FlexGEMM/Triton patch
and viewer controls patch. `--no-deps` keeps the main environment's compatible
versions in place: upstream MoGe's web-app dependencies and LingBot's older
Torch/xFormers constraints are not used by INTERFACE.

Configure CUDA builds once, with the environment active:

```bash
conda env config vars set -n vdplus \
  CUDA_HOME="$CONDA_PREFIX" \
  CPATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}" \
  LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" \
  MAX_JOBS=1
conda deactivate
conda activate vdplus
python scripts/check_runtime.py --render
```

`gsplat` compiles its CUDA extension on the first render and reuses the cached
build later. PyTorch detects the visible GPU's architecture automatically. For
an RTX 5090, `TORCH_CUDA_ARCH_LIST=12.0` can be set explicitly if needed. When
using several different GPUs, use the architecture list appropriate to those GPUs.

LingBot-Depth's adapter uses native PyTorch SDPA for batch-one inference. An
xFormers installation is not required. MoGe-3 uses FlexGEMM/Triton; both the
installed-source patch and adapter's runtime compatibility helpers are included.

## Download weights

```bash
bash download_weights.sh
```

This downloads the required Fooocus files to `tools/Fooocus/models/` and caches
MoGe-3, LingBot-Depth, Mask2Former, SD1.5 and LCM-LoRA through Hugging Face.
Interrupted Hub downloads use the Hub cache; Fooocus files are copied into place
atomically. Downloaded weights are ignored by Git.

For a custom model directory, update `model.paint.fooocus.ckpts` in a copy of
the YAML and use the same configuration for downloading and running:

```bash
bash download_weights.sh --config my_config.yaml
python INTERFACE.py --config my_config.yaml
```

`--fooocus-only` downloads only the Fooocus files. Optional LLaVA captioning
requires `--include-llava`, `llava` in `tools.preload`, and
`scene.outpaint.lvm_prompt: true`. It is disabled by default.

For offline operation, download the models first. You may then set
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` when launching. To populate Fooocus's
model directory from an existing Hub cache, use `bash download_weights.sh --offline`.
The prompt expansion tokenizer/config files are bundled with the runtime and
copied automatically when downloading to a custom Fooocus model directory. Two
small inherited VAE assets are downloaded from a pinned VistaDream Git revision
and checked against SHA-256 hashes.

## Run and verify

```bash
python INTERFACE.py --output-dir temp/my_scene
```

Open `http://127.0.0.1:7860`. The console prints the output directory. Use
`--output-dir` again after restarting to recover the coarse scene with **ReLoad**.
The server exposes one shared reconstruction per process.

```bash
python scripts/smoke_interface.py --image /path/to/image.png
```

This test runs the real model callbacks with fewer RGB diffusion/GS optimization
steps. MoGe-3's depth resolution and refinement remain as configured; it verifies
that MoGe-3 and LingBot-Depth are actually called. `--skip-mcs` omits only the
MCS refinement portion. The supplied smoke test takes no dependency on the
original research tree or any personal checkpoint directory.

## Troubleshooting

- **CUDA unavailable:** check `nvidia-smi`, the NVIDIA driver, and the active Conda environment.
- **CUDA extension build errors:** confirm `which nvcc`, `CUDA_HOME`, `CPATH`, and `LIBRARY_PATH` point to the active environment; use `MAX_JOBS=1`.
- **Out of memory:** close other GPU jobs, reduce `scene.input.resize_long_edge`, lower depth `resolution_level`, or reduce MCS views/iterations.
- **Checkpoint not found:** run the downloader with the same `--config` used by INTERFACE. The default Fooocus folder is repository-relative.
- **Viewer controls missing:** rerun `python scripts/patch_gradio_gs_camera.py`; it requires viewer version 0.0.3.

Research-only installation steps (Depth Pro, DAAnyPrior, VGGT, detectron2,
OneFormer and torch-cluster) are not required by this release.
