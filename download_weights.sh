#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# Fooocus basic model
wget -P tools/Fooocus/models/checkpoints/ https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_v8Rundiffusion.safetensors 
# Fooocus lora model
wget -P tools/Fooocus/models/loras/ https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors 
# Fooocus inpaint model
wget -P tools/Fooocus/models/inpaint/ https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch?download=true
mv tools/Fooocus/models/inpaint/inpaint_v26.fooocus.patch?download=true tools/Fooocus/models/inpaint/inpaint_v26.fooocus.patch
# Fooocus Prompt-Extension
wget -P tools/Fooocus/models/prompt_expansion/fooocus_expansion/ https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin?download=true
mv tools/Fooocus/models/prompt_expansion/fooocus_expansion/fooocus_expansion.bin?download=true tools/Fooocus/models/prompt_expansion/fooocus_expansion/pytorch_model.bin

# Depth Pro
wget -P tools/DepthPro/checkpoints https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt 
# Depth Prior
wget -P tools/DAAnyPrior/checkpoints https://huggingface.co/Rain729/Prior-Depth-Anything/resolve/main/depth_anything_v2_vitb.pth?download=true
wget -P tools/DAAnyPrior/checkpoints https://huggingface.co/Rain729/Prior-Depth-Anything/resolve/main/prior_depth_anything_vitb.pth?download=true

# stable diffusion - lcm
# original SD1.5 ckpt will be automatically downloaded from https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
wget -P tools/StableDiffusion/lcm_ckpt https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors