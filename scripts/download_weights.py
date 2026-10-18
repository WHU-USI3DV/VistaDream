"""Download only INTERFACE weights, using the Hugging Face cache."""
import argparse
import hashlib
import urllib.request
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from huggingface_hub import hf_hub_download, snapshot_download
from pipe.cfgs import load_cfg

# repository, remote filename, path beneath Fooocus's model directory
FOOOCUS_FILES = (
    ('lllyasviel/fav_models', 'fav/juggernautXL_v8Rundiffusion.safetensors', 'checkpoints/juggernautXL_v8Rundiffusion.safetensors'),
    ('stabilityai/stable-diffusion-xl-base-1.0', 'sd_xl_offset_example-lora_1.0.safetensors', 'loras/sd_xl_offset_example-lora_1.0.safetensors'),
    ('lllyasviel/fooocus_inpaint', 'inpaint_v26.fooocus.patch', 'inpaint/inpaint_v26.fooocus.patch'),
    ('lllyasviel/fooocus_inpaint', 'fooocus_inpaint_head.pth', 'inpaint/fooocus_inpaint_head.pth'),
    ('lllyasviel/misc', 'fooocus_expansion.bin', 'prompt_expansion/fooocus_expansion/pytorch_model.bin'),
    ('lllyasviel/misc', 'fooocus_upscaler_s409985e5.bin', 'upscale_models/fooocus_upscaler_s409985e5.bin'),
    ('lllyasviel/misc', 'xlvaeapp.pth', 'vae_approx/xlvaeapp.pth'),
)


# These two inherited Fooocus assets are hosted in VistaDream, not lllyasviel/misc.
VISTADREAM_REVISION = '6b3e98f06b7c649230029641392ca4c25f26d6c9'
RAW_ASSETS = (
    ('vaeapp_sd15.pth', '4f88c9078bb2238cdd0d8864671dd33e3f42e091e41f08903f3c15e4a54a9b39'),
    ('xl-to-v1_interposer-v4.0.safetensors', 'da4c2d772c621134ba1a2d961a439cbc4b4c85c43aae1197ab3c024e2a3efa57'),
 )


def _raw_asset(name, digest, target, offline):
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
        print(f'Already present: {name}')
        return
    if offline:
        raise RuntimeError(f'Missing {target}; run the downloader online once before offline use.')
    url = f'https://raw.githubusercontent.com/WHU-USI3DV/VistaDream/{VISTADREAM_REVISION}/tools/Fooocus/models/vae_approx/{name}'
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.partial')
    with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as file:
        shutil.copyfileobj(response, file)
    if hashlib.sha256(partial.read_bytes()).hexdigest() != digest:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f'Checksum mismatch for {name}')
    partial.replace(target)
    print(f'Ready: {name}')


def download(config=None, fooocus_only=False, include_llava=False, offline=False):
    cfg = load_cfg(config)
    model_dir = Path(cfg.model.paint.fooocus.ckpts)
    tokenizer_dir = ROOT / 'tools/Fooocus/models/prompt_expansion/fooocus_expansion'
    target_tokenizer = model_dir / 'prompt_expansion/fooocus_expansion'
    target_tokenizer.mkdir(parents=True, exist_ok=True)
    for bundled in tokenizer_dir.iterdir():
        if bundled.suffix in ('.json', '.txt') and not (target_tokenizer / bundled.name).exists():
            shutil.copy2(bundled, target_tokenizer / bundled.name)
    for name, digest in RAW_ASSETS:
        _raw_asset(name, digest, model_dir / 'vae_approx' / name, offline)
    for repo, filename, relative in FOOOCUS_FILES:
        target = Path(cfg.model.paint.fooocus.ckpts) / relative
        if target.is_file() and target.stat().st_size:
            print(f'Already present: {relative}')
            continue
        cached = hf_hub_download(repo, filename, local_files_only=offline)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + '.partial')
        shutil.copyfile(cached, partial)
        partial.replace(target)
        print(f'Ready: {relative}')
    if fooocus_only:
        return
    for model in (cfg.model.dpt.moge3, cfg.model.dpt.lingbot):
        if Path(model.ckpt).exists():
            continue
        hf_hub_download(model.ckpt, 'model.pt', revision=model.get('revision'), local_files_only=offline)
    # Download one weight format and only the diffusers components used by MCS.
    snapshots = (
        (cfg.model.sky.mask2former.ckpt, ['*.json', 'model.safetensors']),
        (cfg.model.optimize.sd, ['model_index.json', 'feature_extractor/*.json',
                               'scheduler/*.json', 'tokenizer/*',
                               'safety_checker/*.json', 'safety_checker/model.safetensors',
                               'text_encoder/*.json', 'text_encoder/model.safetensors',
                               'unet/*.json', 'unet/diffusion_pytorch_model.safetensors',
                               'vae/*.json', 'vae/diffusion_pytorch_model.safetensors']),
        (cfg.model.optimize.lcm, ['pytorch_lora_weights.safetensors']),
    )
    for checkpoint, patterns in snapshots:
        if Path(checkpoint).exists():
            continue
        snapshot_download(checkpoint, local_files_only=offline, allow_patterns=patterns)
    if include_llava:
        snapshot_download(cfg.model.vlm.llava.ckpt, local_files_only=offline,
                          allow_patterns=['*.json', '*.txt', '*.safetensors', '*.model'])
    print('INTERFACE weights ready.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--fooocus-only', action='store_true')
    parser.add_argument('--include-llava', action='store_true')
    parser.add_argument('--offline', action='store_true', help='Use previously cached files only')
    args = parser.parse_args()
    download(args.config, args.fooocus_only, args.include_llava, args.offline)
