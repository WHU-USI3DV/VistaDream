"""Load portable INTERFACE configuration; local paths are repository-relative."""
from pathlib import Path
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_cfg(cfg_path=None):
    path = Path(cfg_path) if cfg_path else PROJECT_ROOT / 'pipe/cfgs/basic.yaml'
    cfg = OmegaConf.load(path)
    if cfg.scene.reconstruction.type != 'mdi':
        raise ValueError('This release requires reconstruction.type=mdi: MoGe-3 + LingBot-Depth.')
    model_dir = Path(cfg.model.paint.fooocus.ckpts).expanduser()
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    cfg.model.paint.fooocus.ckpts = str(model_dir.resolve())
    return cfg


def merge_cfgs(cfg1, cfg2):
    return OmegaConf.merge(cfg1, cfg2)
