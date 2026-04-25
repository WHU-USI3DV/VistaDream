from pipe.cfgs import load_cfg
from pipe.vistadream import Pipeline

cfg = load_cfg(f'pipe/cfgs/basic.yaml')
cfg.scene.input.rgb = 'data/sd_readingroom/color.png'
vistadream = Pipeline(cfg)
vistadream()
