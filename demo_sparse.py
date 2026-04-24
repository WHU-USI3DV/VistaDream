from glob import glob
from pipe.cfgs import load_cfg
from pipe.vistadream_sparse import Pipeline_Sparse

base = f'data/bedroom'
images = glob(f'{base}/*.png')

cfg = load_cfg(f'pipe/cfgs/basic.yaml')
cfg.scene.input.rgbs = images
vistadream = Pipeline_Sparse(cfg)
vistadream()
