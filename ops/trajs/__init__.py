from .rot import Rot
from .spiral import Spiral
from .wobble import Wobble
from .interp import Interp
from .spline import Spline

def _generate_trajectory(cfg, scene, nframes=None):
    method = scene.traj_type
    nframe = cfg.scene.traj.n_sample*6 if nframes is None else nframes
    if method == 'rot':
        runner = Rot(scene,nframe)
    elif method == 'wobble':
        runner = Wobble(scene,nframe)
    elif method == 'spiral':
        runner = Spiral(scene,nframe)
    elif method == 'interp':
        runner = Interp(scene,nframe)
    elif method == 'spline':
        runner = Spline(scene,nframe)
    else:
        raise TypeError('method = rot / spiral / wobble')
    return runner()