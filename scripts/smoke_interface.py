"""Exercise real INTERFACE callbacks on a CUDA GPU (not a quality benchmark)."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--config')
    parser.add_argument('--output-dir', default=str(ROOT / 'temp' / 'smoke'))
    parser.add_argument('--skip-mcs', action='store_true')
    parser.add_argument('--coarse', action='store_true', help='Also fill the selected trajectory automatically')
    parser.add_argument('--e2e', action='store_true', help='Also exercise the automatic Spiral/MCS callback')
    args = parser.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    import INTERFACE as app
    from pipe.vistadream_interface import load_scene
    app.configure_app(args.config, args.output_dir)
    # Shorten Gaussian optimization and diffusion sampling; use configured depth
    # resolution/refinement unchanged to exercise the actual MoGe-3 kernels.
    app.cfg.scene.gaussian.opt_iters_per_frame = 16
    app.cfg.scene.mcs.gsopt_iters = 16
    app.cfg.scene.mcs.steps = 2
    app.cfg.scene.mcs.n_view = 2
    app.cfg.scene.traj.n_sample = 2
    app.pipe._ensure_tools()
    app.pipe.tools.rgb_inpaint.fooocus.tool.overwrite_step = 4
    counts = {'moge3': 0, 'lingbot': 0}
    reconstructor = app.pipe.tools.dpt_inpaint.recons_tool
    for key, tool in [('moge3', reconstructor.moge3), ('lingbot', reconstructor.lingbot)]:
        original = tool.model.infer
        def tracked(*values, _original=original, _key=key, **kwargs):
            counts[_key] += 1
            return _original(*values, **kwargs)
        tool.model.infer = tracked
    with Image.open(args.image) as image:
        app.interactive_build_scaffold(image, 12345, 'A realistic scene with coherent objects and detailed surfaces.')
    initial_count = len(app.pipe.scene.frames)
    assert counts['moge3'] == 1 and counts['lingbot'] == 0, counts
    # Move sideways by 10% of median scene depth to expose a new region.
    depth = app.pipe.scene.frames[0].dpt
    movement = float(np.median(depth[depth > 0])) * .1
    viewer = [app.SPLAT_PATH, [movement, 0., 0.], [0., 0., 0.]]
    preview = app.interactive_render_splat(viewer)
    assert preview.ndim == 3 and preview.dtype == np.uint8
    app.interactive_render_and_inpaint(viewer, 23456, True)
    assert counts['lingbot'] >= 1, counts
    assert len(app.pipe.scene.frames) == initial_count + 1
    config, saved = load_scene(app.SCENE_PATH)
    assert config.scene.outpaint.seed == 23456
    assert len(saved.frames) == initial_count + 1
    del saved
    app._interactive_step_back_()
    assert len(app.pipe.scene.frames) == initial_count
    app.interactive_render_and_inpaint(viewer, 34567, True)
    if args.coarse:
        app._Traj_Refine_(45678)
    final_coarse_frames = len(app.pipe.scene.frames)
    if not args.skip_mcs:
        app._interactive_MCS(2, 2, .5, 16)
        assert Path(app.SCENE_FINAL_PATH).is_file()
    for path in app._interactive_render_videos():
        assert Path(path).stat().st_size > 0
    # Reset clears the in-memory/UI state; reload recovers the coarse scene.
    app._RESET_()
    assert app.pipe.scene is None
    app._RELOAD_()
    assert len(app.pipe.scene.frames) == final_coarse_frames
    if args.e2e:
        with Image.open(args.image) as image:
            app._E2E_VD_(image, 'A realistic room with coherent objects and detailed surfaces.', 56789)
        assert app.pipe.scene.traj_type == 'spiral'
        assert Path(app.SCENE_FINAL_PATH).is_file()
        assert Path(app.RGB_VIDEO_PATH).stat().st_size > 0
    assert not any(name.startswith(('tools.VGGT', 'tools.DepthPro', 'tools.DAAnyPrior')) for name in sys.modules)
    assert Path(app.SPLAT_PATH).stat().st_size % 32 == 0
    assert torch.isfinite(torch.cat([gf.xyz.reshape(-1, 3) for gf in app.pipe.scene.gaussian_frames])).all()
    print(f'INTERFACE smoke passed: {counts}; output={app.TEMP_DIR}')


if __name__ == '__main__':
    main()
