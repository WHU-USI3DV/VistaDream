"""Check INTERFACE imports and optionally execute gsplat's CUDA rasterizer."""
import argparse
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', action='store_true', help='Compile/run gsplat and check backward gradients')
    args = parser.parse_args()
    for module in ('torch', 'gsplat', 'moge.model.v3', 'mdm.model.v2', 'diffusers',
                   'transformers', 'gradio_model3dgscamera', 'pipe.vistadream_interface'):
        importlib.import_module(module)
        print(f'Import OK: {module}')
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable. Check your NVIDIA driver and GPU visibility.')
    print(f'PyTorch {torch.__version__}; CUDA {torch.version.cuda}; GPU {torch.cuda.get_device_name(0)}')
    if args.render:
        import gsplat
        means = torch.tensor([[0., 0., 2.]], device='cuda', requires_grad=True)
        rgb, alpha, _ = gsplat.rasterization(
            means=means,
            quats=torch.tensor([[1., 0., 0., 0.]], device='cuda'),
            scales=torch.full((1, 3), .1, device='cuda'),
            opacities=torch.ones(1, device='cuda'),
            colors=torch.tensor([[1., 0., 0.]], device='cuda'),
            viewmats=torch.eye(4, device='cuda')[None],
            Ks=torch.tensor([[[32., 0., 16.], [0., 32., 16.], [0., 0., 1.]]], device='cuda'),
            width=32, height=32)
        assert torch.isfinite(rgb).all() and alpha.max() > 0
        rgb.sum().backward()
        assert means.grad is not None and torch.isfinite(means.grad).all()
        print('gsplat CUDA forward/backward: OK')
    print('INTERFACE runtime: OK')


if __name__ == '__main__':
    main()
