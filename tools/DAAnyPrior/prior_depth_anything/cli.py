import torch
import argparse
from . import PriorDepthAnything

def create_and_execute():
    parser = argparse.ArgumentParser(
        prog="priorda",
        description="Setting."
    )
    
    subparsers = parser.add_subparsers(
        title="Commands",
        dest="command",
        required=True,
        help="Now, only inference is available."
    )
    
    test_parser = subparsers.add_parser(
        "test",
        help="Run inference"
    )
    ## Model settings.
    test_parser.add_argument(
        "--coarse_only", default=0, type=bool,
        help="If specified True, predict without the fine stage.")
    test_parser.add_argument(
        "--frozen_model_size", default='vitb', type=str,
        help="Size of model in coarse stage.")
    test_parser.add_argument(
        "--conditioned_model_size", default='vitb', type=str,
        help="Size of model in fine stage.")
    
    ## Case settings.
    test_parser.add_argument(
        "--image_path", required=True, type=str,
        help="Path of RGB. e.g. assets/sample-1/rgb.jpg")
    
    test_parser.add_argument(
        "--prior_path", required=True, type=str,
        help="Path of Prior depth. e.g. assets/sample-1/gt_depth.png")
    
    test_parser.add_argument(
        "--geometric_path", default=None, type=str,
        help="(Optional) Path of geometric depth. e.g. asserts/sample-1/geo_depth.npy")
    
    test_parser.add_argument(
        "--pattern", default=None, type=str,
        help="(Optional) Pattern for sampling sparse depth points additionally in `prior`. If None, the prior depth is used.")
    
    test_parser.add_argument(
        "--visualize", type=int, default=1, 
        help="Whether to visualize the results.")
    test_parser.set_defaults(func=test)
    
    args = parser.parse_args()
    args.func(args)
    
    
def test(args):
    """ To test with models of different sizes, please specify `frozen_model_size` and `conditioned_model_size` """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    priorda = PriorDepthAnything(
        device=device, coarse_only=args.coarse_only,
        frozen_model_size=args.frozen_model_size,
        conditioned_model_size=args.conditioned_model_size
    ) 
    
    """
    image: 
        The path of the image (e.g., '*.jpg') or a tensor/array representing the image. 
        Shape should be [H, W, 3] with values in the range [0, 255].

    prior: 
        The path of the prior depth (e.g., '*.png') or a tensor/array representing the prior depth.
        Shape should be [H, W] with type float32.
        
    geometric (optional): 
        The path of the geometric depth (e.g., '*.png') or a tensor/array representing the geometric depth.
        Shape should be [H, W] with type float32.

    pattern (optional): 
        Pattern for sampling sparse depth points additionally in `prior`. If None, the prior depth is used.
    """
    output = priorda.infer_one_sample(
        image=args.image_path, 
        prior=args.prior_path, 
        geometric=args.geometric_path,
        pattern=args.pattern, 
        visualize=args.visualize
    )
    