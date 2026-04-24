import torch
import torch.nn.functional as F
import numpy as np
from typing import Union
from PIL import Image

from . import PriorDepthAnything
from .utils import (
    depth2disparity,
    disparity2depth
)

class PriorDARefinerMetrics:
    def __init__(self, align_func=None):
        self.align_func = align_func
    
    def calc_errors(self, gt, pred):
        """Compute metrics for 'pred' compared to 'gt'

        Args:
            gt (torch.Tensor): Ground truth values
            pred (torch.Tensor): Predicted values

            gt.shape should be equal to pred.shape

        Returns:
            dict: Dictionary containing the following metrics:
                'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25
                'abs_rel': Absolute relative error
                'rmse': Root mean squared error
        """
        thresh = torch.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).float().mean()
        abs_rel = torch.mean(torch.abs(gt - pred) / gt)

        rmse = (gt - pred) ** 2
        rmse = torch.sqrt(rmse.mean())

        return {k: v.item() for k, v in dict(a1=a1, abs_rel=abs_rel, rmse=rmse).items()}
    
    # Align affine-invariant data to metric data.
    def align_depth_least_square(self, gt, aff, mask, space='depth'):
        """ The input should be in the same size [H, W] or [B, H, W] """
        assert (
            gt.shape == aff.shape == mask.shape
        ), f"{gt.shape}, {aff.shape}, {mask.shape}"
        
        if space == 'depth':
            gt_disparity = depth2disparity(gt)
            aff_disparity = depth2disparity(aff)
        elif space == 'disparity':
            gt_disparity = gt
            aff_disparity = aff
        else:
            raise ValueError("`space` should be in ['depth', 'disparity']")
        
        if len(gt_disparity.shape) == 2:
            gt_disparity = gt_disparity.unsqueeze(0)
            aff_disparity = aff_disparity.unsqueeze(0)
            mask = mask.unsqueeze(0)
            
        assert len(gt_disparity.shape) == 3
        
        aligned_disparity = self.align_func(
            sparse_disparities=gt_disparity,
            pred_disparities=aff_disparity,
            sparse_masks=mask
        )
        
        aligned_depth = disparity2depth(aligned_disparity)
        return aligned_depth.squeeze()
    
    def __call__(self, gt_depth, raw_depth, refined_depth):
        gt_mask = gt_depth > 0.0001
        
        raw_depth = self.align_depth_least_square(gt_depth, raw_depth, gt_mask)
        refined_depth = self.align_depth_least_square(gt_depth, refined_depth, gt_mask)
        
        calc_mask = gt_mask
        raw_m = self.calc_errors(gt_depth[calc_mask], raw_depth[calc_mask])
        ref_m = self.calc_errors(gt_depth[calc_mask], refined_depth[calc_mask])
        
        return raw_m, ref_m

class PriorDARefiner(PriorDepthAnything):
    def __init__(self, 
            device='cuda:0', coarse_only=False, fmde_dir=None, cmde_dir=None,
            ckpt_dir=None, frozen_model_size=None, conditioned_model_size=None):
        
        super(PriorDARefiner, self).__init__(
            device=device, coarse_only=coarse_only, 
            frozen_model_size=frozen_model_size, 
            conditioned_model_size=conditioned_model_size,
            fmde_dir=fmde_dir, cmde_dir=cmde_dir, ckpt_dir=ckpt_dir
        )
        
        self.extra_samples = '500'
        self.metrics_calculater = PriorDARefinerMetrics(align_func=self.completion.ss_completer)
        
        """
        We implement two strategies to filter out low-quality areas in depth_map,
        users can design other filtering methods if neccessary.
        NOTE: For different samples, the sampling strategy could differ for further
        performance improvement.
        """
        self.filter_noisy_depth = {
            'quantile': self.quant_sample,
            'normalization': self.norm_sample
        }
        
    def raw_refined_metrics(self, gt_depth, raw_depth, refined_depth):
        return self.metrics_calculater(gt_depth, raw_depth, refined_depth)
        
    def quant_sample(self, image, depth_map, confidence, quant):
        thres = torch.quantile(confidence, quant)
        extra_depth = depth_map * (confidence < thres).to(torch.float32)
        
        device = depth_map.device
        _, extra_sampled_mask, _ = self.sampler.get_sparse_depth(
            image=image.cpu().numpy(), 
            prior=extra_depth.cpu(), 
            pattern=self.extra_samples
        )
        extra_sampled_mask = extra_sampled_mask.to(device)
        
        sampled = depth_map * ((confidence > thres) | extra_sampled_mask)
        return sampled
    
    def norm_sample(self, image, depth_map, confidence, thres):
        norm_conf = (confidence - confidence.min()) / (confidence.max() - confidence.min())
        extra_depth = depth_map * (norm_conf < thres).to(torch.float32)
        
        device = depth_map.device
        _, extra_sampled_mask, _ = self.sampler.get_sparse_depth(
            image=image.cpu().numpy(), 
            prior=extra_depth.cpu(), 
            pattern=self.extra_samples
        )
        extra_sampled_mask = extra_sampled_mask.to(device)
        
        sampled = depth_map * ((confidence > thres) | extra_sampled_mask)
        return sampled
    
    # Infer one sample once.
    @torch.no_grad()
    def predict(self, 
            image: Union[torch.Tensor, str], # [H, W, 3] in torch.uint8
            depth_map: Union[torch.Tensor, str], # [H, W] in torch.float32
            confidence: Union[torch.Tensor, str], # [H, W] in torch.float32
            thres=0.3 # The `thres` is tunable to obtain better performance.
        ): 
        """ `depth_map` and `confidence` are expected to be on the same deivce. """
        
        # We allow datas to be read locally.
        if isinstance(image, str):
            image = torch.from_numpy(np.asarray(Image.open(image)).astype(np.uint8))
        if isinstance(depth_map, str):
            depth_map = torch.from_numpy(np.asarray(Image.open(depth_map)).astype(np.float32))
        if isinstance(confidence, str):
            confidence = torch.from_numpy(np.asarray(Image.open(confidence)).astype(np.float32))
        h_me, w_me = image.shape[:2]
        
        # The input datas' shape may fit to the output of depth models.
        depth_map = F.interpolate(
            depth_map[None, None, ...], size=(h_me, w_me), mode='bilinear', align_corners=True).squeeze()
        confidence = F.interpolate(
            confidence[None, None, ...], size=(h_me, w_me), mode='bilinear', align_corners=True).squeeze()
        
        # Sample in the pred depth base on the confidence map.
        # We use the combination of two strategies here for more robust outputs.
        keep_mode = ['quantile', 'normalization']
        refineds_with_diff_mode = []
        for md in keep_mode:
            prior = self.filter_noisy_depth[md](image, depth_map, confidence, thres)
            refined = self.infer_one_sample(image=image, prior=prior, geometric=None)
            refineds_with_diff_mode.append(refined)
            
        refined_depth = torch.stack(refineds_with_diff_mode, dim=-1).mean(dim=-1)
        
        return refined_depth, depth_map # return the resized depth_map for evaluation.
    