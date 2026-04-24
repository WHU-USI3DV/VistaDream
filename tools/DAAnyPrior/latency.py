import time
import warnings
import numpy as np
import torch
import glob
from prior_depth_anything import PriorDepthAnything

"""
Inference latency of KNN is dependent on the number of points. We conduct experiments with NYUv2 dataset on a single A100-SXM4-80GB,
the results are listed in https://github.com/SpatialVision/Prior-Depth-Anything/issues/4. This code estimate the latency based on the 
examples we provide and may differ a bit from the results we published.

We also provide several results of different numbers of sampled points here (seconds),
For "100":
| ===== The latency of fmde: 0.04979085922241211         |
| ===== The latency of cmde: 0.04994916915893555         |
| ===== The latency of knn: 0.006492257118225098         |

For "500":
| ===== The latency of fmde: 0.04977917671203613         |
| ===== The latency of cmde: 0.049925029277801514        |
| ===== The latency of knn: 0.0070438385009765625        |

For "1000":
| ===== The latency of fmde: 0.049816131591796875        |
| ===== The latency of cmde: 0.049947917461395264        |
| ===== The latency of knn: 0.00826352834701538          |

For "5000":
| ===== The latency of fmde: 0.04983079433441162         |
| ===== The latency of cmde: 0.04988771677017212         |
| ===== The latency of knn: 0.01604437828063965          |
"""
        
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    priorda = PriorDepthAnything(device=device)
    
    # Prepare datas.
    image_paths = glob.glob("assets/sample-*/rgb.*", recursive=False)
    depth_paths = glob.glob("assets/sample-*/gt_depth.*", recursive=False)
    assert len(image_paths) == len(depth_paths)
    if len(image_paths) == 1:
        warnings.warn(
            "The first inference will be much lower than the others. "
            "Please include more samples."
        )
    
    # Initialize the timer.
    timer = []
    priorda.timer = []
    priorda.completion.timer = []
    
    pattern = "500"
    for img, dpt in zip(image_paths, depth_paths):
        data = priorda.sampler(
            image=img, prior=dpt,
            pattern=pattern
        )
        
        rgb, prior_depth, sparse_depth = data['rgb'], data['prior_depth'], data['sparse_depth']
        cover_mask, sparse_mask = data['cover_mask'], data['sparse_mask']
        
        torch.cuda.synchronize()
        t0 = time.time()
        pred_depth = priorda(
            images=rgb, sparse_depths=sparse_depth, prior_depths=prior_depth,
            sparse_masks=sparse_mask, cover_masks=cover_mask, pattern=pattern
        )
        torch.cuda.synchronize()
        t1 = time.time()
        timer.append(t1 - t0)
        
        
    # We abandon the first two latency items.
    fmde_latency = np.mean(priorda.completion.timer[2:])
    cmde_latency = np.mean(priorda.timer[2:])
    # We attribute all latency of intermediate process to KNN-Latency.
    knn_latency = np.mean(timer[2:]) - fmde_latency - cmde_latency
        
    print("| ===== The latency of fmde: {} \t |".format(fmde_latency))
    print("| ===== The latency of cmde: {} \t |".format(cmde_latency))
    print("| ===== The latency of knn: {} \t |".format(knn_latency))