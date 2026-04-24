import cv2
import torch,PIL
import numpy as np
from glob import glob
from tqdm import tqdm

class Depth_IQA():
    
    def _eval_depth_aligned_(self,refer_dpts,render_dpts):
        absrel,sigma_1 = [],[]
        for i in range(len(refer_dpts)):
            refer = refer_dpts[i]
            render = render_dpts[i]
            render = cv2.resize(render,(refer.shape[1],refer.shape[0]),cv2.INTER_NEAREST)
            check_msk = (refer < 100) & (refer > 1e-3)
            refer = refer[check_msk]
            render = render[check_msk]
            
            # absrel
            absrel_item = np.abs(refer-render) / (refer+0.1)
            # sigma-1
            ref_render = refer/(render+1e-3)
            render_ref = render/(refer+1e-3)
            check_map = np.where(ref_render>render_ref,ref_render,render_ref)
            sigma_1_item = check_map < 1.25
            sigma_1_item = np.mean(sigma_1_item)
            
            if np.sum(check_msk)>5:
                absrel_item = np.mean(absrel_item)
                absrel_item = absrel_item if absrel_item<.5 else .5
                absrel.append(absrel_item)
                sigma_1.append(sigma_1_item)
            
        absrel = np.mean(absrel)
        sigma_1 = np.mean(sigma_1)
        result = {'absrel':absrel,'sigma':sigma_1}
        return result
    
    def _eval_depth_(self,refer_dpts,render_dpts):
        
        def _align_scale_shift_numpy(pred: np.array, target: np.array):
            mask = (target > 0) & (pred < 199)
            target_mask = target[mask]
            pred_mask = pred[mask]
            if np.sum(mask) > 10:
                scale, shift = np.polyfit(pred_mask, target_mask, deg=1)
                if scale < 0:
                    scale = np.median(target[mask]) / (np.median(pred[mask]) + 1e-8)
                    shift = 0
            else:
                scale = 1
                shift = 0
            return scale,shift
            
        # pre - alignment
        scale,shift=0,0
        for i in range(len(refer_dpts)):
            refer = refer_dpts[i]
            render = render_dpts[i]
            render = cv2.resize(render,(refer.shape[1],refer.shape[0]),cv2.INTER_NEAREST)
            check_msk = (refer < 100) & (refer > 1e-3)
            if i == 0:
                scale,shift = _align_scale_shift_numpy(render[check_msk],refer[check_msk])
            render = render * scale + shift
            render_dpts[i] = render
        
        return self._eval_depth_aligned_(refer_dpts,render_dpts)
    
    def __call__(self, scenes):
        '''
        After generate the scene, prepare the views for evaluation
        set an eval dir in the scene dir
        it contains:
            i.refer.rgb.png
            i.refer.dpt.npy
            i.render.rgb.png
            i.render.dpt.npy
        '''
        absrel,sigma = [],[]
        for scene in tqdm(scenes):
            refers,renders = [],[]
            eval_dir = f'{scene}/eval'
            N = len(glob(f'{eval_dir}/*.refer.rgb.png'))
            for i in range(N):
                refer = np.load(f'{eval_dir}/{i}.refer.dpt.npy')
                render = np.load(f'{eval_dir}/{i}.render.dpt.npy')
                # render = np.load(f'{scene}/compare/mvgenmaster/mast3r/{i}.dpt.npy')
                refers.append(refer)
                renders.append(render)
            result = self._eval_depth_(refers,renders)
            absrel.append(result['absrel'])
            sigma.append(result['sigma'])
        print('Depth Evaluation with GT...')
        print('AbsRel:',np.mean(np.array(absrel)))
        print('SigMa-1:',np.mean(np.array(sigma)))