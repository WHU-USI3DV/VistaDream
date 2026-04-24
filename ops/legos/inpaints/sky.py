import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

class Sky_Tool():
    '''
    If we want to keep sky:
        set sky to max-depth-value
        these areas will be kept (add in frame.inpaint)
    if we want to remove sky:
        set sky to 0.
        these areas will be removed (remove from frame.inpaint)
    '''
    def __init__(self,cfg):
        self.cfg = cfg
        self.image_processor = AutoImageProcessor.from_pretrained(cfg.model.sky.mask2former.ckpt)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(cfg.model.sky.mask2former.ckpt)
        # set sky areas to desired value
        self.sky_keep = self.cfg.model.sky.keep
        self.sky_depth = self.cfg.model.sky.value

    def _segment_sky_(self,rgb):
        rgb = rgb*255 if np.amax(rgb) < 1.1 else rgb
        image = Image.fromarray((rgb).astype(np.uint8))
        inputs = self.image_processor(image,return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        pred_semantic_map = self.image_processor.post_process_semantic_segmentation(outputs, target_sizes=[(image.height, image.width)])[0]
        return pred_semantic_map.numpy() == 2

    def _set_sky_depth_(self,input_frame):
        sky = self._segment_sky_(input_frame.rgb)
        valid_dpt = input_frame.dpt[~sky]
        _max = np.percentile(valid_dpt,95)
        self.sky_depth = _max*2
    
    def _remove_sky_(self,frame):  
        sky = self._segment_sky_(frame.rgb)
        dpt = frame.dpt
        ipt = frame.inpaint
        # set dpt to ***
        dpt[sky] = self.sky_depth
        # set sky un-generate gaussians
        ipt[sky] = False
        frame.sky = sky
        frame.dpt = dpt
        frame.inpaint = ipt
        return frame

    def _sphere_sky_(self,frame):
        '''
        intrinsic.inv() @ [u,v,1]*d -- [a,b,c]*d
        extrinsic.inv() @ [a,b,c]*d + [sx,sy,sz] -- Norm2 -- R2      
        '''
        inv_intrinsic = np.linalg.inv(frame.intrinsic)
        inv_extrinsic = np.linalg.inv(frame.extrinsic)
        inv_R, inv_t = inv_extrinsic[0:3,0:3],inv_extrinsic[0:3,-1]
        # get u,v mesh grid
        grid_u,grid_v = np.meshgrid(np.arange(frame.W),np.arange(frame.H))
        grid = np.concatenate((grid_u[...,None],grid_v[...,None],np.ones_like(grid_v[...,None])),axis=-1)
        
        # move everything to torch
        with torch.no_grad():
            grid = torch.from_numpy(grid.astype(np.float32)).cuda()
            inv_R = torch.from_numpy(inv_R.astype(np.float32)).cuda()
            inv_t = torch.from_numpy(inv_t.astype(np.float32)).cuda()
            inv_intrinsic = torch.from_numpy(inv_intrinsic.astype(np.float32)).cuda()
            # inv_extrinsic @ inv_intrinsic @ [u,v,1]*d  -- world coordinate -- norm should be R
            grid_meter = torch.einsum('ab,hwb->hwa',inv_intrinsic,grid)
            grid_meter_inv_R = torch.einsum('ab,hwb->hwa',inv_R,grid_meter)
            # (ad,bd,cd + inv_t)**2 = R**2
            # (ad+t0)**2 + (bd+t1)**2 + (cd+t2)**2 = R*R
            # (a2+b2+c2)d2 + 2(at0,bt1,ct2)d + (t02+t12+t22-R2) = 0
            a = torch.square(grid_meter_inv_R).sum(-1)
            b = 2*(grid_meter_inv_R[...,0]*inv_t[0] + grid_meter_inv_R[...,1]*inv_t[1] + grid_meter_inv_R[...,2]*inv_t[2])
            c = torch.square(inv_t).sum() - self.sky_depth**2
            # solve d
            t = torch.sqrt(b**2-4*a*c)
            d0 = (-b + t) / 2 / a
            d1 = (-b - t) / 2 / a
            d = torch.where(d0>0,d0,d1)
        d = d.cpu().numpy()
        
        # set depth
        sky = self._segment_sky_(frame.rgb)
        dpt = frame.dpt
        ipt = frame.inpaint
        dpt[sky] = d[sky]
        # set sky un-generate gaussians
        ipt[sky] = True
        frame.sky = sky
        frame.dpt = dpt
        frame.inpaint = ipt
        return frame
            
    def _plain_sky_(self,frame):
        sky = self._segment_sky_(frame.rgb)
        dpt = frame.dpt
        ipt = frame.inpaint
        # set dpt to ***
        dpt[sky] = self.sky_depth
        # set sky un-generate gaussians
        ipt[sky] = True
        frame.sky = sky
        frame.dpt = dpt
        frame.inpaint = ipt
        return frame
    
    def _no_precess_(self,frame):
        sky = self._segment_sky_(frame.rgb)
        dpt = frame.dpt
        ipt = frame.inpaint
        # set sky un-generate gaussians
        ipt[sky] = True
        frame.sky = sky
        frame.dpt = dpt
        frame.inpaint = ipt
        return frame 

    def __call__(self, frame):
        if self.sky_keep:
            return self._sphere_sky_(frame)
        else:
            return self._remove_sky_(frame)
        
if __name__ == '__main__':
    from pipe.cfgs import load_cfg
    from ops.utils.utils import save_pic
    
    cfg = load_cfg('pipe/cfgs/basic.yaml')
    tool = Sky_Tool(cfg)
    rgb = np.array(Image.open(f'/mnt/proj/0_Datasets/6_VKITTI/train/0020-15-deg-right/images/00004.png'))
    sky = tool._segment_sky_(rgb)
    save_pic(sky*1.,'sky.png')