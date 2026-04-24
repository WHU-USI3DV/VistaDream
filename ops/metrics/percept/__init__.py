import cv2
import torch,PIL
import numpy as np
from glob import glob
from tqdm import tqdm
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class Perception_IQA():
    def __init__(self):
        _ = torch.manual_seed(123)
    
    def _load_video_(self,video):
        frames = []
        cap = cv2.VideoCapture(video)
        while True:
            ret, frame = cap.read()  # ret 表示是否成功读取，frame 是一帧图像
            if not ret: break
            frames.append(frame[...,[2,1,0]])
        return frames
        
    def _preprocess(self,rgb):
        if isinstance(rgb, PIL.PngImagePlugin.PngImageFile):
            rgb = np.array(rgb)
        if isinstance(rgb, PIL.JpegImagePlugin.JpegImageFile):
            rgb = np.array(rgb)
        if isinstance(rgb,np.ndarray):
            rgb = torch.from_numpy(rgb)
        if torch.amax(rgb) > 1.1:
            rgb = rgb / 255
        if len(rgb.shape) < 4:
            rgb = rgb.permute(2,0,1)[None]
        rgb = rgb[:,0:3]
        return rgb

    def psnr(self, refer, render):
        refer  = self._preprocess(refer)
        render = self._preprocess(render)
        render = F.interpolate(render, size=(refer.shape[-2], refer.shape[-1]), mode='bilinear', align_corners=False)
        metric = PeakSignalNoiseRatio(data_range=1.0)
        return metric(refer, render)

    def ssim(self, refer, render):
        refer  = self._preprocess(refer)
        render = self._preprocess(render)
        render = F.interpolate(render, size=(refer.shape[-2], refer.shape[-1]), mode='bilinear', align_corners=False)
        metric = StructuralSimilarityIndexMeasure(data_range=1.0)
        return metric(refer, render)

    def lpips(self, refer, render):
        refer  = self._preprocess(refer)
        render = self._preprocess(render)
        render = F.interpolate(render, size=(refer.shape[-2], refer.shape[-1]), mode='bilinear', align_corners=False)
        metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        return metric(refer, render)
        
    def __eval__(self, refers, renders):
        psnrs,ssims,lpipss = [],[],[]
        for i in range(len(refers)):
            psnrs.append(self.psnr(refers[i],renders[i]))
            ssims.append(self.ssim(refers[i],renders[i]))
            lpipss.append(self.lpips(refers[i],renders[i]))
        psnr = np.mean(np.array(psnrs))
        ssim = np.mean(np.array(ssims))
        lpips = np.mean(np.array(lpipss))
        return {
            'psnr':psnr,
            'ssim':ssim,
            'lpips':lpips
        }

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
        psnr,ssim,lpips = [],[],[]
        for scene in tqdm(scenes):
            refers,renders = [],[]
            eval_dir = f'{scene}/eval'
            N = len(glob(f'{eval_dir}/*.refer.rgb.png'))
            for i in range(N):
                refer = np.array(PIL.Image.open(f'{eval_dir}/{i}.refer.rgb.png'))
                render = np.array(PIL.Image.open(f'{eval_dir}/{i}.render.rgb.png'))
                # render = np.array(PIL.Image.open(f'{scene}/compare/seva/mast3r/{i}.rgb.png'))
                refers.append(refer)
                renders.append(render)
            result = self.__eval__(refers,renders)
            psnr.append(result['psnr'])
            ssim.append(result['ssim'])
            lpips.append(result['lpips'])
        print('RGB Evaluation with GT...')
        print('PSNR:',np.mean(np.array(psnr)))
        print('SSIM:',np.mean(np.array(ssim)))
        print('LPIPS:',np.mean(np.array(lpips)))