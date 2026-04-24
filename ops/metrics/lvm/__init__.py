import torch
import numpy as np
from tqdm import tqdm

class LVM_IQA():
    def __init__(self,sparse=False):
        self._questions(sparse=sparse)
        
    def _questions(self,sparse=False):
        questions = {'noise-free':'Is the image free of noise or distortion',
        'sharp':'Does the image show clear objects and sharp edges',
        'structure':'Is the overall scene coherent and realistic in terms of layout and proportions in this image',
        'clarity':'Is this image clear, not quite blurry or over-smoothed',
        'quality':'Is this image overall a high quality image with clear objects, sharp edges, nice color, good overall structure, and good visual quailty'}
        
        # black = ', regardless the large black regions within the image.'
        # sparse_questions = {'noise-free':'Is the image contain not much noise'+black,
        # 'sharp':'Does the image show sharp objects'+black,
        # 'structure':'Is the overall scene coherent in this image'+black,
        # 'clarity':'Does this image contain good details'+black,
        # 'quality':'Does this image have a acceptable quailty'+black}
        
        self.questions = questions 
    
    def __process_videos__(self,video_fns,tool):
        results = {}
        for key,_ in self.questions.items():
            results[key] = []
        for video_fn in tqdm(video_fns):
            result = tool(video_fn,self.questions)
            vals = []
            for key,val in result.items():
                results[key].append(val)
                vals.append(val)
            print(video_fn,np.mean(np.array(vals)))
        for key,val in results.items():
            results[key] = np.mean(np.array(val))
        results['MEAN'] = [val for key,val in results.items()]
        results['MEAN'] = np.mean(np.array(results['MEAN']))
        return results
    
    @torch.no_grad()
    def __llava__(self,video_fns):
        from .llava import llava_iqa
        tool = llava_iqa()
        return self.__process_videos__(video_fns,tool)
        
    @torch.no_grad()
    def __qwenvl__(self,video_fns):
        from .qwen import qwen_iqa
        tool = qwen_iqa()
        return self.__process_videos__(video_fns,tool)

    @torch.no_grad()
    def __intervl__(self,video_fns):
        from .intervl import intervl_iqa
        tool = intervl_iqa()
        return self.__process_videos__(video_fns,tool)
    
    def __call__(self, video_fns, lvms = ['llava','qwen','intervl']):
        if 'llava' in lvms:
            print('Using LLaVA...')
            print(self.__llava__(video_fns))
            torch.cuda.empty_cache()
        if 'qwen' in lvms:
            print('Using Qwen-VL...')
            print(self.__qwenvl__(video_fns))
            torch.cuda.empty_cache()
        if 'intervl' in lvms:
            print('Using Inter-VL...')
            print(self.__intervl__(video_fns))   
            torch.cuda.empty_cache()  