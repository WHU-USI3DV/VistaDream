import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from ops.utils.utils import save_pic
from ops.legos.llava import Llava

class llava_iqa():
    def __init__(self) -> None:
        self.n_images = 50
        self._questions()
        self.llava = Llava(device='cuda')
    
    def _questions(self):
        # quailty, noise, structure, texture
        self.questions = {'noise-free':'Is the image free of noise or distortion',
        'sharp':'Does the image show clear objects and sharp edges',
        'structure':'Is the overall scene coherent and realistic in terms of layout and proportions in this image',
        'detail':'Does this image show detailed textures and materials',
        'quality':'Is this image overall a high quality image with clear objects, sharp edges, nice color, good overall structure, and good visual quailty'}

    def _load_renderings(self,video_fn):
        capturer = cv2.VideoCapture(video_fn)
        frames = []
        while True:
            ret,frame = capturer.read()
            if ret == False or frame is None: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame.astype(np.uint8))
            frames.append(frame)
        # random sample...
        idxs = np.random.permutation(len(frames))[0:self.n_images]
        idxs = np.linspace(0,len(frames)-1,self.n_images)
        idxs = idxs.astype(np.int64)
        frames = [frames[i] for i in idxs]
        return frames

    def _query_once_(self, image, question):
        image = np.array(image) if not type(image) is np.ndarray else image
        query = f'<image>\n USER: {question}, just anwser with yes or no? \n ASSISTANT: '
        prompt = self.llava(image,query)
        split  = str.rfind(prompt,'ASSISTANT: ') + len(f'ASSISTANT: ')
        prompt = prompt[split+1:]
        return prompt

    def __call__(self,video_fn=f'data/vistadream/bust/video_rgb.mp4',questions=None):
        self.questions = self.questions if questions is None else questions
        results = {}
        renderings = self._load_renderings(video_fn)
        for key,question in self.questions.items():
            results[key] = []
            for rendering in renderings:
                prompt = self._query_once_(rendering,question)
                if prompt[0:2] == 'Ye': results[key].append(1)
                else: results[key].append(0)
        for key,val in results.items():
            results[key] = np.mean(np.array(val))
        return results
    