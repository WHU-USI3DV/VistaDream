'''
render using frames in GS
inpaint with fooocus
'''
import os
import cv2
import numpy as np
from PIL import Image
        
class Prepare_Phase():
    def __init__(self,cfg) -> None:
        self.cfg = cfg

    def _mkdir(self,dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def _resize_input(self,fn):
        resize_long_edge = int(self.cfg.scene.input.resize_long_edge)
        print(f'[Preprocess...] Resize the long edge of input image to {resize_long_edge}.')
        spl = str.rfind(fn,'.')
        backup_fn = fn[:spl] + '.original' + fn[spl:]
        rgb = Image.open(fn)
        rgb.save(backup_fn) # back up original image 
        rgb = np.array(rgb)[:,:,:3]/255.
        H,W = rgb.shape[0:2]
        if H>W:
            W = int(W*resize_long_edge/H)
            H = resize_long_edge
        else:
            H = int(H*resize_long_edge/W)
            W = resize_long_edge
        rgb = cv2.resize(rgb,(W,H))
        pic = (rgb * 255.0).clip(0, 255)
        pic_save = Image.fromarray(pic.astype(np.uint8))
        pic_save.save(fn)

    def __call__(self):
        rgb_fn = self.cfg.scene.input.rgb
        # resize input 
        self._resize_input(rgb_fn)

    
    
    