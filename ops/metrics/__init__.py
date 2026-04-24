import numpy as np

class Metrics():
    def __init__(self):
        pass
    
    def _wo_refer_(self,
                   video_fns,
                   lvms=['llava','qwen'],
                   tsed=False,
                   K=None,
                   sparse=False):
        from .lvm import LVM_IQA
        tool = LVM_IQA(sparse=sparse)
        tool(video_fns,lvms)
        # tsed?
        if tsed and (K is not None):
            from .depth.tsed import TSED_Tool
            tool = TSED_Tool()
            tsed_metric = []
            for video_fn in video_fns:
                item = tool(video_fn,nframes=30,K=K)
                tsed_metric.append(item)
            tsed_metric = np.array(tsed_metric)
            print('TSED value:',np.mean(tsed_metric))
        
        
    def _w_refer_(self,scene_dirs):
        '''
        After generate the scene, prepare the views for evaluation
        set an eval dir in the scene dir
        it contains:
            i.refer.rgb.png
            i.refer.dpt.npy
            i.render.rgb.png
            i.render.dpt.npy
        '''
        # rgb
        from .percept import Perception_IQA
        tool = Perception_IQA()
        tool(scene_dirs)
        # dpt
        from .depth import Depth_IQA
        tool = Depth_IQA()
        tool(scene_dirs)


