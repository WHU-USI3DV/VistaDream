'''
with given refering RGB-D and Trajectory
'''
import torch
from PIL import Image
from ops.utils.utils import *
from ops.gs.basic import Gaussian_Scene
from ops.gs.train import GS_Train_Tool
from ops.utils.visual_check import Check
from pipe.stages.xmcsrefine import MCS_Phase
from pipe.stages.prepare import Prepare_Phase
from pipe.stages.scaffold import Scaffold_Phase
from pipe.stages.loadtools import Load_Tools_Phase
from pipe.stages.warpextend import WarpExtend_Phase
from ops.trajs import _generate_trajectory

        
class Pipeline_Refer():
    def __init__(self,cfg) -> None:
        self.cfg = cfg
        self.checkor = Check()
        self.traj_type = 'interp'
        self.prepare = Prepare_Phase(self.cfg)()

    def _align_to_refer_(self,scene,refer_dpt):
        
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
        
        anchor = scene.frames[0]
        # align the estimated depth to the refering depth
        sky = anchor.sky
        dpt = deepcopy(anchor.dpt)
        select = (refer_dpt > 1e-3) & (~sky)
        dpt_select = dpt[select]
        refer_dpt_select = refer_dpt[select]
        align_scale, align_shift = _align_scale_shift_numpy(dpt_select,refer_dpt_select)
        
        align_scene:Gaussian_Scene = deepcopy(scene)
        align_scene.frames = []
        align_scene.gaussian_frames = []
        # re add
        for frame in scene.frames:
            frame.dpt = frame.dpt*align_scale+align_shift
            align_scene._add_trainable_frame(frame)
            align_scene._require_grad_(True)
            GS_Train_Tool(align_scene,iters=self.cfg.scene.gaussian.opt_iters_per_frame)(align_scene.frames)

    def _warp_on_pre_define_trajectory_(self,scene,trajs):
        scene.traj_type = self.traj_type
        warp_tool = WarpExtend_Phase(self.cfg,self.tools)
        for traj in trajs:
            frame = warp_tool._pose_to_frame(scene,traj)
            scene = warp_tool._inpaint_next_frame(scene,frame)
        scene.dense_trajs = _generate_trajectory(self.cfg,scene,len(scene.frames)*5)
        return scene

    def __call__(self,rgb_fn,refer_dpt,trajs):
        temp_dir = rgb_fn[:str.rfind(rgb_fn,'/')]
        # prepare stage
        if os.path.exists(f'{temp_dir}/scene.coarse.pth'):
            self.cfg.tools.preload = ['mcs']
            self.tools = Load_Tools_Phase(self.cfg)
            scene = torch.load(f'{temp_dir}/scene.coarse.pth')
        else:
            self.cfg.tools.preload = ['llava','rgb_inpaint','dpt_inpaint','mcs']
            self.tools = Load_Tools_Phase(self.cfg)
            # scaffold stage
            rgb = Image.open(rgb_fn)
            scene = Scaffold_Phase(self.cfg,self.tools)(rgb)
            scene = self._align_to_refer_(scene,refer_dpt)
            # coarse stage
            scene = self._warp_on_pre_define_trajectory_+(scene,trajs)
            torch.save(scene,f'{temp_dir}/scene.coarse.pth')
            self.checkor._render_video(scene,save_dir=f'{temp_dir}/coarse.')
        # refine stage
        scene = MCS_Phase(self.cfg,self.tools,scene,device='cuda')()
        torch.save(scene,f'{temp_dir}/scene.refine.pth')
        self.checkor._render_video(scene,save_dir=f'{temp_dir}/refine.')


