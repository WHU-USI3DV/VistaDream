import os
# First Stage
from ops.legos.llava import Llava
from ops.legos.rgbipt import RGB_Inpaint_Tool
from ops.legos.recons import Depth_Inpaint_Tool
# Second Stage
from ops.legos.mcs import HackSD_MCS

class Load_Tools_Phase():
    def __init__(self,cfg,mcs=False):
        self.cfg = cfg
        self.llava = None
        self.llava_load_failed = False
        self.enable_llava = False
        preloads = self.cfg.tools.preload
        if 'llava' in preloads:
            # Lazy loading: only load when prompt generation really needs it.
            self.enable_llava = True
        if 'rgb_inpaint' in preloads:
            self._load_fooocus_()
        if 'dpt_inpaint' in preloads:
            self._load_dpt_inpaintor_()
        # load refiner
        if ('mcs' in preloads) and mcs:
            self._load_mcs_()

    def _load_llava_(self):
        offline = False
        try:
            offline = bool(self.cfg.model.vlm.llava.offline)
        except Exception:
            offline = False
        self.llava = Llava(
            device='cpu',
            llava_ckpt=self.cfg.model.vlm.llava.ckpt,
            offline=offline,
        )

    def _ensure_llava_(self):
        if not self.enable_llava:
            return None
        if self.llava is not None:
            return self.llava
        if self.llava_load_failed:
            return None
        try:
            self._load_llava_()
            return self.llava
        except Exception as e:
            # Fail once and do not retry repeatedly in this process.
            print(f'[WARN] Skip loading LLaVA: {e}')
            self.llava = None
            self.llava_load_failed = True
            return None

    def _load_fooocus_(self):
        # in case of corrupted fooocus ckpt
        fooocus_path = f'{self.cfg.model.paint.fooocus.ckpts}/checkpoints/juggernautXL_v8Rundiffusion.safetensors'
        if not os.path.exists(fooocus_path):
            if os.path.exists(fooocus_path + '.corrupted'):
                os.system(f'mv {fooocus_path}.corrupted {fooocus_path}')
        self.rgb_inpaint = RGB_Inpaint_Tool(self.cfg)

    def _load_dpt_inpaintor_(self):
        self.dpt_inpaint = Depth_Inpaint_Tool(self.cfg,device='cpu')

    def _load_mcs_(self):
        self.mcs_refiner = HackSD_MCS(
            device='cpu',
            use_lcm=True,
            denoise_steps = self.cfg.scene.mcs.steps,
            total_steps = self.cfg.scene.mcs.total_steps,
            sd_ckpt = self.cfg.model.optimize.sd,
            lcm_ckpt = self.cfg.model.optimize.lcm)