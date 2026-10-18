import os
import sys
from pathlib import Path

FOOOCUS_ROOT = Path(__file__).resolve().parents[3] / 'tools' / 'Fooocus'
sys.path.insert(0, str(FOOOCUS_ROOT))


class Fooocus_Tool():
    def __init__(self,fooocus_ckpts='tools/Fooocus/models') -> None:
        # Fooocus reads presets/configuration from cwd and parses process argv.
        # Isolate that initialization from INTERFACE's own command-line options.
        fooocus_ckpts = str(Path(fooocus_ckpts).resolve())
        previous_cwd, previous_argv = Path.cwd(), sys.argv
        try:
            os.chdir(FOOOCUS_ROOT)
            sys.argv = [previous_argv[0]]
            self._set_ckpt(str(Path(fooocus_ckpts).resolve()))
            self._load_model()
        finally:
            os.chdir(previous_cwd)
            sys.argv = previous_argv

    def _set_ckpt(self,ckpts):
        from fooocus_preset import set_ckpt_dir
        set_ckpt_dir(ckpts)

    def _load_model(self):
        from fooocus_command import Fooocus
        self.tool = Fooocus()

    def __call__(self,
                 image_number = 1,
                 prompt = '',
                 negative_prompt = '',
                 outpaint_selections = ['Left', 'Right', 'Top', 'Bottom'],
                 outpaint_extend_times = 0.4,
                 origin_image = None,
                 mask_image = None,
                 seed = None):
        '''
        origin_image numpy HW3 0-255 / 0-1
        mask_image numpy HW(3) inpaint area be 255,255(,255) / 1,1(,1) else be 0,0(,0)
        '''
        return self.tool(image_number,prompt,negative_prompt,outpaint_selections,outpaint_extend_times,origin_image,mask_image,seed)
