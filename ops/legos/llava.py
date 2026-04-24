import PIL
import torch
import numpy as np
import os
from transformers import AutoProcessor, LlavaForConditionalGeneration

class Llava():
    def __init__(self,device='cuda',
                 llava_ckpt='llava-hf/bakLlava-v1-hf',
                 offline=False,
                 local_files_only=None) -> None:
        self.device = device
        self.model_id = llava_ckpt

        # Allow fully-offline loading from local HuggingFace cache or local model path.
        if local_files_only is None:
            env_offline = os.environ.get('TRANSFORMERS_OFFLINE', '0') == '1' \
                       or os.environ.get('HF_HUB_OFFLINE', '0') == '1' \
                       or os.environ.get('VISTADREAM_LLAVA_OFFLINE', '0') == '1'
            local_files_only = bool(offline or env_offline)

        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            local_files_only=local_files_only,
            ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            local_files_only=local_files_only,
        )

    def __call__(self,image:PIL.Image, prompt=None):

        # input check
        if not isinstance(image,PIL.Image.Image):
            if np.amax(image) < 1.1:
                image = image * 255
            image = image.astype(np.uint8)
            image = PIL.Image.fromarray(image)
        
        prompt = '<image>\n USER: Detaily imagine and describe the scene this image taken from? Do not mention people.\n ASSISTANT: This image is taken from a scene of ' if prompt is None else prompt
        inputs = self.processor(image, prompt, return_tensors='pt').to(self.model.device,torch.float16)
        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)
        answer = self.processor.decode(output[0][2:], skip_special_tokens=True)
        return answer
    
if __name__ == '__main__':
    tool = Llava()
    from PIL import Image
    image = Image.open(f'/mnt/proj/5_VistaDream/VistaDream_v6/VistaDream_v6/data/readingroom/color.png')
    print(tool(image))