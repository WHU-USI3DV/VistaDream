import PIL
import torch
import numpy as np
import os
from transformers import AutoProcessor, LlavaForConditionalGeneration

class Llava():
    def __init__(self,device='cuda',
                 llava_ckpt='llava-hf/bakLlava-v1-hf',
                 offline=False,
                 local_files_only=None,
                 attn_implementation=None,
                 max_new_tokens=200) -> None:
        self.device = device
        self.model_id = llava_ckpt
        self.max_new_tokens = max_new_tokens

        # Allow fully-offline loading from local HuggingFace cache or local model path.
        if local_files_only is None:
            env_offline = os.environ.get('TRANSFORMERS_OFFLINE', '0') == '1' \
                       or os.environ.get('HF_HUB_OFFLINE', '0') == '1' \
                       or os.environ.get('VISTADREAM_LLAVA_OFFLINE', '0') == '1'
            local_files_only = bool(offline or env_offline)

        model_kwargs = {}
        if attn_implementation is not None:
            model_kwargs['attn_implementation'] = attn_implementation

        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
            **model_kwargs,
            ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            local_files_only=local_files_only,
            use_fast=False,
        )

    def __call__(self, image:PIL.Image, prompt=None, allowed_answers=None):

        # input check
        if not isinstance(image,PIL.Image.Image):
            if np.amax(image) < 1.1:
                image = image * 255
            image = image.astype(np.uint8)
            image = PIL.Image.fromarray(image)

        prompt = '<image>\n USER: Detaily imagine and describe the scene this image taken from? Do not mention people.\n ASSISTANT: This image is taken from a scene of ' if prompt is None else prompt
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors='pt',
        ).to(self.model.device, torch.float16)
        input_length = inputs.input_ids.shape[1]
        generation_kwargs = {
            'max_new_tokens': self.max_new_tokens,
            'do_sample': False,
        }
        if allowed_answers is not None:
            allowed_token_ids = []
            for answer in allowed_answers:
                token_ids = self.processor.tokenizer.encode(
                    answer,
                    add_special_tokens=False,
                )
                if len(token_ids) != 1:
                    raise ValueError(
                        f'Constrained answer must be one token: {answer!r} -> {token_ids}'
                    )
                allowed_token_ids.append(token_ids[0])
            generation_kwargs['max_new_tokens'] = 1
            generation_kwargs['prefix_allowed_tokens_fn'] = (
                lambda _batch_id, _input_ids: allowed_token_ids
            )

        output = self.model.generate(**inputs, **generation_kwargs)
        generated = output[0, input_length:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()
