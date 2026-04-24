import cv2,os
import numpy as np
from ops.utils.utils import save_pic
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

class qwen_iqa():
    def __init__(self):
        self.n_images = 50
        # # default: Load the model on the available device(s)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        )
        # default processor
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
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
            frames.append(frame)
        # random sample...
        # idxs = np.random.permutation(len(frames))[0:self.n_images]
        idxs = np.linspace(0,len(frames)-1,self.n_images)
        idxs = idxs.astype(np.int64)
        frames = [frames[i] for i in idxs]
        return frames

    def _query_once_(self,image,question):

        self.messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": question+", just anwser with yes or no? Assistent:"},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(self.messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text
        
    def __call__(self, video_fn=f'data/vistadream/bust/video_rgb.mp4',questions=None):
        self.questions = self.questions if questions is None else questions
        results = {}
        renderings = self._load_renderings(video_fn)
        for key,question in self.questions.items():
            results[key] = []
            for rendering in renderings:
                save_pic(rendering,'temp.png')
                prompt = self._query_once_('temp.png',question)   
                if prompt[0:2] == 'Ye': results[key].append(1)
                else: results[key].append(0)
        os.system('rm temp.png')
        for key,val in results.items():
            results[key] = np.mean(np.array(val))    
        return results       
    