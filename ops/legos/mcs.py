import torch
import numpy as np
from tools.StableDiffusion.Hack_SD_stepwise import Hack_SDPipe_Stepwise

'''
Input: Multiview images with added noise
denoise to x0
denoise from step t1 to step t2
'''    

class HackSD_MCS():
    '''
        transform images to self.latents
        add noise to self.latents
        predict step noise --> x0
        mv RGB-D warp as target image
        target image encode to latent and get target noise
        noise rectification
        step denoise
    '''
    def __init__(self,
                 device='cpu',
                 use_lcm=True,
                 denoise_steps=10,
                 anchor_steps=0,
                 sd_ckpt=f'tools/StableDiffusion/ckpt',
                 lcm_ckpt=f'latent-consistency/lcm-lora-sdv1-5',
                 total_steps=50) -> None:
        '''
        ref_rgb should be -1~1 tensor B*3*H*W
        '''
        self.total_steps = total_steps
        self.device = device
        self.target_type = np.float32
        self.use_lcm = use_lcm
        self.sd_ckpt = sd_ckpt
        self.lcm_ckpt = lcm_ckpt
        self._load_model()
        # define step to add noise and steps to denoise
        self.denoise_steps = denoise_steps
        self.anchor_steps = anchor_steps
        # timesteps in 1000
        self.timesteps = self.model.timesteps

    def _load_model(self):
        self.model = Hack_SDPipe_Stepwise.from_pretrained(self.sd_ckpt)
        self.model._use_lcm(self.use_lcm,self.lcm_ckpt)
        self.model.re_init(num_inference_steps=self.total_steps)
        try:
            self.model.enable_xformers_memory_efficient_attention()
        except:
            pass  # run without xformers
        self.model = self.model.to(self.device)

    def _re_init_(self):
        self.model.re_init(self.total_steps) # this is quite necessary to guarantee the schedular is reset correctly/

    def to(self, device):
        self.device = device
        self.model.to(device)

    @ torch.no_grad()
    def _add_noise_to_latent(self,latents):
        self._re_init_() # this is quite necessary to guarantee the schedular is reset correctly/
        if self.use_lcm:
            self.model.scheduler._init_step_index(self.timesteps[self.total_steps-self.denoise_steps])
            print('Init step-index:',self.model.scheduler.step_index)
        bsz = latents.shape[0]
        # in the Stable Diffusion, the iterations numbers is 1000 for adding the noise and denosing.
        timestep = self.timesteps[-self.denoise_steps]
        timestep = timestep.repeat(bsz).to(self.device)
        # target noise
        noise = torch.randn_like(latents)
        # add noise
        noisy_latent = self.model.scheduler.add_noise(latents, noise, timestep)
        # -------------------- noise for supervision -----------------
        if self.model.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.model.scheduler.config.prediction_type == "v_prediction":
            target = self.model.scheduler.get_velocity(latents, noise, timestep)
        return noisy_latent, timestep, target

    @ torch.no_grad()
    def _encode_mv_init_images(self, images):
        '''
        images should be B3HW
        '''
        images = images * 2 - 1
        self.latents = self.model._encode(images)
        self.latents,_,_ = self._add_noise_to_latent(self.latents)

    @ torch.no_grad()
    def _denoise_to_x0(self, i_in_denoise_steps, prompt_latent:torch.Tensor):
        # temp noise prediction
        timestep_in_1000 = self.timesteps[-self.denoise_steps+i_in_denoise_steps]
        timestep_in_1000 = torch.tensor([timestep_in_1000]).to(self.device)
        noise_pred = self.model._step_noise(self.latents, timestep_in_1000, prompt_latent.repeat(len(self.latents),1,1))
        # solve image
        _,x0 = self.model._solve_x0(self.latents,noise_pred,timestep_in_1000)   
        x0 = (x0 + 1) / 2 # in 0-1
        return noise_pred, x0, timestep_in_1000

    @ torch.no_grad()
    def _step_denoise(self, t, pred_noise, rect_x0, rect_w = 0.7):
        '''
        pred_noise B4H//8W//8
        x0, rect_x0 B3HW
        '''
        # encoder rect_x0 to latent
        rect_x0 = rect_x0 * 2 - 1
        rect_latent = self.model._encode(rect_x0)
        # rectified noise
        rect_noise = self.model._solve_noise_given_x0_latent(self.latents,rect_latent,t)
        # noise rectification
        rect_noise = rect_noise / rect_noise.std(dim=list(range(1, rect_noise.ndim)),keepdim=True) \
                                * pred_noise.std(dim=list(range(1, pred_noise.ndim)),keepdim=True)
        pred_noise = pred_noise*(1.-rect_w) + rect_noise*rect_w
        # step forward
        self.latents = self.model._step_denoise(self.latents,pred_noise,t)

    @torch.no_grad()
    def _encode_text_prompt(self,
                            prompt,
                            negative_prompt='fake,ugly,unreal'):
        prompt_embeds = self.model._encode_text_prompt(prompt,negative_prompt)
        return prompt_embeds

    @ torch.no_grad()
    def _decode_mv_imgs(self):
        imgs = self.model._decode(self.latents)
        imgs = (imgs + 1) / 2
        return imgs

    @ torch.no_grad()
    def _single_sds_(self,rgb,prompt_latent):
        # add noise
        add_steps = 12
        # encode rgb to latent
        rgb = torch.from_numpy(rgb.astype(np.float32)).permute(2,0,1)[None].to(self.device)
        latent = self.model._encode(rgb*2-1)
        # add noise
        t = self.timesteps[-add_steps]
        t = t.repeat(latent.shape[0]).to(self.device)
        noise = torch.randn_like(latent)
        latent = self.model.scheduler.add_noise(latent, noise, t)
        # denoise
        noise_pred = self.model._step_noise(latent, t, prompt_latent.repeat(len(latent),1,1))
        # solve image
        _,x0 = self.model._solve_x0(latent,noise_pred,t)   
        x0 = (x0 + 1) / 2 # in 0-1
        x0 = x0[0].permute(1,2,0).detach().cpu().numpy()
        return x0
        
if __name__ == '__main__':
    from PIL import Image
    tool = HackSD_MCS(device='cuda',use_lcm=True,
                      denoise_steps=9,
                      anchor_steps=5,
                      sd_ckpt='/mnt/proj/0_Checkpoints/6_Stable_Diffusion',
                      lcm_ckpt='/mnt/proj/0_Checkpoints/7_LCM/pytorch_lora_weights.safetensors')
    image = np.array(Image.open(f'/mnt/proj/5_VistaDream/VistaDream+_v1/rect.0.png'))
    image = torch.from_numpy((image/255.).astype(np.float32)).permute(2,0,1)[None].to('cuda')
    image = image * 2 - 1
    
    prompt = torch.load('/mnt/proj/5_VistaDream/VistaDream+_v1/data/main_AblateRecon_MDE/living_room/scene.coarse.pth',weights_only=False)
    prompt = prompt.frames[0].prompt
    rgb_prompt_latent = tool.model._encode_text_prompt(prompt)
    
    
    t = tool.timesteps[-1].repeat(1).to(tool.device)
    
    latent = tool.model._encode(image)
    noise = torch.randn_like(latent)
    latent,_,_ = tool.model.scheduler.add_noise(latent, noise, t)
    
    noise = tool.model._step_noise(latent, t, rgb_prompt_latent)
    # solve image
    _,x0 = tool.model._solve_x0(latent,noise,t)   
    x0 = (x0 + 1) / 2 # in 0-1
    
    x0 = x0[0].permute(1,2,0).detach().cpu().numpy()
    x0 = (x0*255).astype(np.uint8)
    x0 = Image.fromarray(x0)
    x0.save(f'recon.png')
    


