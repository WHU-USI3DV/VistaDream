import torch
from diffusers import LCMScheduler,PNDMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import *


class Hack_SDPipe_Stepwise(StableDiffusionPipeline):

    @torch.no_grad()
    def _use_lcm(self,use=True,ckpt='"latent-consistency/lcm-lora-sdv1-5"'):
        if use:
            self.use_lcm = True
            adapter_id = ckpt
            self.scheduler = LCMScheduler.from_config(self.scheduler.config)
            # load and fuse lcm lora
            self._guidance_scale = 0.0
            self.load_lora_weights(adapter_id)
            self.fuse_lora()
        else:
            self.use_lcm = False
            self._guidance_scale = 7.5

    @torch.no_grad()
    def re_init(self,num_inference_steps=50):
        # hyper-parameters
        eta = 0.0
        timesteps = None
        generator = None
        self._clip_skip = None
        self._interrupt = False
        self._guidance_rescale = 0.0
        self.added_cond_kwargs = None
        self._cross_attention_kwargs = None
        self._do_classifier_free_guidance = self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

        # 2. Define call parameters
        batch_size = 1
        device = self._execution_device
        
        # 4. Reset timesteps and current timestep of scheduler
        self.timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        self.extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.2 Optionally get Guidance Scale Embedding
        self.timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * 1)
            self.timestep_cond = self.get_guidance_scale_embedding(guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim).to(device=device)

    @torch.no_grad()
    def _encode_text_prompt(self,
                            prompt,
                            negative_prompt='fake,ugly,unreal'):
        # 3. Encode input prompt
        lora_scale = (self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None)
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            self._execution_device,
            1,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        return prompt_embeds

    @torch.no_grad()
    def _step_noise(self,
                    latents,
                    time_step,
                    prompt_embeds):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, time_step)
        # predict the noise residual
        noise_pred = self.unet(
            latent_model_input,
            time_step,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=self.timestep_cond,
            cross_attention_kwargs=self.cross_attention_kwargs,
            added_cond_kwargs=self.added_cond_kwargs,
            return_dict=False,
        )[0]
        # perform guidance
        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)
        return noise_pred

    # @torch.no_grad()
    def _encode(self, input):
        '''
        # single condition encoding 
        input: B3HW
        return: B4H'W'
        if low-vram: vae on cpu, input should also on cpu
        '''
        h = self.vae.encoder(input)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        latent = mean * self.vae.config.scaling_factor
        return latent
    
    def _decode(self, latent):
        '''
        single target decoding
        input: B4H'W'
        return: B3HW
        '''
        # scale latent
        latent = latent / self.vae.config.scaling_factor
        # decode
        z = self.vae.post_quant_conv(latent)
        output = self.vae.decoder(z)
        return output.clip(-1,1)

    def _for_sd_denoise(
            self,
            samples: torch.Tensor,
            timesteps: torch.IntTensor,
            ) -> torch.Tensor:
            # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
            # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
            # for the subsequent add_noise calls
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device=samples.device)
            alphas_cumprod = self.scheduler.alphas_cumprod.to(dtype=samples.dtype)
            timesteps = timesteps.to(samples.device)

            sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
            sqrt_alpha_prod = sqrt_alpha_prod.flatten()
            while len(sqrt_alpha_prod.shape) < len(samples.shape):
                sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

            sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
            while len(sqrt_one_minus_alpha_prod.shape) < len(samples.shape):
                sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
            return sqrt_alpha_prod, sqrt_one_minus_alpha_prod

    def _solve_x0(self, latents, noise_pred, t):
        if self.use_lcm:
            x0_latents = self.scheduler.step(noise_pred, t.squeeze(), latents)
            x0_latents = x0_latents.denoised 
            # note here must be a fake denoise
            self.scheduler._step_index-=1
        else:
            sqrt_alpha_prod, sqrt_one_minus_alpha_prod = self._for_sd_denoise(latents,t)
            x0_latents = (latents - sqrt_one_minus_alpha_prod * noise_pred) / sqrt_alpha_prod
        # results
        x0 = self._decode(x0_latents)
        return x0_latents, x0

    def _step_denoise(self, latents, noise_pred, t):
        latents = self.scheduler.step(noise_pred, t.squeeze(), latents).prev_sample   
        return latents
    
    def xt_x0_noise(
        self,
        xt_latents: torch.Tensor,
        x0_latents: torch.Tensor,
        timesteps: torch.IntTensor,
        ) -> torch.Tensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        alphas_cumprod = self.scheduler.alphas_cumprod.to(dtype=xt_latents.dtype,device=xt_latents.device)
        timesteps = timesteps.to(xt_latents.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(xt_latents.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(xt_latents.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noise = (xt_latents - sqrt_alpha_prod * x0_latents) / sqrt_one_minus_alpha_prod
        return noise
    
    def _solve_noise_given_x0_latent(self, latents, x0_latents, t):
        noise = self.xt_x0_noise(latents,x0_latents,t)
        # -------------------- noise for supervision -----------------
        if self.scheduler.config.prediction_type == "epsilon":
            noise = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            noise = self.scheduler.get_velocity(x0_latents, noise, t)
        # ------------------------------------------------------------
        return noise
    


