import copy
import random
import numpy as np
import args_manager
import modules.flags
import modules.config
import modules.flags as flags
import modules.constants as constants
import modules.hack_async_worker_class as worker

class Fooocus():
    def __init__(self) -> None:
        self._preset_parameters()
        self._inpaint_parameters()
        self._define_models()
        self._load_generator()
        
    def _seed_set(self, seed = 0, seed_random = True):
        if seed_random:
            self.image_seed = random.randint(constants.MIN_SEED, constants.MAX_SEED)
        else:
            seed_value = int(seed)
            if constants.MIN_SEED <= seed_value <= constants.MAX_SEED:
                self.image_seed = seed_value
            else:
                self.image_seed = random.randint(constants.MIN_SEED, constants.MAX_SEED)
        
    def _preset_parameters(self):
        '''
        We keep the default parameters of Fooocus unchanged!
        '''
        self.output_format = modules.config.default_output_format # png
        self.style_selections = copy.deepcopy(modules.config.default_styles) # Fooocus style unchanged:Fooocus V2+Enhanec+Sharp
        self.performance_selection = modules.config.default_performance
        self.aspect_ratios_selection = modules.config.default_aspect_ratio

        self.current_tab = 'inpaint'
        self.input_image_checkbox = True
        self.advanced_checkbox = modules.config.default_advanced_checkbox
        self.uov_method = flags.disabled
        self.uov_input_image = None

        # unchanged diffusion setting
        self.dev_mode = False
        self.guidance_scale = modules.config.default_cfg_scale
        self.sharpness = modules.config.default_sample_sharpness

        self.adm_scaler_end = 0.3
        self.adm_scaler_positive = 1.5
        self.adm_scaler_negative = 0.8
        self.refiner_swap_method = flags.refiner_swap_method
        self.adaptive_cfg = modules.config.default_cfg_tsnr
        self.clip_skip = modules.config.default_clip_skip
        self.sampler_name = modules.config.default_sampler
        self.scheduler_name = modules.config.default_scheduler
        self.vae_name = modules.config.default_vae

        self.generate_image_grid = False
        self.overwrite_step = modules.config.default_overwrite_step
        self.overwrite_switch = modules.config.default_overwrite_switch
        self.overwrite_width = -1
        self.overwrite_height = -1
        self.overwrite_vary_strength = -1
        self.overwrite_upscale_strength = -1
        self.disable_preview = modules.config.default_black_out_nsfw
        self.disable_intermediate_results = flags.Performance.has_restricted_features(modules.config.default_performance)
        self.disable_seed_increment = False
        self.read_wildcards_in_order = False

        self.black_out_nsfw = modules.config.default_black_out_nsfw
        self.save_metadata_to_images = modules.config.default_save_metadata_to_images
        self.metadata_scheme = modules.config.default_metadata_scheme

        self.debugging_cn_preprocessor = False
        self.skipping_cn_preprocessor = False
        self.mixing_image_prompt_and_vary_upscale = False
        self.mixing_image_prompt_and_inpaint = False
        self.controlnet_softness = 0.25
        self.canny_low_threshold = 64
        self.canny_high_threshold = 128

        freeu_enabled = False
        freeu_b1 = 1.01
        freeu_b2 = 1.02
        freeu_s1 = 0.99
        freeu_s2 = 0.95
        self.freeu_ctrls = [freeu_enabled, freeu_b1, freeu_b2, freeu_s1, freeu_s2]

        self.image_number = 1 
        self.default_prompt = modules.config.default_prompt
        self.negative_prompt = modules.config.default_prompt_negative

        self.inpaint_additional_prompt = ''
        self.outpaint_selections = []
        self.outpaint_extend_times = 0.0
        self.inpaint_ctrls = [False,False,'None',0,0.0,False,False,0]
        
        self.ip_ctrls = []
        for _ in range(flags.controlnet_image_count):
            ip_image = None
            # WHAT'S THIS
            self.ip_ctrls.append(ip_image)
            default_end, default_weight = flags.default_parameters[flags.default_ip]
            ip_stop = default_end
            self.ip_ctrls.append(ip_stop)
            ip_weight = default_weight
            self.ip_ctrls.append(ip_weight)
            ip_type = flags.default_ip
            self.ip_ctrls.append(ip_type)

    def _upsample_parameteres(self):
        # argue mode
        self.current_tab = 'uov'
        self.uov_method = 'Upscale (2x)'
        
        self.inpaint_additional_prompt = ''
        self.outpaint_selections = []
        self.outpaint_extend_times = 0.0
        self.inpaint_ctrls = [False,False,'None',0,0.0,False,False,0]

    def _inpaint_parameters(self, 
                            inpaint_additional_prompt = '',
                            outpaint_selections = ['Left', 'Right', 'Top', 'Bottom'],
                            outpaint_extend_times = 0.4,
                            inpaint_mode = modules.flags.inpaint_option_default
        ):
        # mode change
        assert inpaint_mode in modules.flags.inpaint_options

        # argue mode
        self.current_tab = 'inpaint'
        self.uov_method = flags.disabled

        debugging_inpaint_preprocessor = False

        if inpaint_mode == modules.flags.inpaint_option_detail:
            self.inpaint_additional_prompt = inpaint_additional_prompt + 'Sharp objects, Clear objects, optimize all blur objects'
            self.outpaint_selections = []
            self.outpaint_extend_times = 0.0
            inpaint_disable_initial_latent = False
            inpaint_engine = 'None'
            inpaint_strength = 0.3
            inpaint_respective_field = 0.0

        elif inpaint_mode == modules.flags.inpaint_option_modify:
            self.inpaint_additional_prompt = inpaint_additional_prompt
            self.outpaint_selections = []
            self.outpaint_extend_times = 0.0
            inpaint_disable_initial_latent = True
            inpaint_engine = modules.config.default_inpaint_engine_version
            inpaint_strength = 1.0
            inpaint_respective_field = 0.0

        else:
            self.inpaint_additional_prompt = ''
            self.outpaint_selections = outpaint_selections
            self.outpaint_extend_times = outpaint_extend_times
            inpaint_disable_initial_latent = False
            inpaint_engine = modules.config.default_inpaint_engine_version
            inpaint_strength = 1.0
            inpaint_respective_field = 0.618

        inpaint_erode_or_dilate = 0
        invert_mask_checkbox = False
        inpaint_mask_upload_checkbox = False

        self.inpaint_ctrls = [debugging_inpaint_preprocessor, inpaint_disable_initial_latent, inpaint_engine,
                              inpaint_strength, inpaint_respective_field,
                              inpaint_mask_upload_checkbox, invert_mask_checkbox, inpaint_erode_or_dilate]

    def _define_models(self):
        # define models
        self.base_model = modules.config.default_base_model_name
        self.refiner_model = modules.config.default_refiner_model_name
        self.refiner_switch = modules.config.default_refiner_switch
        self.lora_ctrls = []
        for i, (enabled, filename, weight) in enumerate(modules.config.default_loras):
            lora_enabled = enabled
            lora_model = filename
            lora_weight = weight
            self.lora_ctrls += [lora_enabled, lora_model, lora_weight]

    def _generate_images(self, task: worker.AsyncTask):
        import ldm_patched.modules.model_management as model_management

        with model_management.interrupt_processing_mutex:
            model_management.interrupt_processing = False
        # outputs=[progress_html, progress_window, progress_gallery, gallery]
        if len(task.args) == 0:
            return 
        imgs = self.worker(task)
        return imgs

    def _set_ctrls(self, 
                   prompt, 
                   negative_prompt,
                   # for inpainting/outpainting
                   inpaint_input_image=None,
                   inpaint_mask_image=None,
                   # for upsampling
                   upsample_input_image=None):
        
        prompt = prompt + self.default_prompt
        negative_prompt = negative_prompt + self.negative_prompt
        currentTask = worker.AsyncTask(args=[])

        inpaint_input_image = {'image':inpaint_input_image,
                               'mask':inpaint_mask_image}
        inpaint_mask_image = None
        ctrls = [currentTask, self.generate_image_grid]
        ctrls += [
            prompt, negative_prompt, self.style_selections,
            self.performance_selection, self.aspect_ratios_selection, self.image_number, self.output_format, self.image_seed,
            self.read_wildcards_in_order, self.sharpness, self.guidance_scale
        ]
        ctrls += [self.base_model, self.refiner_model, self.refiner_switch] + self.lora_ctrls
        ctrls += [self.input_image_checkbox, self.current_tab]
        ctrls += [self.uov_method, upsample_input_image]
        ctrls += [self.outpaint_selections,self.outpaint_extend_times, inpaint_input_image, self.inpaint_additional_prompt, inpaint_mask_image]
        ctrls += [self.disable_preview, self.disable_intermediate_results, self.disable_seed_increment, self.black_out_nsfw]
        ctrls += [self.adm_scaler_positive, self.adm_scaler_negative, self.adm_scaler_end, self.adaptive_cfg, self.clip_skip]
        ctrls += [self.sampler_name, self.scheduler_name, self.vae_name]
        ctrls += [self.overwrite_step, self.overwrite_switch, self.overwrite_width, self.overwrite_height, self.overwrite_vary_strength]
        ctrls += [self.overwrite_upscale_strength, self.mixing_image_prompt_and_vary_upscale, self.mixing_image_prompt_and_inpaint]
        ctrls += [self.debugging_cn_preprocessor, self.skipping_cn_preprocessor, self.canny_low_threshold, self.canny_high_threshold]
        ctrls += [self.refiner_swap_method, self.controlnet_softness]

        ctrls += self.freeu_ctrls
        ctrls += self.inpaint_ctrls

        if not args_manager.args.disable_metadata:
            ctrls += [self.save_metadata_to_images, self.metadata_scheme]
        ctrls += self.ip_ctrls
        return ctrls

    def _get_task(self,args):
        args.pop(0)
        return worker.AsyncTask(args=args)

    def _load_generator(self):
        self.worker = worker.Hack_worker()

    def _refine(self,
                image_number = 1,
                prompt = '', 
                negative_prompt = '',
                outpaint_selections=[],
                outpaint_extend_times=0.0,
                origin_image = None,
                mask_image = None,
                seed = None):
        '''
        origin_image numpy HW3 0-255
        mask_image numpy HW3 inpaint area be 255,255,255 else be 0,0,0
        '''
        # input check
        if np.amax(origin_image) < 1.1:
            origin_image = (origin_image*255).astype(np.uint8)
        if np.amax(mask_image) < 1.1:
            mask_image = (mask_image*255).astype(np.uint8)
        if mask_image.ndim < 3:
            mask_image = mask_image[:,:,None].repeat(3,axis=-1)
        
        # set seed
        self.image_number = image_number
        if seed is None: self._seed_set()
        else: self._seed_set(seed,seed_random=False)
        # conduct inpainting
        self._inpaint_parameters(outpaint_selections=outpaint_selections,
                                 outpaint_extend_times=outpaint_extend_times,
                                 inpaint_mode = modules.flags.inpaint_option_detail)
        ctrls = self._set_ctrls(prompt, negative_prompt, origin_image, mask_image)
        currentTask = self._get_task(ctrls)
        output = self._generate_images(currentTask)
        output = [np.array(o)/255. for o in output]
        return output

    def _upsample(self,
                 image :np.array = None,
                 image_number = 1,
                 prompt = '', 
                 negative_prompt = '',
                 control_strength = 0.382,
                 seed = None):
        # input check
        if np.amax(image) < 1.1:
            image = (image*255).astype(np.uint8)
        # set seed
        if seed is None: self._seed_set()
        else: self._seed_set(seed,seed_random=False)
        self.image_number=image_number
        # conduct inpainting
        self._upsample_parameteres()
        self.overwrite_upscale_strength = control_strength
        ctrls = self._set_ctrls(prompt, negative_prompt, upsample_input_image=image)
        currentTask = self._get_task(ctrls)
        output = self._generate_images(currentTask)
        output = [np.array(o)/255. for o in output]
        return output

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
        # input check
        if np.amax(origin_image) < 1.1:
            origin_image = (origin_image*255).astype(np.uint8)
        if np.amax(mask_image) < 1.1:
            mask_image = (mask_image*255).astype(np.uint8)
        if mask_image.ndim < 3:
            mask_image = mask_image[:,:,None].repeat(3,axis=-1)
            
        # set seed
        if seed is None: self._seed_set()
        else: self._seed_set(seed,seed_random=False)
        self.image_number=image_number
        # conduct inpainting
        self._inpaint_parameters(outpaint_selections=outpaint_selections,
                                 outpaint_extend_times=outpaint_extend_times,
                                 inpaint_mode = modules.flags.inpaint_option_default)
        ctrls = self._set_ctrls(prompt, negative_prompt, origin_image, mask_image)
        currentTask = self._get_task(ctrls)
        output = self._generate_images(currentTask)
        output = [np.array(o)/255. for o in output]
        return output
