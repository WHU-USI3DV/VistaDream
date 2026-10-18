import modules.config

def set_ckpt_dir(ckpts):
    modules.config.paths_checkpoints = [f'{ckpts}/checkpoints/']
    modules.config.paths_loras = [f'{ckpts}/loras/']
    modules.config.path_embeddings =  f'{ckpts}/embeddings/'
    modules.config.path_vae_approx = f'{ckpts}/vae_approx/'
    modules.config.path_vae = f'{ckpts}/vae/'
    modules.config.path_upscale_models = f'{ckpts}/upscale_models/'
    modules.config.path_inpaint = f'{ckpts}/inpaint/'
    modules.config.path_controlnet = f'{ckpts}/controlnet/'
    modules.config.path_clip_vision = f'{ckpts}/clip_vision/'
    modules.config.path_fooocus_expansion = f'{ckpts}/prompt_expansion/fooocus_expansion'
    modules.config.path_safety_checker = f'{ckpts}/safety_checker/'


    modules.config.update_files()
