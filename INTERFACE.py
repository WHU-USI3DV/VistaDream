import os
import time
import gradio as gr
from ops.gs.basic import *
from pipe.cfgs import load_cfg
from pipe.vistadream_interface import *
from gradio_model3dgscamera import Model3DGSCamera

# --------------------------
# constant settings
# --------------------------

# get time-stamp
t = time.localtime()
t = f'{int(t.tm_mon)}_{int(t.tm_mday)}_{int(t.tm_hour)}_{int(t.tm_min)}_{int(t.tm_sec)}'

# saving path
NEAR, FAR = 0.01, 100
TRAJ_TYPE = 'interp' # use interp or spline
TEMP_DIR = f'./temp/{t}'
COLOR_PATH = f'{TEMP_DIR}/color.png'
SCENE_PATH = f"{TEMP_DIR}/scene.coarse.pth"
SPLAT_PATH = f"{TEMP_DIR}/temp.splat"
SCENE_FINAL_PATH = f"{TEMP_DIR}/scene.refine.pth"
RGB_VIDEO_PATH = f"{TEMP_DIR}/temp.video_rgb.mp4"
DPT_VIDEO_PATH = f"{TEMP_DIR}/temp.video_dpt.mp4"
if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)

# basic tools
cfg = load_cfg(f'pipe/cfgs/basic.yaml')
# in case of corrupted fooocus ckpt
fooocus_path = f'{cfg.model.paint.fooocus.ckpts}/checkpoints/juggernautXL_v8Rundiffusion.safetensors'
if not os.path.exists(fooocus_path):
    if os.path.exists(fooocus_path + '.corrupted'):
        os.system(f'mv {fooocus_path}.corrupted {fooocus_path}')
pipe = Pipeline(cfg,traj_type=TRAJ_TYPE) # spline or interp
    
# pid, aviod killed
pid = os.getpid()
os.system(f'echo -1000 | sudo tee /proc/{pid}/oom_score_adj')

# --------------------------
# core processor
# --------------------------

def interactive_build_scaffold(img,seed,text_prompt):  
    img.save(COLOR_PATH)  
    pipe.cfg.scene.input.rgb = COLOR_PATH
    pipe.cfg.scene.outpaint.seed = seed
    if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)
    pipe._build_scaffold_(img,user_prompt=text_prompt)
    torch.cuda.empty_cache()
    save_scene(pipe.cfg,pipe.scene,SCENE_PATH,SPLAT_PATH)
    # pipe.tools.llava = None
    # torch.cuda.empty_cache()
    if os.path.exists(SCENE_FINAL_PATH): os.system(f'rm {SCENE_FINAL_PATH}') # re-start
    return \
        gr.update(width=pipe.scene.frames[0].W*2,
                  height=pipe.scene.frames[0].H*2,
                  camera_fx=pipe.scene.frames[0].intrinsic[0,0]*2,
                  camera_fy=pipe.scene.frames[0].intrinsic[0,0]*2),\
        str(SPLAT_PATH)

def interactive_render_splat(viewer):
    camera_position = viewer[1]
    camera_rotation = viewer[2]    
    extrinsic = get_extrinsic(camera_position,camera_rotation)
    rgb = pipe._interactive_render(extrinsic)
    return rgb
   
def interactive_render_and_inpaint(viewer,seed,anchor):
    # rendering partial images
    camera_position = viewer[1]
    camera_rotation = viewer[2]
    pipe.cfg.scene.outpaint.seed = seed
    extrinsic = get_extrinsic(camera_position,camera_rotation)
    rgb = pipe._interactive_render_and_inpaint(extrinsic,anchor)
    torch.cuda.empty_cache()
    save_scene(pipe.cfg,pipe.scene,SCENE_PATH,SPLAT_PATH)
    return rgb,str(SPLAT_PATH)

def _interactive_step_back_():
    scene = pipe.scene
    scene.frames = scene.frames[:-1]
    scene.gaussian_frames = scene.gaussian_frames[:-1]
    pipe.scene = scene
    save_scene(pipe.cfg,pipe.scene,SCENE_PATH,SPLAT_PATH)
    return str(SPLAT_PATH)

def _interactive_MCS(steps,views,weight):
    pipe.cfg,pipe.scene = load_scene(SCENE_PATH)
    pipe._interactive_MCS_Refinement(int(steps),int(views),weight)
    torch.cuda.empty_cache()
    save_scene(pipe.cfg,pipe.scene,SCENE_FINAL_PATH,SPLAT_PATH)
    return gr.update(width=pipe.scene.frames[0].W*2,
                     height=pipe.scene.frames[0].H*2,
                     camera_fx=pipe.scene.frames[0].intrinsic[0,0]*2,
                     camera_fy=pipe.scene.frames[0].intrinsic[0,0]*2),\
           str(SPLAT_PATH)

def _interactive_render_videos():
    if os.path.exists(SCENE_FINAL_PATH):
        pipe.cfg,pipe.scene = load_scene(SCENE_FINAL_PATH)
    else:
        pipe.cfg,pipe.scene = load_scene(SCENE_PATH)
    pipe.checkor._render_video(pipe.scene,save_dir=f"{TEMP_DIR}/temp.")
    return RGB_VIDEO_PATH, DPT_VIDEO_PATH

def _Traj_Refine_(seed):
    # already have several frames
    pipe.cfg.scene.outpaint.seed = seed
    pipe.cfg,pipe.scene = load_scene(SCENE_PATH)
    pipe._traj_refine_()
    save_scene(pipe.cfg,pipe.scene,SCENE_PATH,SPLAT_PATH)
    return gr.update(width=pipe.scene.frames[0].W*2,
                     height=pipe.scene.frames[0].H*2,
                     camera_fx=pipe.scene.frames[0].intrinsic[0,0]*2,
                     camera_fy=pipe.scene.frames[0].intrinsic[0,0]*2),\
           str(SPLAT_PATH)

def _E2E_VD_(img,text_prompt):
    pipe.scene = None
    # coarse
    pipe._e2e_coarse_scene(img,user_prompt=text_prompt)
    pipe.checkor._render_video(pipe.scene,save_dir=f"{TEMP_DIR}/temp.coarse.")
    save_scene(pipe.cfg,pipe.scene,SCENE_PATH,SPLAT_PATH)
    # refine
    pipe._e2e_MCS_Refinement()    
    pipe.checkor._render_video(pipe.scene,save_dir=f"{TEMP_DIR}/temp.")
    save_scene(pipe.cfg,pipe.scene,SCENE_FINAL_PATH,SPLAT_PATH)
    return str(SPLAT_PATH), RGB_VIDEO_PATH, DPT_VIDEO_PATH
    
def _RESET_():
    pipe.scene = None
    torch.cuda.empty_cache()
    
def _RELOAD_():
    if os.path.exists(SCENE_PATH):
        scene = torch.load(SCENE_PATH,weights_only=False)
        pipe.cfg,pipe.scene = scene['cfg'],scene['scene']
        image = Image.open(COLOR_PATH)
        return gr.update(width=pipe.scene.frames[0].W*2,
                  height=pipe.scene.frames[0].H*2,
                  camera_fx=pipe.scene.frames[0].intrinsic[0,0]*2,
                  camera_fy=pipe.scene.frames[0].intrinsic[0,0]*2),\
               image,SPLAT_PATH,pipe.cfg.scene.outpaint.seed
    
# --------------------------
# Interface app
# --------------------------
with gr.Blocks(title="3D Viewer Interface",theme=gr.themes.Soft()) as demo:
    # State indicator
    
    with gr.Row():
        gr.Markdown(
            """
            # [ICCV 2025] VistaDream: Sampling multiview consistent images for single-view scene reconstruction
            ### [Haiping Wang](https://hpwang-whu.github.io/), [Yuan Liu](https://liuyuan-pal.github.io/), [Ziwei Liu](https://liuziwei7.github.io/), [Wenping Wang](https://www.cs.hku.hk/people/academic-staff/wenping), [Zhen Dong](https://dongzhenwhu.github.io/index.html), [Bisheng Yang](https://3s.whu.edu.cn/info/1025/1415.htm)
            - Paper: https://arxiv.org/abs/2410.16892
            - Code: https://github.com/WHU-USI3DV/VistaDream
            - Project Page: https://vistadream-project-page.github.io/
            
            ### User Guidance:
            - First, upload the given image.
            - Second, click ``Build Scaffold'' to generate a scaffold on the given image. The Scaffold will be shown in ``GS-Viewer'' box.
            - Second, use ``Building Camera Trajectory'' Panel to design your own trajectory by selecting several anchor cameras. We suggest select less than 4 anchor views for VistaDream will interpolate more view by itself. To define a camera:
                - Move the camera in ``GS-Viewer'' box to select a ideal view by mouse moving and ``WASD''.
                - Click ``Preview the view'' to render the view for check. The rendered view will be shown above the button.
                - Click ``Inpaint the view'' to select and inpaint on this view. The inpainted view will be shown above the button.
                - Click ``Re-select the view'' if you find the inpainted view is not ideal.
            - After selecting sevel anchor views, click ``Generate a coarse scene'' button to generate a coarse gaussian scene, which will be shown in the ``GS-Viewer'' box.
            - [Optional] You can click ``Render RGB and Depth videos'' button to check the videos rendered from the coarse scene.
            - Then, click ``Refine the scene by MCS'' to optimize the scene by MCS method. In ``MCS parameters'' panel, you can mannually set the MCS parameters on number of views, MCS steps, and rectification weight.
            - You can click ``Render RGB and Depth videos'' button to check the videos rendered from the refined scene.
            - [Note] If you donnot wanna any human interaction, click ``End-to-end VistaDream'' after you upload the image. We will use ``Spiral'' trajectory by default and directly obtain the refined scene.
            
            ------------------
            
            """
        )
    
    current_splat = gr.State()
    # load inputs
    
    ########################## ALL RESULTS
    with gr.Row():
        gr.Markdown(
            """
            ### Input Image and Output Renderings
            """)
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label='Input Image',type='pil')
            text_prompt = gr.Textbox(
                label='Scene text prompt (optional)',
                lines=3,
                placeholder='If provided, this prompt is used directly and LLaVA captioning is skipped.'
            )
        with gr.Column():
            rgb_video = gr.Video(label="Output RGB renderings")
        with gr.Column():
            dpt_video = gr.Video(label="Output DPT renderings")

    with gr.Row():
        with gr.Column():
            RES_btn = gr.Button("ReSet",variant='primary')
        with gr.Column():
            REL_btn = gr.Button("ReLoad",variant='primary')
        with gr.Column():
            MED_btn = gr.Button("Build Scaffold",variant='primary')
        with gr.Column():
            VIDEO_btn = gr.Button("Rendering RGB and Depth videos", variant="primary")
    
    ########################### HUMAN TRAJECTORY
    with gr.Row():
        gr.Markdown(
            """
            ------------------------------
            ### Building Camera Trajectory
            """)
        
    with gr.Row():
        with gr.Column():
            viewer = Model3DGSCamera(
                label="GS-Viewer(Down-sampled)",
                width=1024,
                height=1024,
                camera_fx=1000,  
                camera_near=NEAR,
                camera_far=FAR,
                interactive=True)
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    render_img = gr.Image(label="Rendering result")
                with gr.Column():
                    inpaint_img = gr.Image(label="Inpainting result")
            with gr.Row():
                with gr.Column():
                    RENDER_btn = gr.Button("Preview the view", variant="primary")
                with gr.Column():
                    INPAINT_btn = gr.Button("Inpaint the view", variant="primary")
                with gr.Column():
                    ANCHOR_cam = gr.Checkbox(label="Select this camera as rendering anchor?", value=True)
                    RETURN_btn = gr.Button("Re-select the view", variant="primary")

    ########################### VistaDream
    with gr.Row():
        gr.Markdown(
            """
            ------------------------------
            ### Two-Stage VistaDream
            """)
        
    with gr.Row():
        with gr.Column():
            VDREFINE_btn = gr.Button("Generate a coarse scene", variant="primary")
        with gr.Column():
            REFINE_btn = gr.Button("Refine the scene by MCS", variant="primary")
        with gr.Column():
            E2E_btn = gr.Button("[No Interaction] End-to-end VistaDream",variant='primary')
    with gr.Row():
        with gr.Accordion("VistDream parameters", open=True):
            vd_seed = gr.Slider(minimum=0, maximum=2147483647261, value=31641916026, step=561, label="# Seed", interactive=True)
            mcs_steps = gr.Slider(minimum=0, maximum=12, value=9, step=1, label="# MCS Refine Steps", interactive=True)
            mcs_views = gr.Slider(minimum=5, maximum=15, value=10, step=2, label="# MCS Refine Views", interactive=True)
            mcs_weigt = gr.Slider(minimum=0, maximum=1., value=0.5, step=0.05, label="# MCS Refine Rectification Weight", interactive=True)        
            
    # task
    RES_btn.click(fn=_RESET_)
    REL_btn.click(fn=_RELOAD_,
                  outputs=[viewer,input_img,viewer,vd_seed])
    E2E_btn.click(fn=_E2E_VD_,
                  inputs=[input_img,text_prompt],
                  outputs=[viewer,rgb_video,dpt_video])
    MED_btn.click(fn=interactive_build_scaffold, 
                  inputs=[input_img,vd_seed,text_prompt], 
                  outputs=[viewer,viewer])
    RENDER_btn.click(fn=interactive_render_splat, 
                     inputs=[viewer], 
                     outputs=[render_img])
    INPAINT_btn.click(fn=interactive_render_and_inpaint, 
                    inputs=[viewer,vd_seed,ANCHOR_cam], 
                    outputs=[inpaint_img,viewer])
    RETURN_btn.click(fn=_interactive_step_back_, 
                    inputs=[], 
                    outputs=[viewer])
    VDREFINE_btn.click(fn=_Traj_Refine_,
                       inputs=[vd_seed],
                       outputs=[viewer,viewer])
    REFINE_btn.click(fn=_interactive_MCS,
                     inputs=[mcs_steps,mcs_views,mcs_weigt],
                     outputs=[viewer,viewer])
    VIDEO_btn.click(fn=_interactive_render_videos,
                    outputs=[rgb_video,dpt_video])

# # --------------------------
# Launch
# --------------------------
if __name__ == "__main__":
    demo.launch()
    