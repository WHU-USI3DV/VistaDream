"""VistaDream interactive reconstruction with MoGe-3 and LingBot-Depth."""
import argparse
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import torch
from PIL import Image
from gradio_model3dgscamera import Model3DGSCamera
from ops.utils.memory import release_memory
from ops.trajs import _generate_trajectory
from pipe.cfgs import load_cfg, PROJECT_ROOT
from pipe.vistadream_interface import Pipeline, get_extrinsic, save_scene, save_splat, load_scene

NEAR, FAR = 0.01, 100
TRAJ_TYPE = 'interp'
VIEWER_MAX_LONG_EDGE = 768
PIPELINE_CONCURRENCY_ID = "vistadream-pipeline"


def configure_app(config_path=None, output_dir=None, traj_type='interp'):
    global cfg, pipe, TEMP_DIR, COLOR_PATH, SCENE_PATH, SPLAT_PATH
    global SCENE_FINAL_PATH, RGB_VIDEO_PATH, DPT_VIDEO_PATH
    cfg = load_cfg(config_path)
    pipe = Pipeline(cfg, traj_type=traj_type, load_tool=False)
    directory = Path(output_dir) if output_dir else PROJECT_ROOT / 'temp' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    TEMP_DIR = str(directory.expanduser().resolve())
    Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
    COLOR_PATH = f'{TEMP_DIR}/color.png'
    SCENE_PATH = f'{TEMP_DIR}/scene.coarse.pth'
    SPLAT_PATH = f'{TEMP_DIR}/temp.splat'
    SCENE_FINAL_PATH = f'{TEMP_DIR}/scene.refine.pth'
    RGB_VIDEO_PATH = f'{TEMP_DIR}/temp.video_rgb.mp4'
    DPT_VIDEO_PATH = f'{TEMP_DIR}/temp.video_dpt.mp4'


configure_app()


def _require_scene():
    if pipe.scene is None:
        raise gr.Error('Upload an image and build the scaffold, or reload a saved scene first.')


def _invalidate_refinement():
    Path(SCENE_FINAL_PATH).unlink(missing_ok=True)
    for path in (RGB_VIDEO_PATH, DPT_VIDEO_PATH):
        Path(path).unlink(missing_ok=True)


def _prepare_image(img, seed):
    if img is None:
        raise gr.Error('Please upload an input image.')
    if not torch.cuda.is_available():
        raise gr.Error('VistaDream requires an NVIDIA GPU with CUDA.')
    pipe.scene = None
    release_memory()
    img = img.convert('RGB')
    img.save(COLOR_PATH)
    pipe.cfg.scene.input.rgb = COLOR_PATH
    pipe.cfg.scene.outpaint.seed = int(seed)
    # A new reconstruction replaces both coarse and refined results.
    Path(SCENE_PATH).unlink(missing_ok=True)
    _invalidate_refinement()
    return img


def _restore_scene(path):
    if not Path(path).is_file():
        raise gr.Error('No saved scene in this output directory. Build a scaffold first.')
    pipe.scene = None
    release_memory()
    saved_cfg, pipe.scene = load_scene(path)
    # Keep the shared config object used by the already loaded tools.
    pipe.cfg.scene = saved_cfg.scene
    pipe.cfg.scene.input.rgb = COLOR_PATH
    pipe.scene.cfg = pipe.cfg


def _viewer_camera_update(value=None):
    frame = pipe.scene.frames[0]
    scale = min(1.0, VIEWER_MAX_LONG_EDGE / max(frame.W, frame.H))
    width, height = max(1, round(frame.W * scale)), max(1, round(frame.H * scale))
    update = dict(width=width, height=height, camera_width=width, camera_height=height,
                  camera_fx=float(frame.intrinsic[0, 0]) * scale,
                  camera_fy=float(frame.intrinsic[1, 1]) * scale)
    if value is not None:
        update['value'] = value
    return gr.update(**update)


def interactive_build_scaffold(img, seed, text_prompt):
    img = _prepare_image(img, seed)
    pipe._build_scaffold_(img, user_prompt=text_prompt)
    save_scene(pipe.cfg, pipe.scene, SCENE_PATH, SPLAT_PATH)
    release_memory()
    return _viewer_camera_update(SPLAT_PATH)


def _camera_extrinsic(viewer):
    _require_scene()
    if viewer is None or len(viewer) < 3:
        raise gr.Error('Select a camera in the GS viewer first.')
    return get_extrinsic(viewer[1], viewer[2])


def interactive_render_splat(viewer):
    return pipe._interactive_render(_camera_extrinsic(viewer))


def interactive_render_and_inpaint(viewer, seed, anchor):
    extrinsic = _camera_extrinsic(viewer)
    pipe._ensure_tools()
    pipe.cfg.scene.outpaint.seed = int(seed)
    rgb = pipe._interactive_render_and_inpaint(extrinsic, anchor)
    _invalidate_refinement()
    save_scene(pipe.cfg, pipe.scene, SCENE_PATH, SPLAT_PATH)
    release_memory()
    return rgb, _viewer_camera_update(SPLAT_PATH)


def _interactive_step_back_():
    _require_scene()
    scaffold_frames = 2 if pipe.cfg.scene.outpaint.sign else 1
    if len(pipe.scene.frames) <= scaffold_frames:
        raise gr.Error('There is no added view to undo.')
    pipe.scene.frames.pop()
    pipe.scene.gaussian_frames.pop()
    pipe.scene.dense_trajs = _generate_trajectory(pipe.cfg, pipe.scene)
    _invalidate_refinement()
    save_scene(pipe.cfg, pipe.scene, SCENE_PATH, SPLAT_PATH)
    release_memory()
    return _viewer_camera_update(SPLAT_PATH)


def _interactive_MCS(steps, views, weight, gsopt_iters):
    _restore_scene(SCENE_PATH)
    pipe._ensure_tools()
    pipe._interactive_MCS_Refinement(int(steps), int(views), weight, int(gsopt_iters))
    save_scene(pipe.cfg, pipe.scene, SCENE_FINAL_PATH, SPLAT_PATH)
    release_memory()
    return _viewer_camera_update(SPLAT_PATH)


def _interactive_render_videos():
    _restore_scene(SCENE_FINAL_PATH if Path(SCENE_FINAL_PATH).exists() else SCENE_PATH)
    pipe.checkor._render_video(pipe.scene, save_dir=f'{TEMP_DIR}/temp.')
    release_memory()
    return RGB_VIDEO_PATH, DPT_VIDEO_PATH


def _Traj_Refine_(seed):
    _restore_scene(SCENE_PATH)
    pipe._ensure_tools()
    pipe.cfg.scene.outpaint.seed = int(seed)
    pipe._traj_refine_()
    _invalidate_refinement()
    save_scene(pipe.cfg, pipe.scene, SCENE_PATH, SPLAT_PATH)
    release_memory()
    return _viewer_camera_update(SPLAT_PATH)


def _E2E_VD_(img, text_prompt, seed):
    img = _prepare_image(img, seed)
    pipe._e2e_coarse_scene(img, user_prompt=text_prompt)
    save_scene(pipe.cfg, pipe.scene, SCENE_PATH, SPLAT_PATH)
    pipe._e2e_MCS_Refinement()
    save_scene(pipe.cfg, pipe.scene, SCENE_FINAL_PATH, SPLAT_PATH)
    pipe.checkor._render_video(pipe.scene, save_dir=f'{TEMP_DIR}/temp.')
    release_memory()
    return _viewer_camera_update(SPLAT_PATH), RGB_VIDEO_PATH, DPT_VIDEO_PATH


def _RESET_():
    pipe.scene = None
    if pipe.tools is not None:
        pipe.tools._unload_llava_()
        pipe.tools._unload_mcs_()
    release_memory()
    return None, None, None, None, None


def _RELOAD_():
    _restore_scene(SCENE_PATH)
    save_splat(pipe.scene, SPLAT_PATH)
    image = Image.open(COLOR_PATH).convert('RGB') if Path(COLOR_PATH).exists() else None
    return _viewer_camera_update(SPLAT_PATH), image, pipe.cfg.scene.outpaint.seed


# --------------------------
# Interface app
# --------------------------
with gr.Blocks(title="3D Viewer Interface",theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # VistaDream: Interactive Single-View Scene Reconstruction
        [ICCV 2025 paper](https://arxiv.org/abs/2410.16892) ·
        [Project page](https://vistadream-project-page.github.io/) ·
        [Original code](https://github.com/WHU-USI3DV/VistaDream)

        Haiping Wang, Yuan Liu, Ziwei Liu, Wenping Wang, Zhen Dong, Bisheng Yang

        Upload an image and **Build Scaffold**. Move the camera in **GS-Viewer**,
        preview a view, then **Inpaint the view** to add it. Select a few anchor
        views and **Generate a coarse scene**; use **Refine the scene by MCS**
        for refinement and export RGB/depth videos with the video button.

        Click the viewer to focus camera controls: drag to orbit, right/middle-drag
        to pan, scroll to zoom, WASD to move, Space/C to move vertically, Shift to
        accelerate. **Re-select the view** undoes the last added frame.

        **End-to-end VistaDream** uses a Spiral trajectory automatically.
        **ReSet** clears the viewer; **ReLoad** restores the saved coarse scene.
        """
    )

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
                placeholder='Describe the scene to guide inpainting. Leave blank to use the default prompt.'
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
        viewer = Model3DGSCamera(
            label="GS-Viewer(Down-sampled)",
            width=768,
            height=512,
            camera_width=768,
            camera_height=512,
            camera_fx=750,
            camera_fy=750,
            camera_near=NEAR,
            camera_far=FAR,
            interactive=True,
            elem_id="gs-viewer",
        )
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
        with gr.Accordion("VistaDream parameters", open=True):
            vd_seed = gr.Slider(minimum=0, maximum=2147483647261, value=31641916026, step=1, label="# Seed", interactive=True)
            mcs_steps = gr.Slider(minimum=2, maximum=16, value=9, step=1, label="# MCS Refine Steps", interactive=True)
            mcs_views = gr.Slider(minimum=5, maximum=20, value=10, step=2, label="# MCS Refine Views", interactive=True)
            mcs_weigt = gr.Slider(minimum=0, maximum=1., value=0.5, step=0.05, label="# MCS Refine Rectification Weight", interactive=True)
            mcs_gsopt_iters = gr.Slider(
                minimum=16,
                maximum=512,
                value=int(cfg.scene.mcs.gsopt_iters),
                step=16,
                label="# GS Optimization Iterations per MCS Step",
                interactive=True,
            )

    # task
    RES_btn.click(fn=_RESET_, outputs=[viewer,render_img,inpaint_img,rgb_video,dpt_video], concurrency_id=PIPELINE_CONCURRENCY_ID, concurrency_limit=1)
    REL_btn.click(fn=_RELOAD_,
                  outputs=[viewer,input_img,vd_seed],
                  concurrency_id=PIPELINE_CONCURRENCY_ID, concurrency_limit=1)
    E2E_btn.click(fn=_E2E_VD_,
                  inputs=[input_img,text_prompt,vd_seed],
                  outputs=[viewer,rgb_video,dpt_video],
                  concurrency_id=PIPELINE_CONCURRENCY_ID, concurrency_limit=1)
    MED_btn.click(fn=interactive_build_scaffold,
                  inputs=[input_img,vd_seed,text_prompt],
                  outputs=[viewer],
                  concurrency_id=PIPELINE_CONCURRENCY_ID, concurrency_limit=1)
    RENDER_btn.click(fn=interactive_render_splat,
                     inputs=[viewer],
                     outputs=[render_img],
                     concurrency_id=PIPELINE_CONCURRENCY_ID, concurrency_limit=1)
    INPAINT_btn.click(fn=interactive_render_and_inpaint,
                    inputs=[viewer,vd_seed,ANCHOR_cam],
                    outputs=[inpaint_img,viewer],
                    concurrency_id=PIPELINE_CONCURRENCY_ID, concurrency_limit=1)
    RETURN_btn.click(fn=_interactive_step_back_,
                    inputs=[],
                    outputs=[viewer],
                    concurrency_id=PIPELINE_CONCURRENCY_ID, concurrency_limit=1)
    VDREFINE_btn.click(fn=_Traj_Refine_,
                       inputs=[vd_seed],
                       outputs=[viewer],
                       concurrency_id=PIPELINE_CONCURRENCY_ID, concurrency_limit=1)
    REFINE_btn.click(fn=_interactive_MCS,
                     inputs=[mcs_steps,mcs_views,mcs_weigt,mcs_gsopt_iters],
                     outputs=[viewer],
                     concurrency_id=PIPELINE_CONCURRENCY_ID, concurrency_limit=1)
    VIDEO_btn.click(fn=_interactive_render_videos,
                    outputs=[rgb_video,dpt_video],
                    concurrency_id=PIPELINE_CONCURRENCY_ID, concurrency_limit=1)

# # --------------------------
# Launch
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='YAML config path (default: pipe/cfgs/basic.yaml)')
    parser.add_argument('--output-dir', help='Scene directory; reuse it with ReLoad after restarting')
    parser.add_argument('--traj-type', choices=['interp', 'spline'], default='interp')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--share', action='store_true')
    args = parser.parse_args()
    configure_app(args.config, args.output_dir, args.traj_type)
    mcs_gsopt_iters.value = int(cfg.scene.mcs.gsopt_iters)
    os.chdir(PROJECT_ROOT)
    print(f'Scene output directory: {TEMP_DIR}')
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=args.share,
        allowed_paths=[TEMP_DIR],
    )
