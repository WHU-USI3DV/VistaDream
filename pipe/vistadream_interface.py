'''
Input image:
1. Build Scaffold
2. Warp for check
3. Warp-and-inpaint
4. update 3D Viewer
5. MCS Refinement
6. Rendering views
7. Allowing Sparial End-to-end generation
'''

from ops.utils.utils import *
from ops.gs.basic import Frame
from ops.utils.visual_check import Check
from pipe.stages.scaffold import Scaffold_Phase
from pipe.stages.loadtools import Load_Tools_Phase
from pipe.stages.warpextend import WarpExtend_Phase
from ops.trajs import _generate_trajectory

class Pipeline():
    def __init__(self,cfg,traj_type='spline',load_tool=True) -> None:
        self.cfg = cfg
        self.scene = None
        self.tools = None
        self.checkor = Check()
        self.traj_type = traj_type
        self.cfg.scene.traj.traj_type = traj_type
        if load_tool:
            # MCS is loaded only when refinement starts. Preloading it alongside
            # Fooocus and depth models creates a large, unnecessary CPU-RAM peak.
            self.tools = Load_Tools_Phase(self.cfg,mcs=False)

    def _ensure_tools(self):
        if self.tools is None:
            self.tools = Load_Tools_Phase(self.cfg, mcs=False)

    def _resize_input(self,rgb):
        # INPUT IS PIL IMAGE
        rgb = np.asarray(rgb.convert("RGB"), dtype=np.float32) / 255.
        H,W = rgb.shape[0:2]
        resize_long_edge = int(self.cfg.scene.input.resize_long_edge)
        if H>W:
            W = int(W*resize_long_edge/H)
            H = resize_long_edge
        else:
            H = int(H*resize_long_edge/W)
            W = resize_long_edge
        rgb = cv2.resize(rgb,(W,H))
        return rgb

    def _build_scaffold_(self,rgb,user_prompt=None):
        self._ensure_tools()
        rgb = self._resize_input(rgb)
        self.scene = Scaffold_Phase(self.cfg,self.tools)(rgb,user_prompt=user_prompt)
        self.scene.traj_type=self.traj_type

    # Manually Selected Trajectory
    def _interactive_render(self,extrinsic):
        # rendering partial images
        H,W,intrinsic = self.scene.frames[0].H,self.scene.frames[0].W,self.scene.frames[0].intrinsic
        frame = Frame(H=H,W=W,intrinsic=intrinsic,extrinsic=extrinsic)
        frame = self.scene._render_for_inpaint(frame)
        rgb = (frame.rgb*255).clip(0,255).astype(np.uint8)
        return rgb

    def _interactive_render_and_inpaint(self,extrinsic,anchor=True):
        pose = np.linalg.inv(extrinsic)
        extentor = WarpExtend_Phase(self.cfg,self.tools)
        next_frame = extentor._pose_to_frame(self.scene,pose)
        # here is user-defined, set to anchor
        if anchor: print('This is an anchor view for final rendering.')
        next_frame.anchor = anchor
        self.scene = extentor._inpaint_next_frame(self.scene,next_frame)
        # we need to re-generate the trajectory
        # the anchor frames no more than 5 frames (include input and outpainted)
        self.scene.dense_trajs = _generate_trajectory(self.cfg,self.scene)
        rgb = (self.scene.frames[-1].rgb*255).clip(0,255).astype(np.uint8)
        return rgb

    def _interactive_MCS_Refinement(
        self,
        steps=None,
        views=None,
        weight=None,
        gsopt_iters=None,
    ):
        self.tools._load_mcs_()
        steps = steps if steps is not None else self.cfg.scene.mcs.steps
        views = views if views is not None else self.cfg.scene.mcs.n_view
        weight = weight if weight is not None else self.cfg.scene.mcs.rect_w
        gsopt_iters = (
            int(gsopt_iters)
            if gsopt_iters is not None
            else int(self.cfg.scene.mcs.gsopt_iters)
        )
        self.cfg.scene.mcs.gsopt_iters = gsopt_iters
        from pipe.stages.xmcsrefine import MCS_Phase
        try:
            self.scene = MCS_Phase(
                self.cfg,
                self.tools,
                self.scene,
                'cuda',
                steps,
                views,
                weight,
                gsopt_iters,
                offload_model=False,
            )()
        finally:
            self.tools._unload_mcs_()

    def _traj_refine_(self):
        # already have several frames, refine the trajectory by warp and inpaint
        extentor = WarpExtend_Phase(self.cfg,self.tools)
        self.scene = extentor._warp_and_inpaint(self.scene)
        self.scene = extentor._inpaint_holes(self.scene)

    def _e2e_coarse_scene(self,rgb,user_prompt=None):
        # scaffold stage
        self._build_scaffold_(rgb,user_prompt=user_prompt)
        self.scene.traj_type = "spiral"
        self.scene.dense_trajs = _generate_trajectory(self.cfg, self.scene)
        # coarse stage
        self.scene = WarpExtend_Phase(self.cfg,self.tools)(self.scene)

    def _e2e_MCS_Refinement(self):
        self.tools._load_mcs_()
        from pipe.stages.xmcsrefine import MCS_Phase
        try:
            self.scene = MCS_Phase(
                self.cfg,
                self.tools,
                self.scene,
                'cuda',
                offload_model=False,
            )()
        finally:
            self.tools._unload_mcs_()

# ------------------------------------ Utils -----------------------------------------

def get_extrinsic(camera_position,camera_rotation):
    t = np.asarray(camera_position, dtype=np.float64)
    euler = np.asarray(camera_rotation, dtype=np.float64)
    cx = np.cos(euler[0])
    sx = np.sin(euler[0])
    cy = np.cos(euler[1])
    sy = np.sin(euler[1])
    cz = np.cos(euler[2])
    sz = np.sin(euler[2])
    Rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx],
    ])
    Ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy],
    ])
    Rz = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1],
    ])

    # The viewer serializes its quaternion with the standard XYZ Euler
    # decomposition, whose column-vector rotation is Rz @ Ry @ Rx.
    camera_to_world_rotation = Rz @ Ry @ Rx
    world_to_camera_rotation = camera_to_world_rotation.T
    view_mtx = np.eye(4, dtype=np.float64)
    view_mtx[:3, :3] = world_to_camera_rotation
    view_mtx[:3, 3] = -world_to_camera_rotation @ t
    return view_mtx

def save_splat(scene,SPLAT_PATH):
    # 1. save to splat file
    xyz       = torch.cat([gf.xyz.reshape(-1,3) for gf in scene.gaussian_frames],dim=0)
    rgb       = torch.cat([gf.rgb.reshape(-1,3) for gf in scene.gaussian_frames],dim=0)
    scale     = torch.cat([gf.scale.reshape(-1,3) for gf in scene.gaussian_frames],dim=0)
    opacity   = torch.cat([gf.opacity.reshape(-1) for gf in scene.gaussian_frames],dim=0)
    rotation  = torch.cat([gf.rotation.reshape(-1,4) for gf in scene.gaussian_frames],dim=0)

    # 2. convert it to right range
    rgb = torch.sigmoid(rgb)
    scale = torch.exp(scale)
    opacity = torch.sigmoid(opacity)
    rgb_opacity = torch.cat((rgb,opacity[:,None]),dim=1)

    xyz = xyz.detach().cpu().numpy()
    scale = scale.detach().cpu().numpy()
    rgb_opacity = rgb_opacity.detach().cpu().numpy()
    rotation = rotation.detach().cpu().numpy()

    select = np.random.permutation(len(xyz))[0:150000]
    xyz = xyz[select]
    scale = scale[select]*1.5
    rotation = rotation[select]
    rgb_opacity = rgb_opacity[select]

    with open(SPLAT_PATH, "wb") as f:
        for p, s, c, r in zip(xyz,scale,rgb_opacity,rotation):
            f.write(p.astype(np.float32).tobytes())
            f.write(s.astype(np.float32).tobytes())
            f.write((c*255).clip(0, 255).astype(np.uint8).tobytes())
            f.write((r*128+128).clip(0, 255).astype(np.uint8).tobytes())

def save_scene(cfg,scene,SCENE_PATH,SPALT_PATH):
    save_splat(scene,SPALT_PATH)
    scene = {'cfg':cfg,'scene':scene}
    torch.save(scene,SCENE_PATH)

def load_scene(SCENE_PATH):
    scene = torch.load(SCENE_PATH,weights_only=False)
    return scene['cfg'], scene['scene']
