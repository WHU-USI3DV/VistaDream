import numpy as np
import open3d as o3d
from ops.utils.utils import dpt2xyz

class Traj_Base():
    def __init__(self, 
                 scene = None,
                 nframe = 100) -> None:
        self.scene = scene
        self.nframe = nframe
        self.min_percentage = scene.traj_min_percentage
        self.max_percentage = scene.traj_max_percentage
        self._radius()

    def _radius(self):
        # get distribution
        sky = self.scene.frames[0].sky
        dpt = self.scene.frames[0].dpt
        intrinsic = self.scene.frames[0].intrinsic
        self.xyz = dpt2xyz(dpt,intrinsic)[~sky]
        if self.xyz.ndim > 2: self.xyz = self.xyz.reshape(-1,3)
        # get range
        _min = np.percentile(self.xyz,self.min_percentage,axis=0)
        _max = np.percentile(self.xyz,self.max_percentage,axis=0)
        _range = _max - _min
        # set radius to mean range of three axes
        self.radius = np.mean(_range)

    def rot_by_look_at(self, camera_position, target_position, camera_up):
        # look at direction
        direction = target_position - camera_position
        direction /= np.linalg.norm(direction)
        up = -camera_up # For the image origin is left-up: y is inverse
        up /= np.linalg.norm(up)
        # calculate rotation matrix
        right = np.cross(up, direction)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)
        rotation_matrix = np.column_stack([right, up, direction])
        return rotation_matrix

    def trans_by_look_at(self, camera_triples):
        '''
        camera_triples list: [(pos:numpy,target:numpy,up:numpy)]
        pos: camera position 
        target: look at position
        up: approximate camera up direction
        coor-system: z(forward) x(right) y(down)
        '''
        camera_poses = []
        for camera in camera_triples:
            pos,target,up = camera
            rotation_matrix = self.rot_by_look_at(pos,target,up)
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation_matrix
            transform_matrix[:3,  3] = pos
            camera_poses.append(transform_matrix[None])
        camera_poses = np.concatenate(camera_poses,axis=0)
        return camera_poses

    def camera_target_up(self):
        pass
    
    def __call__(self):
        camera_triples = self.camera_target_up()
        trajs = self.trans_by_look_at(camera_triples)
        return trajs
    
    def create_camera_geometry(self,pose):
        scale = self.radius * 0.1
        vertices = np.array([
            [0, 0, 0], 
            [-1, -1, 2], [1, -1, 2], [1, 1, 2], [-1, 1, 2],  
        ]) * scale
        vertices = np.hstack((vertices, np.ones((vertices.shape[0], 1))))  
        vertices = (pose @ vertices.T).T[:, :3]  

        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],  
            [1, 2], [2, 3], [3, 4], [4, 1], 
        ]
        colors = [[1, 0, 0] for _ in lines]  
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(vertices),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    def _visualize_traj(self,trajs):
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window()
        
        xyz = self.xyz.reshape(-1,3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        
        visualizer.add_geometry(pcd) 
        for traj in trajs:
            camera = self.create_camera_geometry(traj)
            visualizer.add_geometry(camera)

        visualizer.run()