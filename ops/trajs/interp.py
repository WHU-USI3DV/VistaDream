import scipy
import numpy as np
import scipy.interpolate
from .basic import Traj_Base

class Interp(Traj_Base):
    
    def __init__(self, scene=None, nframe=100):
        super().__init__(scene, nframe)
        # special
        self.spline_degree=5
        self.smoothness=.03
        self.rot_weight=.1
        
    def normalize(self, x):
        """Normalization helper function."""
        return x / np.linalg.norm(x)
    
    def viewmatrix(self, lookdir, up, position):
        """Construct lookat view matrix."""
        vec2 = self.normalize(lookdir)
        vec0 = self.normalize(np.cross(up, vec2))
        vec1 = self.normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m

    def poses_to_points(self, poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(self, points):
        """Converts from (position, lookat, up) format to pose matrices."""
        T = np.array([self.viewmatrix(p - l, u - p, p) for p, l, u in points])
        T = np.concatenate([T,np.array([[[0,0,0,1]]]*len(T))],axis=1)
        return T

    def interp(self, points, n):
        """Runs multidimensional Linear interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        part_n = (n // (sh[0]-1)) + 1
        new_points = []
        for i in range(sh[0]-1):
            part_pts = pts[i:i+2]
            func = scipy.interpolate.interp1d(np.linspace(0,1,2),part_pts,axis=0,kind='linear')
            part_points = np.array(func(np.linspace(0,1,part_n)))[:-1]
            new_points.append(part_points)
        new_points.append(pts[-1:])
        new_points = np.concatenate(new_points,axis=0)
        new_points = np.reshape(new_points, (new_points.shape[0], sh[1], sh[2]))
        return new_points

    def generate_interpolated_path(self, poses):
        """Creates a smooth spline path between input keyframe camera poses.

            Spline is calculated with poses in format (position, lookat-point, up-point).

            Args:
                poses: (n, 3, 4) array of input pose keyframes.
                n_interp: returned path will have n_interp * (n - 1) total poses.
                spline_degree: polynomial degree of B-spline.
                smoothness: parameter for spline smoothing, 0 forces exact interpolation.
                rot_weight: relative weighting of rotation/translation in spline solve.

            Returns:
                Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
        """
        points = self.poses_to_points(poses, dist=self.rot_weight)
        new_points = self.interp(points,self.nframe)
        poses = self.points_to_poses(new_points)
        return poses

    def __call__(self):
        poses = []
        anchor_frames = []
        for frame in self.scene.frames:
            if frame.keep or frame.anchor:
                anchor_frames.append(frame)
        N = len(anchor_frames)
        if N<2: return np.eye(4)[None].repeat(self.nframe,axis=0)
        
        # avoid adjactent same items
        bgn = 1 if anchor_frames[1].extrinsic[0,0] > 0.99 and \
                   anchor_frames[1].extrinsic[1,1] > 0.99 and \
                   anchor_frames[1].extrinsic[2,2] > 0.99 else 0
        anchor_frames = anchor_frames[bgn:]
        N = len(anchor_frames)
        if N<2: return np.eye(4)[None].repeat(self.nframe,axis=0)
        
        for i in range(N):
            frame = anchor_frames[i]
            poses.append(np.linalg.inv(frame.extrinsic[None]))
        poses = np.concatenate(poses,axis=0)
        trajs = self.generate_interpolated_path(poses)
        return trajs