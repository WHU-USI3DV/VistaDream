import numpy as np
from .basic import Traj_Base

class Spiral(Traj_Base):
    def __init__(self, scene=None, nframe=100) -> None:
        super().__init__(scene, nframe)
        # special parameters for spiral
        self.rot_ratio = 0.3
        self.look_at_ratio = 0.5
        self.forward_ratio = self.scene.traj_forward_ratio
        self.backward_ratio = self.scene.traj_backward_ratio

    def camera_target_up(self):
        # get positions
        t = np.linspace(0, 1, self.nframe)
        r = np.sin(2 * np.pi * t) * self.radius * self.rot_ratio
        # rotation angles at each frame
        theta = 2 * np.pi * t * self.nframe
        # try not to change y (up-down for floor and sky)
        x = r * np.cos(theta)
        y = r * np.sin(theta) * 0.3 # here for
        z = -r
        z[z<0]*=self.forward_ratio
        z[z>0]*=self.backward_ratio
        pos = np.vstack([x,y,z]).T
        camera_ups = np.array([[0,-1.,0]]).repeat(self.nframe,axis=0)
        targets = np.array([[0,0,self.radius*self.look_at_ratio]]).repeat(self.nframe,axis=0)
        # camera_triples
        cameras = []
        for i in range(self.nframe):
            cameras.append((pos[i],targets[i],camera_ups[i]))
        return cameras

    def __call__(self):
        '''
        Spiral: A look down version
        Some modifications than the typical spiral for better sway.
        '''
        cameras = self.camera_target_up()
        trajs = self.trans_by_look_at(cameras)
        trajs = [np.linalg.inv(traj)[None] for traj in trajs]
        trajs.reverse()
        trajs = np.concatenate(trajs,axis=0)
        return trajs
