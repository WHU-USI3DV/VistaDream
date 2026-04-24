## Define your camera trajectory.

### 1. Coordinate System Definition

In vistaDream, the coordinate system is a right-handed system where:
- Positive X points to the right,
- Positive Y points down,
- Positive Z points forward.

```
     / Z+
    /
   /
  +-------------> X+ 
  |
  | 
  |  
  v Y+  
```
### 2. Trajectory define 

In Vistadream, the base trajectory class ```Traj_Base``` is defined in ```ops/trajs/basic.py```. 

- The trajectory is defined as a sequence of cameras (position + pose). For each camera, the following parameters need to be provided: 
    - The camera position (Camera);
    - Look-at position (Target): a point along the camera's viewing direction
    - The camera's up vector (Up).

    The illustration of the above parameters are given in Fig-below(a).
    Moreover, As shown in Fig(b), the 'Up' direction does not necessarily align strictly with the camera's up vector. 
    It can be any direction within the upper half of the camera's plane, as long as the angle between it and the true up direction is less than 90 degrees. 
    This makes the 'Up' defination much simple.
    In general, when the camera is approximately vertical, the 'Up' vector can be simply defined as ```[0,-1,0]```.

    <img src="../../data/assert/traj.jpg" alt="Traj" style="zoom:50%;">

- Then, You can then define your own ```Traj``` class by inheriting from this ```Traj_Base``` class.
Examples are in ```ops/trajs/rot.py``` and ```ops/trajs/wobble.py```. 
In your class, you only need to define a List of cameras in the ```camera_target_up``` function, 
where each element in the list is a tuple, defined as ```(Camera, Target, Up)```


- **Tips**: ```Traj_Base``` owns a fundamental attribute that you might need to generate trajectories: ```radius```, which approximates the radius of the scene (excluding the sky).

