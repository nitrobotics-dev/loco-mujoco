# Fitting AMASS data to SMPL Model 

## Config File 

```
name: unitreeh1
mjcf_file: loco_mujoco/environments/data/unitree_h1/h1.xml
extend_config: 
  - joint_name: "left_hand_link"
    parent_name: "left_elbow_link"
    pos: [0.0, 0.0, 0.3]
    rot: [1.0, 0.0, 0.0, 0.0]
  - joint_name: "right_elbow_link"
    parent_name: "pelvis"
    pos: [0.0, 0.0, 0.3]
    rot: [1.0, 0.0, 0.0, 0.0]
  - joint_name: "head_link"
    parent_name: "pelvis"
    pos: [0.0, 0.0, 0.75]
    rot: [1.0, 0.0, 0.0, 0.0]
    
robot_joint_pick: ['pelvis',  'left_hip_yaw_link', "left_knee_link", "left_ankle_link",  'right_hip_yaw_link', 'right_knee_link', 'right_ankle_link', "left_shoulder_roll_link", "left_elbow_link",  "right_shoulder_roll_link", "right_elbow_link",  "head_link"]
smpl_joint_pick: ["Pelvis", "L_Hip",  "L_Knee", "L_Ankle",  "R_Hip", "R_Knee", "R_Ankle", "L_Shoulder", "L_Elbow",  "R_Shoulder", "R_Elbow",  "Head"]

smpl_pose_modifier:
  - L_Shoulder: "[0, 0, -np.pi/2]"
  - R_Shoulder: "[0, 0, np.pi/2]"
  - L_Elbow: "[0, -np.pi/2, 0]"
  - R_Elbow: "[0, np.pi/2, 0]"
```

`extend_config`: If you would like to add some extra joints to the robot, you can add the joint name, parent name, position and rotation to the `extend_config` list.
`robot_joint_pick` and `smpl_joint_pick`: the matching between the robot joints and the SMPL joints.
`smpl_pose_modifier`: the modifier for the SMPL pose to match the initial pose of the robot. 


## Set the Path To The AMASS Data and SMPL Model 



``` 
loco-mujoco-set-amass-path --path "/path/to/amass/data"
```

e.g., `/home/moore/AMASS/AMASS/SMPL+H G`. 

```
loco-mujoco-set-smpl-model-path --path "/path/to/smpl/models"
```
path to the smpl models. 


## Body Shape Fitting 
Using the following command to fit the body shape for your robot. 

```
python examples/data_process/fit_smpl_shape.py robot=unitree_h1
```


## Motion Fitting 


```
python examples/data_process/fit_smpl_motion.py robot=unitree_h1
```