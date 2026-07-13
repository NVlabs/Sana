# DROID Annotations
This repo contains additional annotation data for the DROID dataset which we completed after the initial dataset release.

Concretely, it contains the following information:
- the full language annotations for the DROID dataset (3x annotations for 95+% of the 75k successful demonstrations)
- higher-accuracy camera calibrations (for 36k episodes)


## Language Annotations
The released RLDS dataset only contains a subset of the language labels. 
Here, we provide a single json file with a mapping from episode ID to three natural language annotations, 
for 75k success episodes in DROID (95% of all DROID success episodes).


## Camera Calibrations
The original DROID release included noisy camera calibrations. In post-hoc processing, we generated more accurate extrinsic camera calibration parameters
for a subset of the DROID episodes. Concretely, we provide the following three calibration files:
- `cam2base_extrinsics.json`: Contains ~36k entries with either the left or right camera calibrated with respect to base.
- `cam2cam_extrinsics.json`: Contains ~90k entries with cam2cam relative poses and camera parameters for all of DROID.
- `cam2base_extrinsic_superset.json`: Contains ~24k unique entries, total ~48k poses for both left and right camera calibrated with respect to the base.

These files map episodes' unique ID (see Accessing Annotation Data below) to another dictionary containing metadata (e.g., detection quality metrics, see Appendix G of paper), as well as a map from camera ID to the extrinsics values. Said extrinsics is represented as a 6-element list of floats, indicating the translation and rotation. It can be easily converted into a homogeneous pose matrix:
```
from scipy.spatial.transform import Rotation as R

# Assume extrinsics is that 6-element list
pos = extrinsics[0:3]
rot_mat = R.from_euler("xyz", extracted_extrinsics[3:6]).as_matrix()

# Make homogenous transformation matrix
cam_to_target_extrinsics_matrix = np.eye(4)
cam_to_target_extrinsics_matrix[:3, :3] = rot_mat
cam_to_target_extrinsics_matrix[:3, 3] = pos
```
This represents a transformation matrix from the camera's frame to the target frame. Inverting it gets the transformation from target frame to camera frame (which is usually desirable, e.g., if one wants to project a point in the robot frame into the camera frame).

As the raw DROID video files were recorded on Zed cameras and saved in SVO format, they contain camera intrinsics which can be used in conjunction with the above. For convenience, we have extracted and saved all these annotations to `intrinsics.json` (~72k entries). This `json` has the following format:
```
<episode ID>:
    <external camera 1's serial>: [fx, cx, fy, cy for camera 1]
    <external camera 2's serial>: [fx, cx, fy, cy for camera 2]
    <wrist camera 1's serial>: [fx, cx, fy, cy for wrist camera]
```
One can thus convert the list for a particular camera to a projection matrix via the following:
```
import numpy as np

# Assume intrinsics is that 4-element list
fx, cx, fy, cy = intrinsics
intrinsics_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
])
```
Note that the intrinsics tend to not change much between episodes, but using the specific values corresponding to a particular episode tends to give the best results.

## Example Calibration Use Case
Using the calibration information, one can project points in the robot's frame into pixel coordinates for the cameras. We will demonstrate how to map the robot gripper position to pixel coordinates for the external cameras with extrinsics in `cam2base_extrinsics.json`, see `CalibrationExample.ipynb` for the full code.
```
gripper_position_base = <Homogeneous gripper position in the base frame, as gotten from TFDS episode. Shape 4 x 1>
cam_to_base_extrinsics_matrix = <extrinsics matrix for some camera>
intrinsics_matrix = <intrinsics matrix for that same camera>

# Invert to get transform from base to camera frame
base_to_cam_extrinsics_matrix = np.linalg.inv(cam_to_base_extrinsics_matrix)

# Transform gripper position to camera frame, then remove homogeneous component
robot_gripper_position_cam = base_to_cam_extrinsics_matrix @ gripper_position_base
robot_gripper_position_cam = robot_gripper_position_cam[:3] # Now 3 x 1

# Project into pixel coordinates
pixel_positions = intrinsics_matrix @ robot_gripper_position_cam
pixel_positions = pixel_positions[:2] / pixel_positions[2] # Shape 2 x 1 # Done!
```

## Filtering Data
Many episodes in DROID contain significant pauses. This is an issue when training models, as these pauses typically happen at the start of episodes, causing the policy to likewise output idle actions when in the home position. To remediate this, we recommend filtering the data you train your policy on, removing all frames that map to idle actions.

We provide `keep_ranges_1_0_1.json` which maps episode keys to a list of time step ranges that should *not* be filtered out. The episode keys uniquely identify each episode, and are defined as `f"{recording_folderpath}--{file_path}"`. We opt for this unique identifier because both pieces of information are found in the episodes' RLDS metadata, and thus is easy to compute (even with TensorFlow symbolic operations).

To use this data, we recommend creating a `tf.lookup.StaticHashTable` identifying all frames that should not be filtered (with all other frames being filtered by default). Frames can be uniquely identified by simply concatenating their episode key with their time step within the episode. See [here](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/droid_rlds_dataset.py#L67) for an example implementation of how to use this `json` for filtering.

This particular filter `json` is meant for `droid/1.0.1`, NOT `droid/1.0.0`. It was computed by finding all continuous sequences in episodes of non-idle actions that are at least of length 16 (1 second of wallclock time) that are not interrupted by 8 or more idle actions.

## Accessing Annotation Data

All annotations are stored in `json` files which you can download from this repository.
To access the respective annotation for a particular episode, you can compute the episode's ID. 
It corresponds to the file name `metadata_<episode_id>.json` in every episode's folder in the raw DROID data. 
To extract episode IDs and corresponding file paths, you can use the code below. 
You can then e.g. match the filepath to the one stored in the RLDS dataset if you want to match language annotations / camera extrinsics to RLDS episodes.

```
import tensorflow as tf

episode_paths = tf.io.gfile.glob("gs://gresearch/robotics/droid_raw/1.0.1/*/success/*/*/metadata_*.json")
for p in episode_paths:
    episode_id = p[:-5].split("/")[-1].split("_")[-1]
```

As using the above annotations requires these episode IDs (but the TFDS dataset only contains paths), we have included `episode_id_to_path.json` for convenience. The below code snippet loads this `json`, then gets the mapping from episode paths to IDs.

```
import json
episode_id_to_path_path = "<path/to/episode_id_to_path.json>"
with open(episode_id_to_path_path, "r") as f:
    episode_id_to_path = json.load(f)
episode_path_to_id = {v: k for k, v in episode_id_to_path.items()}
```
