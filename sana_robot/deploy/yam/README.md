# YAM example

SANA-Robot on two [YAM](https://github.com/i2rt-robotics/i2rt) arms with three RealSense cameras. It is the
[client-server deployment](../README.md) with the robot's side filled in, as three processes on the robot's
workstation:

```
RealSense cameras ──> cameras.py ──> shared memory ──┐
                                                     ├──> client.py <── TCP ──> policy server (GPU machine)
YAM arms <── i2rt ──> follower.py <──── LCM ─────────┘
```

| File | What it is |
|---|---|
| [cameras.py](cameras.py) | Writes the newest RGB frame of every camera (480×640) to shared memory. |
| [follower.py](follower.py) | Drives the arms through the i2rt SDK at 300 Hz from the commands it receives over LCM, and publishes their state at 100 Hz. |
| [client.py](client.py) | `YamRobot`, the `Robot` of the control loop: frames from shared memory, state and commands over LCM. |
| [mapping.json](mapping.json) | The mapping of a YAM LeRobot dataset to the robot fields, for training and for the server. |

A state and a command are 14 values, as the dataset stores them: the six joints of the left arm, the six of the
right arm, then the opening of the left and of the right gripper (1 for open).

## 1. Setup on the workstation

```bash
git clone https://github.com/i2rt-robotics/i2rt.git && pip install -e i2rt
pip install lcm pyrealsense2
git clone -b robot https://github.com/NVlabs/Sana.git && export PYTHONPATH=$PWD/Sana
```

- Bring up the two CAN interfaces as the [i2rt instructions](https://github.com/i2rt-robotics/i2rt#installation)
  describe. The follower opens `can_left` and `can_right`; `--left_can` and `--right_can` name others.
- The follower was written against i2rt release v1.4.7 (commit `8c1b3ab`). Its gripper control reads and writes the SDK's `MotorChainRobot`
  directly (`_commands`, `_joint_state`, `_gripper_limits`, `remapper`), so another version of the SDK may need
  `command_arm` adapted.
- The serial numbers of the cameras: `rs-enumerate-devices -s`.

## 2. Data and policy

A YAM dataset in the LeRobot format trains with [mapping.json](mapping.json), which reads the joints and the
grippers from the 14 values of `observation.state` and `action`, and the cameras `top`, `left_wrist` and
`right_wrist`:

```bash
python -m sana_robot.data.utils.compute_normalization --dataset data/my_yam_dataset \
    --lerobot-mapping sana_robot/deploy/yam/mapping.json --output data/my_yam_dataset/normalization.json
bash scripts/train/train_ar.sh my_yam_recipe.yaml
```

where the recipe is one of [configs](../../../configs) with `data.path`, `data.normalization` and
`data.lerobot_mapping` set. The deploy config names the same three:

```yaml
policy_type: ar
checkpoint: checkpoints/my_yam_policy.pth
vae: checkpoints/LTX-2.3-Diffusers
text_encoder: checkpoints/gemma-2-2b-it
policy:
  spatial_layout: t_shape
  actions_per_chunk: 32
  sliding_window_chunks: 1
dataset: data/my_yam_dataset
lerobot_mapping: sana_robot/deploy/yam/mapping.json
normalization: data/my_yam_dataset/normalization.json
view_sizes: [[384, 480], [192, 256], [192, 256]]
chunk_size: 32
video_frame_stride: 4
steps: 4
```

With a LeRobot dataset the server converts by the mapping, so the client sends and receives the 14 values as the
dataset stores them.

## 3. Run

```bash
# GPU machine
bash scripts/deploy/server.sh my_yam_deploy.yaml --host 0.0.0.0 --port 8000 --instructions "['stack the three cups']"

# workstation, one terminal each
python -m sana_robot.deploy.yam.cameras --cameras top=<serial> left_wrist=<serial> right_wrist=<serial>
python -m sana_robot.deploy.yam.follower
python -m sana_robot.deploy.yam.client --host <GPU machine> --port 8000 --instruction "stack the three cups"
```

- The follower waits for the client. The client first holds the arms where they are for 3 s, in which the
  follower reaches its first target; the arms then follow the policy for `--control_steps` control steps and stay
  at the last command.
- `python -m sana_robot.deploy.yam.follower --sim` runs the SDK's simulated arms instead of the real ones.

## 4. What the processes do

- **Commands.** The policy gives one command per control step, at the dataset's frequency. The client sends the
  follower a target at 100 Hz that moves linearly from the command of the running step to that of the next, and
  holds a command when the plan ends with it or the next plan is late.
- **Grippers.** The follower drives a gripper by torque, not by position: a proportional-derivative term on the
  gripper's motor angle, limited to what 8 N at the fingertips need, plus a constant torque towards the target
  that overcomes the static friction of the drive. The limit bounds the force on a held object. The constants at
  the top of [follower.py](follower.py) are those of the `linear_4310` gripper.
- **Cameras.** The bridge leaves exposure and white balance as the cameras have them. Set them as they were when
  the training data was recorded.

The example has no safety layer: no limit on the speed of a command, no stop when a command is far from the
arms or when the cameras or the state stand still, and no key to stop. Add what your setup needs before you run
it on the arms.
