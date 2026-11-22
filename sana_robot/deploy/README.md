# Real-world deployment

SANA-Robot is deployed as two processes. The policy server holds the models on the GPU machine; the client is the
robot's control loop on the robot's machine and needs only NumPy and OpenCV.

```
robot machine (NumPy, OpenCV)                   GPU machine (SANA-Robot environment)
┌───────────────────────────────┐              ┌──────────────────────────────────┐
│ Robot            (you fill)   │              │ policy server                    │
│   read() -> frames, state     │     TCP      │   Deployment + ChunkRunner       │
│   send(command)               │ <──────────> │   RobotAdapter                   │
│ control loop + PolicyClient   │              │     state   -> robot row [80]    │
│                  (client.py)  │              │     action rows -> commands      │
└───────────────────────────────┘              └──────────────────────────────────┘
```

| File | Runs on | What it is |
|---|---|---|
| [server.py](server.py) | GPU machine | The policy server, and `RobotAdapter`: what it converts for your robot. |
| [client.py](client.py) | robot machine | A template: the protocol, `PolicyClient`, the control loop and the `Robot` you fill in. |
| [yam](yam/README.md) | robot machine | An example: the `Robot` of two YAM arms with RealSense cameras. |

## 1. Start the policy server

```bash
bash scripts/deploy/server.sh my_deploy.yaml --host 0.0.0.0 --port 8000
```

`my_deploy.yaml` is the deploy config of the main README. The server adds four fields to it:

| Field | Default | Meaning |
|---|---|---|
| `host` | `127.0.0.1` | The address to listen on; `0.0.0.0` accepts a client on another machine. |
| `port` | `8000` | The port to listen on. |
| `instructions` | `[]` | Tasks to prepare before the first client connects. |
| `warmup_steps` | `128` | Control steps of the episode of black frames a task is prepared with. |

A task is prepared by an episode of black frames, in which the CUDA graphs of its prompt are captured. A task that
is not in `instructions` is prepared when its first episode starts, before the first chunk is generated. On an
H100 the first task takes about 70 s, in which the video encoder is compiled, and every other task about 3 s. The
server serves one client at a time and keeps running when the client leaves.

## 2. Connect your robot

Copy [client.py](client.py) to the robot's machine, which needs `numpy` and `opencv-python`, and fill in the two
methods of `Robot`:

```python
class Robot:
    def read(self):
        """Return the newest frame of every camera, {camera: uint8 [H, W, 3]} in RGB, and the state vector."""

    def send(self, command, following):
        """Send the command of one control step to the robot; `following` is the command of the next step."""
```

```bash
python client.py --host <GPU machine> --port 8000 --instruction "put the red block into the box"
```

`PolicyClient(host, port).description` is what the policy expects, e.g. for the RoboDojo checkpoint:

```python
{"fps": 25.0,                    # the control frequency
 "video_frame_stride": 4,        # control steps between two frames the policy reads
 "cameras": {"cam_high": [384, 480], "cam_left_wrist": [192, 256], "cam_right_wrist": [192, 256]},
 "resize": "stretch",            # how a frame is brought to its size in `cameras`
 "async_delay_steps": 0,         # section 4
 "state_fields": [["left_joints", 6], ["left_eef_position", 3], ["left_eef_rotation", 6], ["left_gripper", 1],
                  ["right_joints", 6], ["right_eef_position", 3], ["right_eef_rotation", 6], ["right_gripper", 1]],
 "command_fields": [...]}        # the same fields
```

`Robot.read` may return the frames at the cameras' own resolution: the client resizes every frame to its size in
`cameras`, as the training data was resized, and sends it at that size. A state vector holds the values of
`state_fields` one after the other, and a command those of `command_fields`.

The control loop of `client.py` runs at `fps`: it sends one command per control step, together with the command of
the step after it for a controller that interpolates (None at the end of a plan), hands a frame to the server
every `video_frame_stride` steps, after the command of that step has gone out, and takes the next plan when the
running one is used up. It is a template. What
a robot needs beyond it is yours to add: limits on the commands, a stop, interpolation to the rate of your
controller, and what the robot does while it waits for a plan (the loop sends nothing then).

## 3. Convert your robot's state and commands

The policy reads the robot state and writes the actions as robot rows of 80 slots, filled as the training dataset
fills them ([robot80.py](../data/format/robot80.py)). `RobotAdapter` in [server.py](server.py) is the conversion
between those rows and what the client sends and receives. As written, the client speaks the dataset's own
values, as the fields above show; with a LeRobot dataset whose mapping reads the state from one column and the
action from one column, it speaks those columns as the dataset stores them, gripper readings included. Rewrite
its two methods when your robot reports or takes something else. For
example, the RoboDojo dataset holds for each arm the joint angles, the pose of the flange and the gripper
closedness, while the robot reports joint angles and a gripper opening and takes joint targets:

```python
import numpy as np

from sana_robot.data.format.robot80 import field_slots
from sana_robot.data.utils.convert_robodojo import ARX_X5_JOINTS, GRIPPER_CLOSED, GRIPPER_OPEN
from sana_robot.data.utils.kinematics import dual_arm_rows

JOINT_FIELDS = [["left_joints", 6], ["left_gripper", 1], ["right_joints", 6], ["right_gripper", 1]]


class RobotAdapter:
    def __init__(self, storage):
        (self.state_fields, self.command_fields) = (JOINT_FIELDS, JOINT_FIELDS)
        self.command_slots = [slot for (name, width) in JOINT_FIELDS for slot in field_slots(name, width)]

    def state_row(self, state):
        return dual_arm_rows(np.asarray(state)[None], ARX_X5_JOINTS, GRIPPER_OPEN, GRIPPER_CLOSED)[0][0]

    def commands(self, rows):
        return rows[:, self.command_slots]
```

`state_row` must give the row the converter of the training dataset gives for the same reading. The gripper
value of a command is here the dataset's closedness in [0, 1].

## 4. Asynchronous execution

`async_delay_steps` and `n_action_steps` of the deploy config set how the server schedules the chunks; the client
is the same for all of them.

- `async_delay_steps: 0` (default): the robot waits for every chunk. `n_action_steps` control steps of a chunk
  are run before the next one is generated, the whole chunk by default.
- `async_delay_steps: d`: the next chunk is generated while the robot runs `d` more control steps of the running
  one. SANA-Robot-AR uses memory RTC, which keeps one memory per execution phase so that every request reads a
  memory laid out as in training. The loop is real time when the latency `L` of a request satisfies
  `L ≤ d · Δt`, e.g. `d = 8` at 30 Hz for `L ≤ 267 ms`.

```bash
bash scripts/deploy/server.sh my_deploy.yaml --host 0.0.0.0 --port 8000 --async_delay_steps 8
```

## 5. Protocol

For a client in another language: a message is the 8 bytes `SANAROBT`, the size of the header (uint32) and of
the arrays (uint64) in network byte order, a JSON header and the bytes of the arrays. The header lists the arrays
in order as `{"name", "dtype", "shape"}` with NumPy dtype strings (`|u1`, `<f4`, `<f8`). A request holds
`method` and `arguments`; a reply holds `result`, or `error` when the request failed.

| Method | Arguments | Arrays | Reply |
|---|---|---|---|
| `describe` | | | The description of section 2. |
| `start` | `instruction` | `image/<camera>` for every camera, `state` | `chunk`, `generate_ms`; array `commands` [n, K] |
| `push_frame` | | `image/<camera>` for every camera, `state` | |
| `next_plan` | | | `chunk`, `generate_ms`; array `commands` [n, K] |

`start` opens an episode and returns the commands of its first chunk. `push_frame` hands over a frame taken
`video_frame_stride` control steps after the one before. `next_plan` returns the commands that follow the running
ones, waiting for them when they are late. A frame `image/<camera>` is uint8 [h, w, 3] in RGB at its size in
`cameras`, and `state` is float64.
