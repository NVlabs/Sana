# RoboTwin 2.0 evaluation

Closed-loop evaluation of SANA-Robot-AR RoboTwin 2.0 with RoboTwin's own `script/eval_policy.py`: 50 tasks in the
`demo_clean` and `demo_randomized` settings, 100 episodes each. The simulator and the policy run as two processes,
each in its own environment: RoboTwin loads [sana_robot_policy.py](sana_robot_policy.py), which sends the frames of the
head and wrist cameras and the joint readings to the policy server on a local port and runs the end-effector
commands it gets back, 16 control steps per chunk of 32.

## 1. Setup

Install [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) and its assets as its documentation
describes, in its own environment. The numbers below were measured at commit `bf44be5` (the copy in
[FastWAM](https://github.com/yuantianyuan01/FastWAM) `7faa711`):

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin && git -C RoboTwin checkout bf44be5
```

The checkpoint comes with the description of its training data and the normalization of its robot rows:

```
checkpoints/sana_robot_ar_robotwin/
├── sana_robot_ar_robotwin.pth
├── meta.json
└── normalization.json
```

## 2. Run

Start the policy server in the SANA-Robot environment:

```bash
python -m eval.robotwin.server --config_path eval/robotwin/deploy.yaml --port 8000
```

Then evaluate a task from the RoboTwin directory, in the RoboTwin environment:

```bash
SANA_ROBOT=/path/to/sana-robot
PYTHONPATH=$SANA_ROBOT/eval/robotwin python script/eval_policy.py \
    --config $SANA_ROBOT/eval/robotwin/sana_robot_policy.yml \
    --overrides --task_name adjust_bottle --task_config demo_clean --port 8000
```

RoboTwin writes the success rate of the task under `eval_result/`. On one H100 a control step takes about half a
second, an episode one to ten minutes. A server answers one evaluation at a time; to use several GPUs, start one
server per GPU on its own port.

## 3. Results

20 denoising steps, video guidance 6, seed 0:

| Setting           | Success rate (%) |
|-------------------|------------------|
| `demo_clean`      | 91.84            |
| `demo_randomized` | 73.28            |
