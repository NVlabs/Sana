# 💻 How to Inference & Test Metrics (FID, CLIP Score, GenEval, DPG-Bench, etc...)

This ToolKit will automatically inference your model and log the metrics results onto wandb as chart for better illustration. We curerntly support:

- [x] [FID](https://github.com/mseitzer/pytorch-fid) & [CLIP-Score](https://github.com/openai/CLIP)
- [x] [GenEval](https://github.com/djghosh13/geneval)
- [x] [DPG-Bench](https://github.com/TencentQQGYLab/ELLA)
- [x] [ImageReward](https://github.com/THUDM/ImageReward/tree/main)

### 0. Install corresponding env for GenEval and DPG-Bench

Make sure you can activate the following envs:

- `conda activate geneval`([GenEval](https://github.com/djghosh13/geneval))
- `conda activate dpg`([DGB-Bench](https://github.com/TencentQQGYLab/ELLA))

### 0.1 Prepare data.

Metirc FID & CLIP-Score on [MJHQ-30K](https://huggingface.co/datasets/playgroundai/MJHQ-30K)

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
  repo_id="playgroundai/MJHQ-30K",
  filename="mjhq30k_imgs.zip",
  local_dir="data/test/PG-eval-data/MJHQ-30K/",
  repo_type="dataset"
)
```

Unzip mjhq30k_imgs.zip into its per-category folder structure.

```
data/test/PG-eval-data/MJHQ-30K/imgs/
├── animals
├── art
├── fashion
├── food
├── indoor
├── landscape
├── logo
├── people
├── plants
└── vehicles
```

### 0.2 Prepare checkpoints

```bash
huggingface-cli download  Efficient-Large-Model/Sana_1600M_1024px --repo-type model --local-dir ./output/Sana_1600M_1024px --local-dir-use-symlinks False
```

### 1. directly [Inference and Metric] a .pth file

```bash
# We provide four scripts for evaluating metrics:
fid_clipscore_launch=scripts/bash_run_inference_metric.sh
geneval_launch=scripts/bash_run_inference_metric_geneval.sh
dpg_launch=scripts/bash_run_inference_metric_dpg.sh
image_reward_launch=scripts/bash_run_inference_metric_imagereward.sh

# Use following format to metric your models:
# bash $correspoinding_metric_launch $your_config_file_path $your_relative_pth_file_path

# example
bash $geneval_launch \
    configs/sana_config/1024ms/Sana_1600M_img1024.yaml \
    output/Sana_1600M_1024px/checkpoints/Sana_1600M_1024px.pth
```

### 2. [Inference and Metric] a list of .pth files using a txt file

You can also write all your pth files of a job in one txt file, eg. [model_paths.txt](../model_paths.txt)

```bash
# Use following format to metric your models, gathering in a txt file:
# bash $correspoinding_metric_launch $your_config_file_path $your_txt_file_path_containing_pth_path

# We suggest follow the file tree structure in our project for robust experiment
# example
bash scripts/bash_run_inference_metric.sh \
    configs/sana_config/1024ms/Sana_1600M_img1024.yaml \
    asset/model_paths.txt
```

### 3. You will get the following data tree.

```
output
├──your_job_name/  (everything will be saved here)
│  ├──config.yaml
│  ├──train_log.log

│  ├──checkpoints    (all checkpoints)
│  │  ├──epoch_1_step_6666.pth
│  │  ├──epoch_1_step_8888.pth
│  │  ├──......

│  ├──vis    (all visualization result dirs)
│  │  ├──visualization_file_name
│  │  │  ├──xxxxxxx.jpg
│  │  │  ├──......
│  │  ├──visualization_file_name2
│  │  │  ├──xxxxxxx.jpg
│  │  │  ├──......
│  ├──......

│  ├──metrics    (all metrics testing related files)
│  │  ├──model_paths.txt  Optional(👈)(relative path of testing ckpts)
│  │  │  ├──output/your_job_name/checkpoings/epoch_1_step_6666.pth
│  │  │  ├──output/your_job_name/checkpoings/epoch_1_step_8888.pth
│  │  ├──fid_img_paths.txt  Optional(👈)(name of testing img_dir in vis)
│  │  │  ├──visualization_file_name
│  │  │  ├──visualization_file_name2
│  │  ├──cached_img_paths.txt  Optional(👈)
│  │  ├──......
```
