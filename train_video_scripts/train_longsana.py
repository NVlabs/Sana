# ruff: noqa: E402

import argparse
import os

# Importing the shared Stage-1 helpers must not disable xFormers for legacy
# LongSANA trainers. The WM branch opts out explicitly after config dispatch.
_disable_xformers_before_wm_import = os.environ.get("DISABLE_XFORMERS")
_tokenizers_parallelism_before_wm_import = os.environ.get("TOKENIZERS_PARALLELISM")
os.environ.setdefault("DISABLE_XFORMERS", "0")

import wandb
from omegaconf import OmegaConf

import diffusion.model.nets.sana_blocks as sana_blocks
import diffusion.model.nets.sana_multi_scale_video_camctrl as sana_video_camctrl
from diffusion.longsana.trainer.longsana_trainer import LongSANATrainer
from diffusion.longsana.trainer.ode import ODESANATrainer
from diffusion.longsana.trainer.sana_wm_distill import SanaWMDistillTrainer
from diffusion.longsana.trainer.self_forcing_trainer import Trainer as SelfForcingScoreDistillationTrainer

if _disable_xformers_before_wm_import is None:
    os.environ.pop("DISABLE_XFORMERS", None)
if _tokenizers_parallelism_before_wm_import is None:
    os.environ.pop("TOKENIZERS_PARALLELISM", None)
del _disable_xformers_before_wm_import
del _tokenizers_parallelism_before_wm_import


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument(
        "--wandb-save-dir", type=str, default="./wandb", help="Path to the directory to save wandb logs"
    )
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument(
        "--no-auto-resume", action="store_true", help="Disable auto resume from latest checkpoint in logdir"
    )
    parser.add_argument("--max_iters", type=int, default=10000, help="Maximum number of iterations")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb name")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/sana_video_config/longsana/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    # get the filename of config_path
    config_name = os.path.dirname(args.config_path).split("/")[-1] if args.wandb_name is None else args.wandb_name
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    config.disable_wandb = args.disable_wandb
    config.auto_resume = not args.no_auto_resume
    config.max_iters = args.max_iters

    if config.trainer == "self_forcing":
        trainer = SelfForcingScoreDistillationTrainer(config)
    elif config.trainer == "longsana":
        trainer = LongSANATrainer(config)
    elif config.trainer == "ode":
        trainer = ODESANATrainer(config)
    elif config.trainer in {"wm_ode", "wm_self_forcing"}:
        os.environ["DISABLE_XFORMERS"] = "1"
        sana_blocks._xformers_available = False
        sana_video_camctrl._xformers_available = False
        config.mode = config.trainer.removeprefix("wm_")
        config.wandb_name = args.wandb_name
        trainer = SanaWMDistillTrainer(config)
    else:
        raise ValueError(f"Unknown trainer: {config.trainer}")
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
