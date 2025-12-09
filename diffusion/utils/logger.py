# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pytz
import torch.distributed as dist
from mmcv.utils.logging import logger_initialized
from termcolor import colored

from .dist_utils import is_local_master


def get_root_logger(
    log_file=None, log_level=logging.INFO, name=colored("[Sana]", attrs=["bold"]), timezone="Asia/Shanghai"
):
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
        name (str): logger name
    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    if log_file is None:
        log_file = "/dev/null"
    logger = get_logger(name=name, log_file=log_file, log_level=log_level, timezone=timezone)
    return logger


class TimezoneFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, tz=None):
        super().__init__(fmt, datefmt)
        self.tz = pytz.timezone(tz) if tz else None

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, self.tz)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            s = dt.isoformat()
        return s


def get_logger(name, log_file=None, log_level=logging.INFO, timezone="UTC"):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        timezone (str): Timezone for the log timestamps.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    logger.propagate = False  # disable root logger to avoid duplicate logging

    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, "a")
        handlers.append(file_handler)

    formatter = TimezoneFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", tz=timezone
    )

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    # only rank0 for each node will print logs
    log_level = log_level if is_local_master() else logging.ERROR
    logger.setLevel(log_level)

    logger_initialized[name] = True

    return logger


def rename_file_with_creation_time(file_path):
    # 获取文件的创建时间
    creation_time = os.path.getctime(file_path)
    creation_time_str = datetime.fromtimestamp(creation_time).strftime("%Y-%m-%d_%H-%M-%S")

    # 构建新的文件名
    dir_name, file_name = os.path.split(file_path)
    name, ext = os.path.splitext(file_name)
    new_file_name = f"{name}_{creation_time_str}{ext}"
    new_file_path = os.path.join(dir_name, new_file_name)

    # 重命名文件
    os.rename(file_path, new_file_path)
    # print(f"File renamed to: {new_file_path}")
    return new_file_path


class TimezoneFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, tz=None):
        super().__init__(fmt, datefmt)
        self.tz = pytz.timezone(tz) if tz else None

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, self.tz)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            s = dt.isoformat()
        return s


class LogBuffer:
    def __init__(self):
        self.val_history = OrderedDict()
        self.n_history = OrderedDict()
        self.output = OrderedDict()
        self.ready = False

    def clear(self) -> None:
        self.val_history.clear()
        self.n_history.clear()
        self.clear_output()

    def clear_output(self) -> None:
        self.output.clear()
        self.ready = False

    def update(self, vars: dict, count: int = 1) -> None:
        assert isinstance(vars, dict)
        for key, var in vars.items():
            if key not in self.val_history:
                self.val_history[key] = []
                self.n_history[key] = []
            self.val_history[key].append(var)
            self.n_history[key].append(count)

    def average(self, n: int = 0) -> None:
        """Average latest n values or all values."""
        assert n >= 0
        for key in self.val_history:
            values = np.array(self.val_history[key][-n:])
            nums = np.array(self.n_history[key][-n:])
            avg = np.sum(values * nums) / np.sum(nums)
            self.output[key] = avg
        self.ready = True


def tracker(args, result_dict, label="", pattern="epoch_step", metric="FID"):
    if args.report_to == "wandb":
        import wandb

        wandb_name = f"[{args.log_metric}]_{args.name}"
        wandb.init(project=args.tracker_project_name, name=wandb_name, resume="allow", id=wandb_name, tags="metrics")
        run = wandb.run
        if pattern == "step":
            pattern = "sample_steps"
        elif pattern == "epoch_step":
            pattern = "step"
        custom_name = f"custom_{pattern}"
        run.define_metric(custom_name)
        # define which metrics will be plotted against it
        run.define_metric(f"{metric}_{label}", step_metric=custom_name)

        steps = []
        results = []

        def extract_value(regex, exp_name):
            match = re.search(regex, exp_name)
            if match:
                return match.group(1)
            else:
                return "unknown"

        for exp_name, result_value in result_dict.items():
            if pattern == "step":
                regex = r".*step(\d+)_scale.*"
                custom_x = extract_value(regex, exp_name)
            elif pattern == "sample_steps":
                regex = r".*step(\d+)_size.*"
                custom_x = extract_value(regex, exp_name)
            else:
                regex = rf"{pattern}(\d+(\.\d+)?)"
                custom_x = extract_value(regex, exp_name)
                custom_x = 1 if custom_x == "unknown" else custom_x

            assert custom_x != "unknown"
            steps.append(float(custom_x))
            results.append(result_value)

        sorted_data = sorted(zip(steps, results))
        steps, results = zip(*sorted_data)

        for step, result in sorted(zip(steps, results)):
            run.log({f"{metric}_{label}": result, custom_name: step})
    else:
        print(f"{args.report_to} is not supported")
