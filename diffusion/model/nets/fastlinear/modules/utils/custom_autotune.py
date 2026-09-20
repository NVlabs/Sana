# Copyright 2024 MIT Han Lab
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

import builtins
import os
import pickle
import time

import torch
import torch.distributed as dist
from triton.runtime.autotuner import Autotuner


class CustomAutotuner(Autotuner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use device name if CUDA is available, otherwise use a default name
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0).replace(" ", "_")
        else:
            device_name = "cpu"

        self.best_config_cache_path = os.path.expanduser(
            os.path.join(
                "~",
                ".triton",
                "best_config_cache",
                device_name,
                self.base_fn.__name__ + ".pkl",
            )
        )
        if os.path.exists(self.best_config_cache_path):
            with open(self.best_config_cache_path, "rb") as f:
                self.cache = pickle.load(f)

    def run(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        used_cached_result = True
        if len(self.configs) > 1:
            all_args = {**self.nargs, **kwargs}
            _args = []
            for name in self.arg_names:
                if name in all_args:
                    _args.append(all_args[name])
            key = [_args[i] for i in self.key_idx]
            for arg in _args:
                if hasattr(arg, "dtype"):
                    key.append(str(arg.dtype))
            key = tuple(key)
            if key not in self.cache:
                # prune configs
                used_cached_result = False
                pruned_configs = self.prune_configs(kwargs)
                bench_start = time.time()
                timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
                bench_end = time.time()
                self.bench_time = bench_end - bench_start
                self.cache[key] = builtins.min(timings, key=timings.get)
                self.pre_hook(args, reset_only=True)
                self.configs_timings = timings
                if not dist.is_initialized() or dist.get_rank() == 0:
                    best_config_cache_dir = os.path.dirname(self.best_config_cache_path)
                    os.makedirs(best_config_cache_dir, exist_ok=True)
                    with open(self.best_config_cache_path, "wb") as f:
                        pickle.dump(self.cache, f)
            config = self.cache[key]
        else:
            config = self.configs[0]
        self.best_config = config
        if os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1" and not used_cached_result:
            print(
                f"Triton autotuning for function {self.base_fn.__name__} finished after "
                f"{self.bench_time:.2f}s; best config selected: {self.best_config};"
            )
        if config.pre_hook is not None:
            config.pre_hook({**self.nargs, **kwargs, **config.all_kwargs()})
        ret = self.fn.run(
            *args,
            **kwargs,
            **config.all_kwargs(),
        )
        self.nargs = None
        return ret


def custom_autotune(
    configs,
    key,
    prune_configs_by=None,
    reset_to_zero=None,
    restore_value=None,
    pre_hook=None,
    post_hook=None,
    warmup=25,
    rep=100,
    use_cuda_graph=False,
):
    def decorator(fn):
        return CustomAutotuner(
            fn,
            fn.arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook=pre_hook,
            post_hook=post_hook,
            prune_configs_by=prune_configs_by,
            warmup=warmup,
            rep=rep,
            use_cuda_graph=use_cuda_graph,
        )

    return decorator
