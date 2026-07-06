# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
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

"""Multi-GPU Bernini Renderer inference: data parallel + Ulysses sequence parallel.

Launch with torchrun. `--ulysses N` uses N GPUs of sequence parallel per
sample; the remaining `world_size / N` ranks run data parallel over the task
list (`--inputs`, or a single `--prompt`).

Example (8 GPUs, 8-way sequence parallel for one long video):
    torchrun --nproc-per-node 8 infer_multi_gpu.py \\
        --high_noise_ckpt <path> --low_noise_ckpt <path> --ulysses 8 \\
        --case assets/testcases/v2v/v2v.json

Example (8 GPUs, 2-way sequence parallel, 4-way data parallel over a batch):
    torchrun --nproc-per-node 8 infer_multi_gpu.py \\
        --high_noise_ckpt <path> --low_noise_ckpt <path> --ulysses 2 --inputs tasks.json
"""

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from bernini.cli import (
    add_common_args,
    apply_case_file,
    build_pipeline,
    generation_kwargs,
    load_tasks,
    resolve_system_prompt,
    setup_logging,
)
from bernini.parallel import init_parallel_state
from bernini.pipeline import BerniniPipeline


def rewrite_prompt(rewriter, task, default_task_type, ps):
    """Rewrite the prompt; under sequence parallel only the group's rank 0 calls
    the endpoint, then broadcasts the result to the rest of the group."""
    prompt = task["prompt"]
    if rewriter is None:
        return prompt
    task_type = task.get("task_type", default_task_type)
    media = dict(video=task.get("video"), image=task.get("image"), images=task.get("images"))
    if not ps.ulysses_enabled:
        return rewriter(task_type, prompt, **media)

    holder = [None]
    if ps.ulysses_rank == 0:
        holder[0] = rewriter(task_type, prompt, **media)
    src = ps.dp_rank * ps.ulysses_size  # global rank of this group's rank 0
    dist.broadcast_object_list(holder, src=src, group=ps.ulysses_group, device=torch.device("cpu"))
    return holder[0]


def main():
    parser = argparse.ArgumentParser(description="Bernini Renderer multi-GPU inference")
    add_common_args(parser)
    parser.add_argument("--ulysses", type=int, default=1, help="Ulysses sequence-parallel size")
    args = parser.parse_args()
    apply_case_file(args)
    setup_logging()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="cuda:nccl,cpu:gloo",
        timeout=timedelta(seconds=3600),
        rank=rank,
        world_size=world_size,
    )
    ps = init_parallel_state(ulysses_size=args.ulysses)
    pipeline = build_pipeline(args, device)

    rewriter = None
    if args.use_pe:
        from bernini.prompt_enhancer import PromptEnhancer

        rewriter = PromptEnhancer(model=args.pe_model)

    tasks = load_tasks(args)
    my_tasks = tasks[ps.dp_rank :: ps.dp_size]  # data-parallel split across Ulysses groups
    common = generation_kwargs(args)

    for task in my_tasks:
        prompt = rewrite_prompt(rewriter, task, args.task_type, ps)
        task_name = task.get("task_type", args.task_type)
        # BerniniPipeline takes task_name as first arg, BerniniRendererPipeline takes prompt
        if isinstance(pipeline, BerniniPipeline):
            pipeline(
                task_name,
                prompt,
                video=task.get("video"),
                image=task.get("image"),
                images=task.get("images"),
                output_path=task.get("output", args.output),
                write_output=(ps.ulysses_rank == 0),
                system_prompt=resolve_system_prompt(task, args),
                **common,
            )
        else:
            pipeline(
                prompt,
                video=task.get("video"),
                image=task.get("image"),
                images=task.get("images"),
                output_path=task.get("output", args.output),
                write_output=(ps.ulysses_rank == 0),
                system_prompt=resolve_system_prompt(task, args),
                **common,
            )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
