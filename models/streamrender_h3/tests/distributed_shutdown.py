"""CPU torchrun regression for the worker's UP-exit / teardown ordering."""
from runtime.config import mount_vendor
mount_vendor()
import torch.distributed as dist
from dev.yanzuolu.common.distributed.unified_parallel import init_unified_parallel, use_unified_parallel
from dev.yanzuolu.common.distributed.init import destroy_distributed

dist.init_process_group("gloo")
world = dist.get_world_size()
init_unified_parallel((1, world), 1)

def leave():
    with use_unified_parallel(world):
        dist.barrier()
        return

leave()
destroy_distributed()
assert not dist.is_initialized()
print("DISTRIBUTED_SHUTDOWN_PASS", flush=True)
