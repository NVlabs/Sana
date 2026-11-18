"""Block streaming: memory-efficient sequential-block inference.
Streams transformer blocks from safetensors to GPU one at a time.
Block weights are provided by a :class:`WeightsProvider` which handles
CPU-to-GPU copies, caching, and stream synchronization.  Two weight
source strategies are available:
- **RAM streaming** (default): all blocks pre-loaded into pinned CPU
  buffers with LoRA fusion at build time.  Fast, higher CPU memory.
- **Disk streaming** (``cpu_slots < blocks_number``): blocks are read from
  disk on demand by a :class:`DiskWeightSource`, on a background worker
  thread.  Slower, lower CPU memory.
"""

from ltx_core.block_streaming.builder import DISK_CPU_SLOTS, StreamingModelBuilder
from ltx_core.block_streaming.wrapper import BlockStreamingWrapper

__all__ = [
    "DISK_CPU_SLOTS",
    "BlockStreamingWrapper",
    "StreamingModelBuilder",
]
