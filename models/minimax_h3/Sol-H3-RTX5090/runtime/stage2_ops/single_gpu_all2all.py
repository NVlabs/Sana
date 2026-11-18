"""Construction-scoped copy elision for the original singleton AV transfers."""
from contextlib import contextmanager
import math


class SingleGPUAll2All:
    """No IPC/CUDA allocation; intentionally limited to the audited AV route."""

    def __init__(self, rank, world_size, seqlen, hidden_dim, num_sms,
                 tensor_dtype, group=None, timeout_seconds=None, *, torch_module):
        if (rank != 0 or world_size != 1 or type(seqlen) is not int or seqlen < 1
                or hidden_dim != 4096 or type(num_sms) is not int or num_sms < 1
                or tensor_dtype != torch_module.bfloat16):
            raise ValueError("REFUSE: singleton route requires rank0/world1, BF16, hidden4096 and positive capacity/SMs")
        self.rank, self.world_size = rank, world_size
        self.seqlen, self.hidden_dim, self.num_sms = seqlen, hidden_dim, num_sms
        self.tensor_dtype, self.group = tensor_dtype, group
        # Logical original capacity, NOT an allocated buffer.
        self.buffer_size = seqlen * hidden_dim * 2
        self.rank_tokens = None
        self.destroyed = False
        self.timeout_seconds = 10.0
        if timeout_seconds is not None:
            self.set_timeout_seconds(timeout_seconds)

    def set_rank_tokens(self, rank_num_tokens):
        if (self.destroyed or len(rank_num_tokens) != 1
                or type(rank_num_tokens[0]) is not int
                or not 0 < rank_num_tokens[0] <= self.seqlen):
            raise ValueError("REFUSE: singleton token metadata invalid or object destroyed")
        # Called outside compiled blocks by set_seqlen_all2all; no GPU sync.
        self.rank_tokens = rank_num_tokens[0]

    def set_timeout_seconds(self, seconds):
        if self.destroyed or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("REFUSE: timeout must be finite/nonnegative on a live object")
        self.timeout_seconds = seconds

    def _identity(self, x, copy_out):
        # Metadata-only guards: no .item(), data_ptr(), global counters, CUDA
        # calls or state mutation inside the compiled hot path.
        if self.destroyed or self.rank_tokens is None:
            raise RuntimeError("REFUSE: set_rank_tokens required on a live singleton")
        if (x.ndim != 4 or not x.is_contiguous() or x.device.type != "cuda"
                or x.device.index != 0 or x.dtype != self.tensor_dtype
                or x.requires_grad):
            raise ValueError("REFUSE: expected contiguous CUDA:0 BF16 inference BTHD")
        if (x.shape[0] != 1 or x.shape[1] != self.rank_tokens
                or x.shape[2] != 32 or x.shape[3] not in (64, 128)
                or x.numel() * 2 > self.buffer_size):
            raise ValueError("REFUSE: unsupported BTHD shape, token metadata or capacity")
        if type(copy_out) is not bool:
            raise ValueError("REFUSE: copy_out must be bool")
        return x.clone() if copy_out else x

    def send_recv_heads(self, x, *, copy_out=False):
        return self._identity(x, copy_out)

    def gather_heads(self, x, *, copy_out=False):
        return self._identity(x, copy_out)

    def destroy(self):
        self.destroyed = True


@contextmanager
def single_gpu_all2all_factory(*, torch_module=None, kernels_module=None):
    """Patch exported constructor only; all multi-rank calls delegate unchanged.

    Yield a construction-only receipt. Existing shim instances stay valid
    after restoration. This never touches all_to_all.All2All or its registry.
    """
    if torch_module is None:
        import torch as torch_module
    if kernels_module is None:
        import ltx_kernels as kernels_module
    original = kernels_module.All2All
    receipt = {"single_gpu_instances": 0, "delegated_instances": 0,
               "original_data_capacity_bytes_elided": 0,
               "actual_cuda_allocations_by_shim": 0,
               "input_aliasing_copy_out_false": True,
               "factory_restored": False}

    def factory(rank, world_size, seqlen, hidden_dim, num_sms, tensor_dtype,
                group=None, timeout_seconds=None):
        kwargs = dict(rank=rank, world_size=world_size, seqlen=seqlen,
                      hidden_dim=hidden_dim, num_sms=num_sms,
                      tensor_dtype=tensor_dtype, group=group,
                      timeout_seconds=timeout_seconds)
        if world_size != 1:
            result = original(**kwargs)
            receipt["delegated_instances"] += 1
            return result
        result = SingleGPUAll2All(**kwargs, torch_module=torch_module)
        receipt["single_gpu_instances"] += 1
        receipt["original_data_capacity_bytes_elided"] += result.buffer_size
        return result

    kernels_module.All2All = factory
    try:
        yield receipt
    finally:
        kernels_module.All2All = original
        receipt["factory_restored"] = True
