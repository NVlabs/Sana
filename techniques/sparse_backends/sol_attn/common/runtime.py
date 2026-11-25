"""Small host helpers shared by the architecture backends."""

from cutlass.cute.runtime import from_dlpack


def to_cute_tensor(tensor, *, assumed_align=16):
    return from_dlpack(
        tensor,
        assumed_align=assumed_align,
        enable_tvm_ffi=True,
    ).mark_layout_dynamic(leading_dim=tensor.ndim - 1)


__all__ = ["to_cute_tensor"]
