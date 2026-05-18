import torch

from diffusion.refiner.vendor.ltx_core.loader.module_ops import ModuleOps
from diffusion.refiner.vendor.ltx_core.loader.sd_ops import SDOps
from diffusion.refiner.vendor.ltx_core.model.transformer.model import LTXModel


def compile_transformer(model: LTXModel) -> LTXModel:
    model.transformer_blocks = torch.nn.ModuleList(torch.compile(m) for m in model.transformer_blocks)

    def patched_dynamo_forward(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        with (
            torch._inductor.config.patch(unsafe_skip_cache_dynamic_shape_guards=True),
            torch._dynamo.config.patch(  # type: ignore[attr-defined]
                inline_inbuilt_nn_modules=True, cache_size_limit=256, allow_unspec_int_on_nn_module=True
            ),
        ):
            return model.forward_without_compilation(*args, **kwargs)

    model.forward_without_compilation = model.forward
    model.forward = patched_dynamo_forward
    return model


COMPILE_TRANSFORMER = ModuleOps(
    name="compile_transformer",
    matcher=lambda model: isinstance(model, LTXModel),
    mutator=lambda model: compile_transformer(model),
)


def modify_sd_ops_for_compilation(original_sd_ops: SDOps, number_of_blocks: int = 48) -> SDOps:
    for i in range(number_of_blocks):
        original_sd_ops = original_sd_ops.with_replacement(
            f"transformer_blocks.{i}.", f"transformer_blocks.{i}._orig_mod."
        )
    return original_sd_ops
