from dataclasses import dataclass

from ltx_core.loader.fuse_loras import FuseRule, bf16_fuse_rule
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_ops import SDOps
from ltx_core.model.model_protocol import ModelConfigurator
from ltx_core.model.transformer.model import LTXModel


@dataclass(frozen=True)
class QuantizationPolicy:
    """Configuration for model quantization during loading.
    Attributes:
        sd_ops: State-dict operations applied to each tensor during load.
        module_ops: Post-load module transformations applied to the meta model.
        model_configurator: Configurator class to use when constructing the transformer.
        fuse_rule: How LoRA deltas merge into this policy's weight layout.
            Default ``bf16_fuse_rule`` is used when no policy is configured.
    """

    sd_ops: SDOps | None = None
    module_ops: tuple[ModuleOps, ...] = ()
    model_configurator: type[ModelConfigurator[LTXModel]] | None = None
    fuse_rule: FuseRule = bf16_fuse_rule
