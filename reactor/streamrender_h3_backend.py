"""Native two-step renderer, with distributed processes owned by the caller."""

import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile


def continuous_time_settings(settings: dict) -> dict:
    """Keep H3's normalized diffusion clock continuous after JSON loading.

    The upstream timestep class treats an integer T as a discrete schedule.
    With T=1 that rounds both denoising stages to 1. Float horizons preserve
    the released two-step shifts (video: 6, audio: 1.5).
    """
    for specification in settings["inference"]["diffusion"].values():
        args = specification["args"]
        if "T" in args:
            args["T"] = float(args["T"])
    return settings


class StreamRenderBackend:
    def __init__(self):
        self.pipe = None

    def load(self, source_root: Path, assets_path: Path) -> None:
        root = Path(source_root).resolve() / "models/streamrender_h3"
        spec = importlib.util.spec_from_file_location(
            "streamrender_native_config", root / "runtime/config.py"
        )
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        settings, _ = config_module.load_config(
            root / "configs/runtime.json", assets_path
        )
        continuous_time_settings(settings)
        sys.path.insert(0, str(root))
        config_module.mount_vendor()

        import torch
        import torch.distributed as dist
        from dev.yanzuolu.common.config import CfgNode
        from dev.yanzuolu.common.distributed.unified_parallel import (
            init_unified_parallel,
        )
        from dev.yanzuolu.common.model import build_models
        from dev.yanzuolu.common.seed import seed_everything
        from render.pipeline import StreamingPipeline

        # DistributedRunner already binds CUDA and creates the process group.
        # Native configure_distributed would assert and initialize it a second time.
        if not dist.is_initialized() or dist.get_world_size() not in (4, 8):
            raise RuntimeError(
                "StreamRender-H3 requires an initialized 4- or 8-rank group"
            )
        config = CfgNode(settings["inference"])
        world = dist.get_world_size()
        # Native FSDP placement expects torchrun's node-local topology. The
        # runner owns a single-node group; derive it from that existing group.
        os.environ["LOCAL_WORLD_SIZE"] = str(world)
        config.distributed.up_size = [1, world]
        torch.backends.cudnn.benchmark = config.distributed.get(
            "cudnn_benchmark", False
        )
        seed_everything(config.seed)
        init_unified_parallel((1, world), 1)
        models = build_models(config.models)
        if len(models["backbone"].dit.blocks) != 50:
            raise RuntimeError("The released renderer must execute all 50 DiT layers")
        # Every rank enters the same native topology and execution-phase scopes.
        self.pipe = StreamingPipeline(config, models, settings)
        self.world_size = world
        self.reference_cache = (
            Path(assets_path).resolve().parent / "reactor_reference_cache"
        )

    def start(self, reference, prompt: str, seed: int) -> None:
        import torch.distributed as dist
        from dev.yanzuolu.common.distributed.unified_parallel import (
            use_unified_parallel,
        )
        from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
        from PIL import Image

        # Upstream mixes the absolute picture path into its RNG seed. All ranks
        # must use one stable content-addressed path, including after a restart.
        digest = hashlib.sha256(
            str(reference.shape).encode() + reference.tobytes()
        ).hexdigest()
        picture = self.reference_cache / f"{digest}.png"
        status = [None]
        if dist.get_rank() == 0:
            try:
                self.reference_cache.mkdir(parents=True, exist_ok=True, mode=0o700)
                with NamedTemporaryFile(
                    dir=self.reference_cache, suffix=".png", delete=False
                ) as file:
                    temporary = Path(file.name)
                try:
                    Image.fromarray(reference).save(temporary)
                    os.replace(temporary, picture)
                finally:
                    temporary.unlink(missing_ok=True)
            except OSError as error:
                status[0] = str(error)
        dist.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise OSError(status[0])
        with (
            execution_phase(ExecutionPhase.VALIDATION),
            use_unified_parallel(self.world_size),
        ):
            self.pipe.start(prompt=prompt, picture=str(picture), seed=seed)

    def frames_wanted(self) -> int:
        return self.pipe.next_counts()[1]

    def render(self, semantics):
        import torch
        from dev.yanzuolu.common.distributed.unified_parallel import (
            use_unified_parallel,
        )
        from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase

        pixels = (
            torch.from_numpy(semantics)
            .to(self.pipe.device)
            .permute(0, 3, 1, 2)
            .float()
            .div_(255)
        )
        pixels = (
            torch.nn.functional.interpolate(pixels, size=(480, 832), mode="nearest")
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        with (
            execution_phase(ExecutionPhase.VALIDATION),
            use_unified_parallel(self.world_size),
        ):
            rgb, _ = self.pipe.render(pixels)
        if rgb is None:
            return None
        return rgb[0].permute(1, 2, 3, 0).clamp(0, 1).mul(255).byte().cpu().numpy()

    def reset(self) -> None:
        if self.pipe is not None:
            from render.optimizations import STATE

            self.pipe.states = None
            self.pipe.encoder = None
            self.pipe.decoder = None
            self.pipe.meta._stream_pictures = []
            self.pipe.meta._stream_qwen_contexts = []
            self.pipe.meta._validation_picture_cache.clear()
            STATE.reset()
