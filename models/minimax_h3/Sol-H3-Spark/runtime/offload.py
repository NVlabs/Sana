"""Single-card CPU placement without changing the released sampling recipe."""
from contextlib import contextmanager, nullcontext


def enabled(config):
    return config.get("execution", {}).get("offload") == "cpu"


def validate_device(torch, config):
    capability = tuple(torch.cuda.get_device_capability(0))
    expected = (12, 0) if enabled(config) else (12, 1)
    if torch.cuda.device_count() != 1 or capability != expected:
        raise RuntimeError(f"Expected one visible SM{expected[0]}{expected[1]} GPU; "
                           f"got {torch.cuda.device_count()} devices and {capability}")
    return capability


def mutation_scope(module):
    state = getattr(module, "_sol_h3_offload_state", None)
    return state.mutate() if state is not None else nullcontext()


def install_stage1_loader():
    """Use native CPU loading and LoRA scopes, including FP8 nonpersistent buffers.

    The pinned loader otherwise constructs an entire BF16 DiT on CUDA before
    installing offload. Its original offloader also omits quantized buffers.
    These substitutions are limited to this dedicated FastVideo worker.
    """
    import torch
    from fastvideo.models.loader import component_loader
    from fastvideo.hooks.hooks import ForwardHook, ModuleHookManager

    original_load = component_loader.maybe_load_fsdp_model

    def cpu_load(*args, **kwargs):
        if kwargs.get("fsdp_inference") or kwargs.get("training_mode"):
            raise RuntimeError("The single-card H3 offloader requires inference without FSDP")
        # Native option normalization disables dit_cpu_offload when layerwise
        # offload is enabled. Override only this worker's construction call;
        # the native stage must not later move the whole DiT onto CUDA.
        kwargs["cpu_offload"] = True
        kwargs["device"] = torch.device("cpu")
        return original_load(*args, **kwargs)

    component_loader.maybe_load_fsdp_model = cpu_load

    class State:
        def __init__(self, module, stats):
            self.module, self.stats = module, stats
            self.saved = None
            module.to("cpu")

        @torch.compiler.disable
        def load(self):
            if self.saved is not None:
                raise RuntimeError("Overlapping execution of one offloaded block")
            self.saved = (dict(self.module.named_parameters()), dict(self.module.named_buffers()))
            # Save data tensors, not Parameter objects whose .data is replaced.
            self.saved = ({name: param.detach() for name, param in self.saved[0].items()}, self.saved[1])
            if any(t.device.type != "cpu" for group in self.saved for t in group.values()):
                raise RuntimeError("An idle offloaded block retained GPU weights")
            self.module.to("cuda:0")
            self.stats["block_loads"] += 1
            self.stats["active_blocks"] += 1
            self.stats["max_active_blocks"] = max(self.stats["max_active_blocks"], self.stats["active_blocks"])

        @torch.compiler.disable
        def release(self, *, changed=False):
            if self.saved is None:
                return
            torch.cuda.synchronize()
            if changed:
                # LoRA merge, exact lookup replacement and FP8 conversion may
                # change tensor names, shapes and dtypes. Capture the new state.
                self.module.to("cpu")
            else:
                for name, param in self.module.named_parameters():
                    param.data = self.saved[0][name]
                for name in list(dict(self.module.named_buffers())):
                    parent, _, leaf = name.rpartition(".")
                    owner = self.module.get_submodule(parent) if parent else self.module
                    owner._buffers[leaf] = self.saved[1][name]
            self.saved = None
            self.stats["active_blocks"] -= 1

        @contextmanager
        def mutate(self):
            self.load()
            try:
                yield
            finally:
                self.release(changed=True)

    class Hook(ForwardHook):
        def __init__(self, state):
            self.state = state

        @classmethod
        def name(cls):
            # This is the name queried by native LoRAPipeline._get_hook_ctx.
            return "LayerwiseOffloadHook"

        def mutate_params_scope(self):
            return self.state.mutate()

    class Manager:
        enabled = True

        def __init__(self, model):
            blocks = model.transformer_blocks
            if len(blocks) != 50:
                raise RuntimeError("Expected the released fifty H3 body blocks")
            self.stats = {"mode": "cpu", "blocks": 50, "parameters_and_buffers": True,
                          "block_loads": 0, "active_blocks": 0, "max_active_blocks": 0}
            self.states, self.handles = [], []
            for block in blocks:
                state = State(block, self.stats)
                block._sol_h3_offload_state = state
                self.states.append(state)
                ModuleHookManager.get_from_or_default(block).append_forward_hook(Hook(state))
            # Small shared modules and the two text blocks retain their native
            # placement; only the fifty large repeated body blocks are streamed.
            for name, child in model.named_children():
                if name != "transformer_blocks":
                    child.to("cuda:0")

        def prepare_compilation(self):
            for state in self.states:
                manager = ModuleHookManager.get_from(state.module)
                if manager is None or set(manager.forward_hooks) != {"LayerwiseOffloadHook"}:
                    raise RuntimeError("Unexpected H3 construction hooks")
                ModuleHookManager.remove_from_manager(state.module)

        def install_inference_hooks(self):
            # PyTorch module hooks remain outside each compiled .forward.
            for state in self.states:
                self.handles.append(state.module.register_forward_pre_hook(
                    lambda module, args, state=state: state.load()))
                self.handles.append(state.module.register_forward_hook(
                    lambda module, args, output, state=state: state.release(), always_call=True))
            self.stats["fp8_buffer_count"] = sum(
                name.endswith("_fp8_weight") for state in self.states for name, _ in state.module.named_buffers())

        def release_all(self):
            for state in self.states:
                state.release()

    def install(model, is_replace=False):
        if is_replace or hasattr(model, "_layerwise_offload_manager"):
            raise RuntimeError("Install the dedicated H3 offloader only once")
        model._layerwise_offload_manager = Manager(model)

    component_loader.enable_layerwise_offload = install


def prepare_stage1_compile(model):
    manager = getattr(model, "_layerwise_offload_manager", None)
    if manager is not None:
        manager.prepare_compilation()
    return manager
