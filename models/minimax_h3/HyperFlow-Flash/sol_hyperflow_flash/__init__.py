"""Sol-Engine runner for HyperFlow H3 Flash (6-step, INT8 MiniMax-H3): t2v and ref2va at 1344x768, 24 fps.

    from sol_hyperflow_flash import config, runtime
    rt = runtime.HyperFlowFlashRuntime(weights, base_model, task="t2va")      # torchrun with world > 1: Ulysses SP
    rt.apply(config.get("ours"))
    rec, media = rt.generate(prompt, num_frames=124, seed=0, keep=True)
"""
__version__ = "0.1.0"
