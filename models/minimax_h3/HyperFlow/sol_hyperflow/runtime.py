"""Resident HyperFlow workflows with shared conditioners and the validated Sol optimizations."""
import functools
import inspect
import json
import logging
from pathlib import Path
import textwrap
import time

import torch
import torch.distributed as dist
from .bootstrap import ensure_sol_h3

ensure_sol_h3()

from diffusers import ContextParallelConfig
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline
from hyperflow_h3 import hyperflow_blocks, load_hyperflow_lora
from hyperflow_h3.blocks import configure_hyperflow_blocks
from h3_runtime import fusion_install, sparse_attention, ulysses, vae_parallel
from h3_runtime.lora_fusion import install as install_lora_fusion
from h3_runtime.cp_plan import MINIMAX_H3_CP_PLAN, assert_no_attention_mask
from .adaln_cache import make_plans, precompute, install_hook


def install_reference_resize():
    from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
    if getattr(MiniMaxH3Ref2VASetupStep.__call__, '_sol_hyperflow_reference_resize', False):
        return
    original = inspect.unwrap(MiniMaxH3Ref2VASetupStep.__call__)
    source = textwrap.dedent(inspect.getsource(original))
    old = 'scale = components.config.reference_image_short_edge / min(width, height)'
    new = 'scale = min(1.0, math.sqrt((1344 * 768) / (width * height)))'
    assert source.count(old) == 1, 'Pinned reference geometry implementation changed'
    namespace = original.__globals__.copy()
    exec(compile(source.replace(old, new), __file__ + ':reference_resize', 'exec'), namespace)
    patched = functools.wraps(original)(namespace['__call__'])
    patched._sol_hyperflow_reference_resize = True
    MiniMaxH3Ref2VASetupStep.__call__ = patched


def optimize(pipe, model, metadata, world, *, reference):
    model.set_attention_backend('native')
    model.requires_grad_(False).eval()
    assert all(not m.merged for m in model.modules() if hasattr(m,'lora_A'))
    original_forward = model.forward
    assert_no_attention_mask(model)
    functools.update_wrapper(model.forward, original_forward)
    model.enable_parallelism(
        config=ContextParallelConfig(ulysses_degree=world, ulysses_anything=True),
        cp_plan=MINIMAX_H3_CP_PLAN)
    model._consumer_lora_fusion = install_lora_fusion(model)
    model._consumer_lora_fusion.audit = True
    fusion_install.install(model, lora=True)
    cache = precompute(model, make_plans(pipe, metadata.sigmas))
    policy = dict(backend='sol_bsa', tau=1.0,
                  dense_steps=0 if reference else 1,
                  dense_layers=0 if reference else 2,
                  sink_mode='text_audio' if reference else 'prefix')
    sparse = sparse_attention.install(model, **policy)
    ulysses.install(model, attention_fn=sparse, lora=True)
    model.eval()
    return sparse, dict(quant=dict(mode='none'), adaln=cache, sparse=policy,
        lora=dict(mode='fused',targets=len(model._consumer_lora_fusion.targets),adapter='hyperflow'))


def build_all(model_path, weights, device, world, *, lora_mode='fused', trim_conditioner=True):
    from .config import validate_world_size
    validate_world_size(world)
    if lora_mode not in {'fused', 'separate'}:
        raise ValueError('HyperFlow supports fused or separate BF16 LoRA branches.')
    started = time.perf_counter()
    logging.basicConfig(level=logging.INFO if dist.get_rank() == 0 else logging.WARNING)
    # The released 15-second preset aligns to 362 frames, i.e. slightly over 15 seconds.
    MiniMaxH3ModularPipeline.max_duration = property(lambda self: 364 / 24)
    install_reference_resize()
    load = dict(dtype=torch.bfloat16, local_files_only=True,
                pretrained_model_name_or_path=model_path)
    main = hyperflow_blocks('t2va').init_pipeline(model_path)
    main.load_components(**load)
    from .conditioner_memory import trim_and_verify
    conditioner_report = trim_and_verify(main, device) if trim_conditioner else dict(enabled=False)
    if dist.get_rank() == 0:
        print('CONDITIONER_MEMORY_VERIFIED', json.dumps(conditioner_report), flush=True)
    meta = load_hyperflow_lora(main, weights)
    assert len(meta.sigmas) - 1 == 8
    main.to(device)
    main.audio_vae.set_attention_backend('native')
    sparse, main_report = optimize(main, main.transformer, meta, world, reference=False)
    main.set_progress_bar_config(disable=True)

    image = hyperflow_blocks('fl2va').init_pipeline(model_path)
    image.update_components(**{k:v for k,v in main.components.items() if k in image._component_specs})
    configure_hyperflow_blocks(image.blocks, meta)
    image.set_progress_bar_config(disable=True)

    ref = hyperflow_blocks('ref2va').init_pipeline(model_path)
    ref.update_components(**{k:v for k,v in main.components.items() if k in ref._component_specs and k != 'transformer'})
    ref.load_components(names=['transformer_ref'], **load)
    ref_meta = load_hyperflow_lora(ref, weights)
    # The uncompressed AdaLN weights exist only during initialization. Temporarily
    # stage the idle main DiT on CPU while the reference tables are built.
    main.transformer.to('cpu')
    torch.cuda.empty_cache()
    ref.to(device)
    ref_sparse, ref_report = optimize(ref, ref.transformer_ref, ref_meta, world, reference=True)
    main.transformer.to(device)
    ref.set_progress_bar_config(disable=True)
    install_hook()
    vae_parallel.install(main.vae, batched=True, compile_mode='default', encode_parallel=True)
    for component in ['text_encoder','vae','audio_vae','tokenizer','processor']:
        assert getattr(main,component) is getattr(image,component) is getattr(ref,component)
    assert image.transformer is main.transformer
    assert ref.transformer_ref is not main.transformer
    if tuple(meta.sigmas) != tuple(ref_meta.sigmas) or meta.version != ref_meta.version:
        raise RuntimeError('The two workflow partitions loaded inconsistent HyperFlow metadata.')
    for model, details in [(main.transformer, main_report), (ref.transformer_ref, ref_report)]:
        model._consumer_lora_fusion.enabled = lora_mode == 'fused'
        details['lora']['mode'] = lora_mode
    report = dict(
        model='MiniMax-H3-HyperFlow', adapter_version=meta.version,
        adapter_file=Path(weights).name, steps=8, lora_mode=lora_mode, compute_quant='none',
        lora_fusion_source='Sana PR #503', conditioner_memory=conditioner_report,
        transformer_instances=2, shared_components=['text_encoder','vae','audio_vae','tokenizer','processor'],
        t2v=main_report, ref2v=ref_report, reference_kv_cache=False,
        reference_resize='match', load_s=time.perf_counter()-started)
    return {'t2v':main,'i2v':image,'ref2v':ref}, {'t2v':sparse,'ref2v':ref_sparse}, report


def begin_request(sparse, *, task, mode):
    # Finalize using the preceding request's policy before selecting this one.
    sparse._close_request()
    sparse.request = -1
    sparse._layout_key = None
    sparse._prev_timestep = None
    sparse._direction = 0
    sparse.step = -1
    sparse.layer = 0
    sparse._sparse_at_request_start = sparse.sparse_calls
    if mode == 'quality':
        sparse.dense_steps, sparse.dense_layers = 8, 0
    elif task == 'ref2v':
        sparse.dense_steps, sparse.dense_layers = 0, 0
    else:
        sparse.dense_steps, sparse.dense_layers = 1, 2
    ulysses.reset_row_counts()


def run(pipe, request, sparse, *, expect_fused=True):
    begin_request(sparse, task=request['task'], mode=request['performance_mode'])
    before = sparse.sparse_calls, sparse.dense_calls
    model = pipe.transformer_ref if request['task']=='ref2v' else pipe.transformer
    fusion = model._consumer_lora_fusion
    assert fusion.enabled == expect_fused
    calls_before = dict(fusion.calls)
    kw = dict(prompt=request['prompt'], height=request['height'], width=request['width'],
              num_frames=request['num_frames'], num_inference_steps=8,
              generator=torch.Generator().manual_seed(request['seed']),
              output=['videos','audio','sampling_rate'], output_type='pt')
    if request['task'] == 'i2v':
        kw['image'] = request['image']
    elif request['task'] == 'ref2v':
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference
        kw['references'] = [MiniMaxH3ImageReference(image=value) for value in request['reference_images']]
    dist.barrier()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        result = pipe(**kw)
    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - started
    counts = dict(sparse_calls=sparse.sparse_calls-before[0], dense_calls=sparse.dense_calls-before[1])
    assert counts['sparse_calls'] + counts['dense_calls'] == 400, counts
    expected_sparse = 0 if request['performance_mode'] == 'quality' else (400 if request['task'] == 'ref2v' else 336)
    assert counts['sparse_calls'] == expected_sparse, counts
    assert result['videos'][0].shape == (request['num_frames'],3,request['height'],request['width'])
    calls = {k:v-calls_before.get(k,0) for k,v in fusion.calls.items()}
    if expect_fused:
        assert calls == dict(split_linear=2400,qkv_pack=400,attention_output=400,swiglu=400,ffn_output=400), calls
    else:
        assert all(v==0 for v in calls.values()), calls
    model._last_consumer_fusion_calls = calls
    return result, elapsed, counts
