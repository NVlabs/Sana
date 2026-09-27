"""Resident native NVFP4 Qwen in a dedicated ComfyUI Python process."""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import time

from .qwen_ops.media import input_spec, prepare_inputs, reference_target_area


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def resource_snapshot(torch):
    meminfo = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable', 'MemFree', 'Cached', 'SwapFree'):
            meminfo[key] = int(value.split()[0]) * 1024
    io = {}
    for line in Path('/proc/self/io').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('rchar', 'read_bytes'):
            io[key] = int(value)
    free, total = torch.cuda.mem_get_info()
    return {'monotonic_ns': time.monotonic_ns(), 'proc_meminfo_bytes': meminfo,
            'proc_self_io': io, 'cuda': {'allocated_bytes': torch.cuda.memory_allocated(),
                'reserved_bytes': torch.cuda.memory_reserved(), 'free_bytes': free,
                'total_bytes': total}}


def weight_snapshot(model, torch):
    """Inspect actual packed/scaled storages, not QuantizedTensor's logical size."""
    signatures, storages = {}, {}
    quantized_weights = 0
    other_quantized_weights = 0

    def add(name, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f'non-tensor quantized weight field: {name}')
        if tensor.device.type != 'cuda' or tensor.device.index not in (None, 0):
            raise RuntimeError(f'Qwen weight not CUDA resident: {name}: {tensor.device}')
        storage = tensor.untyped_storage()
        pointer, size = int(storage.data_ptr()), int(storage.nbytes())
        if size and not pointer:
            raise RuntimeError(f'Qwen weight has no storage: {name}')
        signatures[name] = (pointer, size, tensor.storage_offset(), tuple(tensor.shape),
                            tuple(tensor.stride()), str(tensor.dtype), str(tensor.device))
        storages[(pointer, size)] = size

    for kind, values in (('parameter', model.named_parameters()), ('buffer', model.named_buffers())):
        for name, tensor in values:
            key = f'{kind}.{name}'
            if hasattr(tensor, '_qdata'):
                add(key + '._qdata', tensor._qdata)
                fields = tuple(tensor._params._tensor_fields())
                if set(fields) == {'scale', 'block_scale'}:
                    quantized_weights += 1
                elif set(fields) == {'scale'}:
                    # Native checkpoint also carries a separately quantized
                    # token embedding; preserve it rather than reinterpret it.
                    other_quantized_weights += 1
                else:
                    raise RuntimeError(f'unrecognized quantized weight fields: {name}: {fields}')
                for field in fields:
                    add(key + '._params.' + field, getattr(tensor._params, field))
            else:
                add(key, tensor)
    if quantized_weights != 350:
        raise RuntimeError(f'expected 350 unchanged NVFP4 weights, found {quantized_weights}')
    fingerprint = hashlib.sha256(json.dumps(signatures, sort_keys=True).encode()).hexdigest()
    return signatures, {'all_weights_cuda_resident': True,
        'nvfp4_weight_count': quantized_weights, 'physical_weight_storage_bytes': sum(storages.values()),
        'other_quantized_weight_count': other_quantized_weights,
        'unique_weight_storages': len(storages), 'weight_tensor_bindings': len(signatures),
        'storage_identity_fingerprint': fingerprint}


def load_resident_checkpoint(checkpoint, torch):
    """Reuse verified O_DIRECT chunks to avoid whole-checkpoint CPU mmap peaks."""
    from safetensors import safe_open
    from .qwen_ops.direct_io import (
        DIRECT_CHUNK_BYTES, DIRECT_ALIGNMENT_BYTES, read_safetensors_records,
        iter_direct_chunks, preadv_fill_chunk,
    )
    checkpoint = Path(checkpoint).resolve(strict=True)
    # Native metadata only, not get_tensor / mmap payload materialization.
    with safe_open(str(checkpoint), framework='pt', device='cpu') as header:
        metadata = header.metadata()
    records = read_safetensors_records(checkpoint)
    dtype_names = {'U8': 'uint8', 'I8': 'int8', 'I32': 'int32', 'I64': 'int64',
                   'F32': 'float32', 'F16': 'float16', 'BF16': 'bfloat16',
                   'F8_E4M3': 'float8_e4m3fn', 'F8_E5M2': 'float8_e5m2'}
    backing = torch.empty(DIRECT_CHUNK_BYTES + DIRECT_ALIGNMENT_BYTES - 1,
                          dtype=torch.uint8, pin_memory=True)
    skew = (-backing.data_ptr()) % DIRECT_ALIGNMENT_BYTES
    slot = backing[skew:skew + DIRECT_CHUNK_BYTES]
    view = memoryview(slot.numpy())
    stats = {'mode': 'startup_only_ODIRECT_to_resident_CUDA', 'pinned_slots': 1,
             'pinned_bytes': backing.numel(), 'tensor_count': len(records),
             'logical_bytes': 0, 'read_calls': 0, 'physical_read_bytes': 0,
             'fallback_count': 0, 'direct_read_ns': 0}
    state = {}
    fd = os.open(checkpoint, os.O_RDONLY | os.O_DIRECT)
    try:
        file_size = checkpoint.stat().st_size
        for record in records:
            dtype = getattr(torch, dtype_names[record.dtype_code])
            raw = torch.empty(record.nbytes, dtype=torch.uint8, device='cuda:0')
            for chunk in iter_direct_chunks(record.file_offset, record.nbytes):
                started = time.monotonic_ns()
                received, calls, _, _, _ = preadv_fill_chunk(fd, view, chunk, file_size=file_size)
                stats['direct_read_ns'] += time.monotonic_ns() - started
                stats['read_calls'] += calls
                stats['physical_read_bytes'] += received
                # Blocking copy drains the one slot before its next disk read.
                raw[chunk.destination_offset:chunk.destination_offset + chunk.copy_bytes].copy_(
                    slot[chunk.source_offset:chunk.source_offset + chunk.copy_bytes], non_blocking=False)
            value = raw.view(dtype).view(record.shape)
            # Comfy parses these tiny JSON control records via .numpy(); they
            # are not model weights. Keep the original native CPU contract.
            state[record.name] = value.cpu() if record.name.endswith('.comfy_quant') else value
            stats['logical_bytes'] += record.nbytes
        torch.cuda.synchronize()
        return state, metadata, stats
    finally:
        os.close(fd)
        view.release()
        del view, slot, backing
        gc.collect()
        empty_host_cache = getattr(torch._C, '_host_emptyCache', None)
        if callable(empty_host_cache):
            empty_host_cache()


class Session:
    def __init__(self, paths: dict, work_dir: str, config: dict):
        self.reference_target_area = reference_target_area(config)
        checkpoint = Path(paths['qwen_checkpoint']).expanduser().resolve(strict=True)
        comfy_root = Path(paths['comfy_root']).expanduser().resolve(strict=True)
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.closed = False
        self.failed = False
        self.completed_requests = 0
        self.model_load_count = 0
        self.weight_reload_attempts = 0
        self.disk_weight_read_attempts = 0
        self._guards = []
        self.started_ns = time.monotonic_ns()
        import torch
        from .qwen_ops.conditioner import (
            MiniMaxNVFP4Conditioner, normalize_and_validate_conditioning, _import_comfy_sd,
        )
        if torch.cuda.device_count() != 1 or tuple(torch.cuda.get_device_capability()) != (12, 1):
            raise RuntimeError('Qwen resident runtime requires the single Spark SM121')
        self.torch = torch
        self.normalize = normalize_and_validate_conditioning
        self.before_load = resource_snapshot(torch)
        # Existing loader: load/offload/initial_device CUDA, disable_dynamic=True.
        # Only this exact checkpoint's startup reader changes, not Comfy's
        # quantization interpretation or model construction.
        _import_comfy_sd(Path(comfy_root))
        import comfy.utils as utils
        original_load = utils.load_torch_file
        self.startup_checkpoint_reads = 0
        def resident_file(ckpt, safe_load=False, device=None, return_metadata=False):
            if Path(ckpt).resolve() != Path(checkpoint).resolve() or self.startup_checkpoint_reads:
                raise RuntimeError('unexpected extra resident Qwen checkpoint load')
            state, metadata, self.startup_reader = load_resident_checkpoint(checkpoint, torch)
            self.startup_checkpoint_reads += 1
            return (state, metadata) if return_metadata else state
        utils.load_torch_file = resident_file
        try:
            self.owner = MiniMaxNVFP4Conditioner(Path(checkpoint), Path(comfy_root))
        finally:
            utils.load_torch_file = original_load
        self.clip = self.owner.clip
        import comfy.model_management as management
        management.load_models_gpu([self.clip.patcher], force_full_load=True)
        self.clip.cond_stage_model.eval().requires_grad_(False)
        torch.cuda.synchronize()
        self.model_load_count = 1
        if self.clip.patcher.is_dynamic():
            raise RuntimeError('resident runtime unexpectedly selected DynamicVRAM')
        self.owner.residency_summary()
        self._signature, self.weight_info = weight_snapshot(self.clip.cond_stage_model, torch)
        # All weights have been copied into CUDA allocations. Drop only this
        # checkpoint's reclaimable file cache, not user/system-wide page cache.
        gc.collect()
        self.checkpoint_page_cache_advice = 'unavailable'
        if hasattr(os, 'posix_fadvise'):
            with Path(checkpoint).open('rb') as handle:
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            self.checkpoint_page_cache_advice = 'POSIX_FADV_DONTNEED_single_checkpoint'
        torch.cuda.empty_cache()
        # A dedicated process: reject any later checkpoint reopen or file-slice
        # weight streaming. No global hooks affect the separate video workers.
        import comfy.utils as utils
        import comfy.memory_management as memory
        for module, name, counter in ((utils, 'load_torch_file', 'weight_reload_attempts'),
                                     (memory, 'read_tensor_file_slice_into', 'disk_weight_read_attempts')):
            if not hasattr(module, name):
                raise RuntimeError(f'existing Comfy weight I/O seam missing: {name}')
            original = getattr(module, name)
            def blocked(*args, _counter=counter, _name=name, **kwargs):
                setattr(self, _counter, getattr(self, _counter) + 1)
                raise RuntimeError(f'resident Qwen attempted {_name} after startup')
            setattr(module, name, blocked)
            self._guards.append((module, name, original, blocked))
        self.session_receipt = {'status': 'READY', 'producer_mode': 'resident',
            'model_load_count': self.model_load_count, 'device_contract': self.owner.device_contract,
            'loader_identity': self.owner.loader_identity, 'checkpoint': str(checkpoint),
            'started_monotonic_ns': self.started_ns, 'ready_monotonic_ns': time.monotonic_ns(),
            'before_load': self.before_load, 'after_load': resource_snapshot(torch),
            'checkpoint_page_cache_advice': self.checkpoint_page_cache_advice,
            'startup_checkpoint_reads': self.startup_checkpoint_reads,
            'startup_reader': self.startup_reader,
            'weight_residency': self.weight_info}

    def release_idle_cache(self):
        """Protocol name reused for metadata-only residency evidence; no offload."""
        if self.closed or self.failed:
            raise RuntimeError('Qwen resident session is unavailable')
        signature, info = weight_snapshot(self.clip.cond_stage_model, self.torch)
        if signature != self._signature:
            self.failed = True
            raise RuntimeError('resident Qwen weight storage changed after startup')
        return {'status': 'PASS', 'boundary': 'resident_Qwen_metadata_only_snapshot',
            'model_load_count': self.model_load_count, 'completed_requests': self.completed_requests,
            'weight_reload_attempts': self.weight_reload_attempts,
            'disk_weight_read_attempts': self.disk_weight_read_attempts,
            'weight_storage_unchanged': True, 'weight_residency': info,
            'resources': resource_snapshot(self.torch)}

    def run(self, case, output_root):
        if self.closed or self.failed:
            raise RuntimeError('Qwen resident session is unavailable')
        if (not isinstance(case, dict) or not isinstance(case.get('prompt'), str)
                or not case['prompt'].strip() or not case.get('case_id') or 'seed' not in case):
            raise ValueError('original case identity and prompt required')
        started_ns = time.monotonic_ns()
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=False)
        torch = self.torch
        try:
            before = self.release_idle_cache()
            torch.cuda.reset_peak_memory_stats()
            tokenize_kwargs, prepared_media = prepare_inputs(case, torch, target_area=self.reference_target_area)
            identity = {'task': case.get('task', 't2va'), 'external_anchor_used': False,
                'input_conditioned': case.get('task', 't2va') != 't2va',
                'input_spec': input_spec(case), 'prepared_media': prepared_media}
            with torch.inference_mode():
                # Native Comfy owns visual tokens, temporal patches and tags.
                tokens = self.clip.tokenize(case['prompt'], **tokenize_kwargs)
                encoded = self.clip.encode_from_tokens(tokens, return_dict=True)
                value = self.normalize(encoded)
            torch.cuda.synchronize()
            payload = {'prompt': case['prompt'], **identity,
                'prompt_embeds': value.cond, 'text_token_tags': value.minimax_token_tags}
            path = root / 'conditioning.pt'
            temporary = path.with_suffix('.tmp')
            with temporary.open('xb') as stream:
                torch.save(payload, stream)
            temporary.replace(path)
            self.completed_requests += 1
            after = self.release_idle_cache()
            receipt = {'status': 'PASS', **identity,
                'prompt': case['prompt'], 'case_id': case['case_id'], 'seed': case['seed'],
                'payload': str(path), 'payload_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'shape': list(value.cond.shape), 'dtype': str(value.cond.dtype),
                'producer_mode': 'resident', 'model_load_count': self.model_load_count,
                'request_index': self.completed_requests, 'residency_before': before,
                'residency_after': after, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                'started_monotonic_ns': started_ns, 'completed_monotonic_ns': time.monotonic_ns(),
                'caller_must_wait_for_exit_zero': False,
                'completion_protocol': 'resident_RPC_after_atomic_payload_commit'}
            receipt['conditioning_path'] = str((root / 'conditioning.json').resolve())
            receipt['request_wall_s'] = (receipt['completed_monotonic_ns'] - started_ns) / 1e9
            atomic_json(root / 'conditioning.json', receipt)
            return receipt
        except BaseException:
            self.failed = True
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        for module, name, original, guard in reversed(self._guards):
            if getattr(module, name) is guard:
                setattr(module, name, original)
        self._guards.clear()
        if self.clip is not None:
            # Remove only our model from Comfy's residency registry.
            import comfy.model_management as management
            for index in reversed(range(len(management.current_loaded_models))):
                loaded = management.current_loaded_models[index]
                if loaded.model is self.clip.patcher:
                    loaded.model_unload()
                    management.current_loaded_models.pop(index)
        self.clip = self.owner = None
        gc.collect()
        self.torch.cuda.empty_cache()
