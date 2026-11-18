"""Merge the IC LoRA and LMS Sharp v1.0 into one LoRA file (CPU, FP32), as Open-H3-Regenerate-2K ships its adapters.

    python merge_lora.py --weights ../weights            # writes weights/lora/open_h3_regenerate_2k_ic_merged_lora.safetensors,
                                                         # sharp_curve_branches.safetensors, MERGE_VERIFY.json

Per targeted linear the factors are concatenated along the rank axis with the strengths folded into B:
A = [A_ic; A_sharp], B = [B_ic, s * B_sharp], so B @ A = B_ic A_ic + s * B_sharp A_sharp exactly (IC scale is 1:
alpha = rank = 384; Sharp deltas carry no alpha factor, s = --sharp, default 1.0). A module touched by only one adapter
keeps its own factors. Sharp's 50 AdaLN branches take the 8-D timestep curve (interpolated from adaln_t_table), not the
layer input, so they are written separately (sharp_curve_branches.safetensors, strength folded into curve_B).

The result is a plain LoRA (`transformer.<module>.lora_A.weight` / `.lora_B.weight`, scale 1) for the Ref2VA
transformer (`transformer_ref`), to be applied after the HyperFlow 8-step adapter. infer.py loads the IC LoRA and Sharp
separately (the default path); this file is for runtimes that take one LoRA per model.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ref2va_ic.sharp import convert

A, B = '.lora_A.weight', '.lora_B.weight'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=Path, required=True)
    ap.add_argument('--sharp', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    config = json.loads((a.weights / 'config.json').read_text())
    ic = load_file(str(a.weights / config['components']['lora']), device='cpu')
    linears, curves = convert(a.weights / config['components']['sharp_lora'])
    ic_mods = {k[len('transformer.'):-len(A)] for k in ic if k.endswith(A)}
    assert len(ic_mods) == 312 and len(curves) == 50
    merged, ranks, both = {}, {}, 0
    for m in sorted(ic_mods | set(linears)):
        parts = []
        if m in ic_mods:
            parts.append((ic[f'transformer.{m}{A}'].float(), ic[f'transformer.{m}{B}'].float()))
        if m in linears:
            sa, sb = linears[m]
            parts.append((sa.float(), a.sharp * sb.float()))
        both += len(parts) == 2
        merged[f'transformer.{m}{A}'] = torch.cat([p[0] for p in parts], 0).contiguous()
        merged[f'transformer.{m}{B}'] = torch.cat([p[1] for p in parts], 1).contiguous()
        ranks[m] = int(merged[f'transformer.{m}{A}'].shape[0])
    # verification: matvec relative error per module, and full B @ A on a few modules
    g = torch.Generator().manual_seed(a.seed)
    errs, full = [], []
    for i, m in enumerate(sorted(ranks)):
        am, bm = merged[f'transformer.{m}{A}'].double(), merged[f'transformer.{m}{B}'].double()
        x = torch.randn(am.shape[1], 4, generator=g, dtype=torch.float64)
        ref = 0
        if m in ic_mods:
            ref = ref + ic[f'transformer.{m}{B}'].double() @ (ic[f'transformer.{m}{A}'].double() @ x)
        if m in linears:
            ref = ref + a.sharp * (linears[m][1].double() @ (linears[m][0].double() @ x))
        got = bm.float().double() @ (am.float().double() @ x)
        errs.append(float((got - ref).norm() / ref.norm()))
        if i % 60 == 0:
            delta = (ic[f'transformer.{m}{B}'].double() @ ic[f'transformer.{m}{A}'].double() if m in ic_mods else 0) + \
                    (a.sharp * linears[m][1].double() @ linears[m][0].double() if m in linears else 0)
            full.append(dict(module=m, max_abs_err=float((bm @ am - delta).abs().max()), max_abs_delta=float(delta.abs().max())))
    out = a.weights / 'lora'
    save_file(merged, str(out / 'open_h3_regenerate_2k_ic_merged_lora.safetensors'),
              metadata=dict(format='lora', scale='1 (strengths folded into B)', dtype='float32',
                            target='MiniMaxAI/MiniMax-H3 transformer_ref, after the HyperFlow 8-step adapter',
                            sources=f'IC LoRA + LMS Sharp v1.0 x {a.sharp}'))
    save_file({f'transformer.{m}.curve_A.weight': ca.float().contiguous() for m, (ca, cb) in curves.items()} |
              {f'transformer.{m}.curve_B.weight': (a.sharp * cb.float()).contiguous() for m, (ca, cb) in curves.items()},
              str(out / 'sharp_curve_branches.safetensors'),
              metadata=dict(format='lms-sharp-curve-adaln', strength=f'{a.sharp} folded',
                            input='8-D timestep curve interpolated from adaln_t_table [1025, 8] at the model timestep'))
    report = dict(matvec_rel_err_max=max(errs), matvec_rel_err_mean=sum(errs) / len(errs), modules_checked=len(errs),
                  modules_in_both=both, rank_histogram={str(r): list(ranks.values()).count(r) for r in sorted(set(ranks.values()))},
                  full_matrix_checks=full, sharp_strength=a.sharp, curve_branches=len(curves),
                  seconds=round(time.time() - t0, 1))
    (out / 'MERGE_VERIFY.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ('matvec_rel_err_max', 'matvec_rel_err_mean', 'modules_checked',
                                              'modules_in_both', 'rank_histogram', 'seconds')}))


if __name__ == '__main__':
    main()
