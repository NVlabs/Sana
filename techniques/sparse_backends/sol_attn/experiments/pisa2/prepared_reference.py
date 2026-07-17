import math

import torch


def pisa2_prepared_route_logits_and_exact_mask(
    q_i8,
    q_scale,
    kc_i8,
    kc_scale,
    global_thresh,
    softmax_scale,
    q_block,
    block_size=64,
):
    """Return canonical per-row route logits and selector bits for one tile."""

    B, H, T, _ = q_i8.shape
    NT = (T + block_size - 1) // block_size
    if isinstance(q_block, bool) or not isinstance(q_block, int):
        raise ValueError("q_block must be an integer")
    if not 0 <= q_block < NT:
        raise ValueError(f"q_block {q_block} is outside [0, {NT})")
    q_start = q_block * block_size
    q_stop = min(q_start + block_size, T)
    q_len = q_stop - q_start
    q_tile = q_i8[:, :, q_start:q_stop].float()
    route_logits = torch.matmul(
        q_tile, kc_i8.float().transpose(-1, -2)
    )
    sm_scale = (
        q_scale[:, :, q_block, 0].float()
        * (softmax_scale * math.log2(math.e))
    )
    route_logits *= (
        sm_scale[:, :, None, None]
        * kc_scale[:, :, None, :, 0].float()
    )
    col_mean = route_logits.sum(dim=-2) / q_len
    block_ids = torch.arange(NT, device=q_i8.device)
    exact = (col_mean > global_thresh[:, :, q_block, None].float()) | (
        (block_ids - q_block).abs() <= 1
    )
    return route_logits, exact


def pisa2_prepared_reference(
    q_i8,
    k_i8,
    v,
    q_scale,
    k_scale,
    kc_i8,
    kc_scale,
    vc,
    global_thresh,
    softmax_scale,
    block_size=64,
    group_size=64,
    force_all_approx=False,
):
    B, H, T, _ = q_i8.shape
    DV = v.shape[-1]
    NT = (T + block_size - 1) // block_size
    scale_log2 = softmax_scale * math.log2(math.e)
    out = torch.empty((B, H, T, DV), device=q_i8.device, dtype=torch.float32)
    lse = torch.empty((B, H, T), device=q_i8.device, dtype=torch.float32)

    qf = q_i8.float()
    kf = k_i8.float()
    kcf = kc_i8.float()
    vf = v.float()
    vcf = vc.float()

    for b in range(B):
        for h in range(H):
            for m in range(NT):
                q_start = m * block_size
                q_stop = min(q_start + block_size, T)
                q_len = q_stop - q_start
                q_tile = qf[b, h, q_start:q_stop]
                qs = q_scale[b, h, m, 0].float()
                row_logits = []
                row_values = []
                row_lens = []

                for group_start in range(0, NT, group_size):
                    group_stop = min(group_start + group_size, NT)
                    kc_tile = kcf[b, h, group_start:group_stop]
                    route_logits = (q_tile @ kc_tile.T) * (
                        qs
                        * kc_scale[b, h, group_start:group_stop, 0].float()
                        * scale_log2
                    )
                    col_mean = route_logits.sum(dim=0) / q_len
                    block_ids = torch.arange(group_start, group_stop, device=q_i8.device)
                    exact = (col_mean > global_thresh[b, h, m].float()) | (
                        (block_ids - m).abs() <= 1
                    )
                    if force_all_approx:
                        exact = torch.zeros_like(exact)
                    approx = ~exact

                    if approx.any():
                        approx_idx = torch.nonzero(approx, as_tuple=False).flatten()
                        global_idx = approx_idx + group_start
                        row_logits.append(route_logits[:, approx_idx])
                        row_values.append(vcf[b, h, global_idx])
                        lens = torch.full(
                            (approx_idx.numel(),),
                            block_size,
                            device=q_i8.device,
                            dtype=torch.float32,
                        )
                        last_block = NT - 1
                        tail = T - last_block * block_size
                        lens = torch.where(
                            global_idx == last_block, torch.full_like(lens, tail), lens
                        )
                        row_lens.append(lens)

                    for n in torch.nonzero(exact, as_tuple=False).flatten().tolist():
                        n_idx = group_start + n
                        kv_start = n_idx * block_size
                        kv_stop = min(kv_start + block_size, T)
                        k_tile = kf[b, h, kv_start:kv_stop]
                        exact_logits = (q_tile @ k_tile.T) * (
                            qs * k_scale[b, h, n_idx, 0].float() * scale_log2
                        )
                        row_logits.append(exact_logits)
                        row_values.append(vf[b, h, kv_start:kv_stop])
                        row_lens.append(
                            torch.ones(
                                (kv_stop - kv_start,),
                                device=q_i8.device,
                                dtype=torch.float32,
                            )
                        )

                logits = torch.cat(row_logits, dim=1)
                values = torch.cat(row_values, dim=0)
                lens = torch.cat(row_lens, dim=0)
                mval = logits.max(dim=1, keepdim=True).values
                prob = torch.exp2(logits - mval)
                denom = (prob * lens[None, :]).sum(dim=1, keepdim=True)
                numer = prob @ values
                out[b, h, q_start:q_stop] = numer / denom
                lse[b, h, q_start:q_stop] = (
                    (mval[:, 0] + torch.log2(denom[:, 0])) * math.log(2.0)
                )

    return out, lse


def pisa2_prepared_reference_query_block(
    q_i8,
    k_i8,
    v,
    q_scale,
    k_scale,
    kc_i8,
    kc_scale,
    vc,
    global_thresh,
    softmax_scale,
    q_block,
    block_size=64,
    *,
    last_block_len_override=None,
):
    """Vectorized prepared reference for one query block, including tails."""

    B, H, T, _ = q_i8.shape
    DV = v.shape[-1]
    NT = (T + block_size - 1) // block_size
    if isinstance(q_block, bool) or not isinstance(q_block, int):
        raise ValueError("q_block must be an integer")
    if not 0 <= q_block < NT:
        raise ValueError(f"q_block {q_block} is outside [0, {NT})")
    q_start = q_block * block_size
    q_stop = min(q_start + block_size, T)
    q_len = q_stop - q_start
    scale_log2 = softmax_scale * math.log2(math.e)
    out = torch.empty((B, H, q_len, DV), device=q_i8.device, dtype=torch.float32)
    lse = torch.empty((B, H, q_len), device=q_i8.device, dtype=torch.float32)
    block_lens = torch.full(
        (NT,), block_size, device=q_i8.device, dtype=torch.float32
    )
    tail_len = T - (NT - 1) * block_size
    if last_block_len_override is not None:
        if (
            isinstance(last_block_len_override, bool)
            or not isinstance(last_block_len_override, int)
            or not 1 <= last_block_len_override <= block_size
        ):
            raise ValueError(
                "last_block_len_override must be an integer in [1, block_size]"
            )
        tail_len = last_block_len_override
    block_lens[-1] = tail_len
    route_logits_all, exact_all = (
        pisa2_prepared_route_logits_and_exact_mask(
            q_i8,
            q_scale,
            kc_i8,
            kc_scale,
            global_thresh,
            softmax_scale,
            q_block,
            block_size,
        )
    )

    for b in range(B):
        for h in range(H):
            q_tile = q_i8[b, h, q_start:q_stop].float()
            qs = q_scale[b, h, q_block, 0].float()
            route_logits = route_logits_all[b, h]
            exact = exact_all[b, h]
            approx_ids = torch.nonzero(~exact, as_tuple=False).flatten()
            all_token_blocks = (
                torch.arange(T, device=q_i8.device) // block_size
            )
            exact_token_mask = exact[all_token_blocks]
            exact_tokens = torch.nonzero(
                exact_token_mask, as_tuple=False
            ).flatten()

            logits = []
            values = []
            lens = []
            if approx_ids.numel():
                logits.append(route_logits[:, approx_ids])
                values.append(vc[b, h, approx_ids].float())
                lens.append(block_lens[approx_ids])
            if exact_tokens.numel():
                token_blocks = exact_tokens // block_size
                exact_logits = (
                    q_tile @ k_i8[b, h, exact_tokens].float().T
                ) * (
                    qs
                    * k_scale[b, h, token_blocks, 0].float()
                    * scale_log2
                )
                logits.append(exact_logits)
                values.append(v[b, h, exact_tokens].float())
                lens.append(
                    torch.ones(
                        exact_tokens.numel(),
                        device=q_i8.device,
                        dtype=torch.float32,
                    )
                )
            logits_cat = torch.cat(logits, dim=1)
            values_cat = torch.cat(values, dim=0)
            lens_cat = torch.cat(lens, dim=0)
            row_max = logits_cat.max(dim=1, keepdim=True).values
            prob = torch.exp2(logits_cat - row_max)
            denom = (prob * lens_cat[None, :]).sum(dim=1, keepdim=True)
            out[b, h] = (prob @ values_cat) / denom
            lse[b, h] = (
                (row_max[:, 0] + torch.log2(denom[:, 0])) * math.log(2.0)
            )
    return out, lse


def pisa2_prepared_lse_reference(
    q_i8,
    k_i8,
    q_scale,
    k_scale,
    kc_i8,
    kc_scale,
    global_thresh,
    softmax_scale,
    block_size=64,
):
    """Compute independent LSE for every query without the unnecessary PV.

    This follows the same prepared PISA2 semantics as
    :func:`pisa2_prepared_reference`: approximate centroids contribute their
    physical token count to the denominator, while selected exact blocks
    contribute one term per valid token.  Avoiding the value matmul makes a
    full T=66000 LSE gate practical while retaining every query position.
    """

    B, H, T, _ = q_i8.shape
    NT = (T + block_size - 1) // block_size
    scale_log2 = softmax_scale * math.log2(math.e)
    lse = torch.empty((B, H, T), device=q_i8.device, dtype=torch.float32)
    token_blocks = torch.arange(T, device=q_i8.device) // block_size
    block_lens = torch.full(
        (NT,), block_size, device=q_i8.device, dtype=torch.float32
    )
    block_lens[-1] = T - (NT - 1) * block_size

    for b in range(B):
        for h in range(H):
            k = k_i8[b, h].float()
            for q_block in range(NT):
                q_start = q_block * block_size
                q_stop = min(q_start + block_size, T)
                q_tile = q_i8[b, h, q_start:q_stop].float()
                qs = q_scale[b, h, q_block, 0].float()
                route_logits_all, exact_all = (
                    pisa2_prepared_route_logits_and_exact_mask(
                        q_i8,
                        q_scale,
                        kc_i8,
                        kc_scale,
                        global_thresh,
                        softmax_scale,
                        q_block,
                        block_size,
                    )
                )
                route_logits = route_logits_all[b, h]
                exact = exact_all[b, h]
                approx_ids = torch.nonzero(~exact, as_tuple=False).flatten()
                exact_tokens = torch.nonzero(
                    exact[token_blocks], as_tuple=False
                ).flatten()

                logits = []
                lens = []
                if approx_ids.numel():
                    logits.append(route_logits[:, approx_ids])
                    lens.append(block_lens[approx_ids])
                if exact_tokens.numel():
                    exact_token_blocks = token_blocks[exact_tokens]
                    logits.append(
                        (q_tile @ k[exact_tokens].T)
                        * (
                            qs
                            * k_scale[
                                b, h, exact_token_blocks, 0
                            ].float()
                            * scale_log2
                        )
                    )
                    lens.append(
                        torch.ones(
                            exact_tokens.numel(),
                            device=q_i8.device,
                            dtype=torch.float32,
                        )
                    )

                logits_cat = torch.cat(logits, dim=1)
                lens_cat = torch.cat(lens, dim=0)
                row_max = logits_cat.max(dim=1, keepdim=True).values
                denom = (
                    torch.exp2(logits_cat - row_max) * lens_cat[None, :]
                ).sum(dim=1)
                lse[b, h, q_start:q_stop] = (
                    row_max[:, 0] + torch.log2(denom)
                ) * math.log(2.0)

    return lse


def _self_check_case(T=256, seed=0):
    from cutlass.cute.runtime import from_dlpack
    import cuda.bindings.driver as cuda

    from kernels.pisa2_sm90 import PISA2_GROUP_SIZE, build_pisa2_sm90_fwd

    torch.manual_seed(seed)
    B, H, D = 1, 1, 128
    block_size = 64
    group_size = PISA2_GROUP_SIZE
    NT = (T + block_size - 1) // block_size
    q = torch.randint(-8, 8, (B, H, T, D), device="cuda", dtype=torch.int8)
    k = torch.randint(-8, 8, (B, H, T, D), device="cuda", dtype=torch.int8)
    kc = torch.randint(-8, 8, (B, H, NT, D), device="cuda", dtype=torch.int8)
    v = torch.randn((B, H, T, D), device="cuda", dtype=torch.bfloat16)
    vc = torch.randn((B, H, NT, D), device="cuda", dtype=torch.bfloat16)
    o = torch.empty_like(v)
    q_scale = torch.rand((B, H, NT, 1), device="cuda", dtype=torch.float32) * 0.02 + 0.01
    k_scale = torch.rand((B, H, NT, 1), device="cuda", dtype=torch.float32) * 0.02 + 0.01
    kc_scale = torch.rand((B, H, NT, 1), device="cuda", dtype=torch.float32) * 0.02 + 0.01
    thresh = torch.full((B, H, NT), 1.0e9, device="cuda", dtype=torch.float32)
    lse = torch.empty((B, H, T), device="cuda", dtype=torch.float32)

    op = build_pisa2_sm90_fwd(T)
    op(
        from_dlpack(q).mark_layout_dynamic(),
        from_dlpack(k).mark_layout_dynamic(),
        from_dlpack(v).mark_layout_dynamic(),
        from_dlpack(o).mark_layout_dynamic(),
        from_dlpack(q_scale).mark_layout_dynamic(),
        from_dlpack(k_scale).mark_layout_dynamic(),
        from_dlpack(kc).mark_layout_dynamic(),
        from_dlpack(kc_scale).mark_layout_dynamic(),
        from_dlpack(vc).mark_layout_dynamic(),
        from_dlpack(thresh).mark_layout_dynamic(),
        from_dlpack(lse).mark_layout_dynamic(),
        128**-0.5,
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    torch.cuda.synchronize()

    ref_o, ref_lse = pisa2_prepared_reference(
        q,
        k,
        v,
        q_scale,
        k_scale,
        kc,
        kc_scale,
        vc,
        thresh,
        128**-0.5,
        block_size,
        group_size,
    )
    diff = (o.float() - ref_o).abs()
    lse_diff = (lse - ref_lse).abs()
    print(
        "CASE",
        {"T": T, "seed": seed, "NT": NT},
        "out_max_abs",
        float(diff.max().cpu()),
        "out_mean_abs",
        float(diff.mean().cpu()),
        "lse_max_abs",
        float(lse_diff.max().cpu()),
        "lse_mean_abs",
        float(lse_diff.mean().cpu()),
    )
    if not torch.all(torch.isfinite(o)) or not torch.all(torch.isfinite(lse)):
        raise AssertionError("CuteDSL output contains non-finite values")
    if float(diff.max().cpu()) > 0.25 or float(diff.mean().cpu()) > 0.03:
        raise AssertionError("output mismatch above diagnostic tolerance")


def main():
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    _self_check_case(T=128, seed=0)
    _self_check_case(T=256, seed=0)
    print("PISA2_CUTEDSL_PREPARED_REFERENCE_CHECK_DONE")


if __name__ == "__main__":
    main()
