"""Preflight the per-owner broadcast pattern used by _sharded_load_and_broadcast.

Simulates: N named tensors distributed across `world` owners (round-robin, like shard
files). Each rank fills ONLY its owned tensors with distinctive real values; the rest are
empty/garbage. Then each tensor is broadcast from its owner. Every rank must converge to
each tensor's owner value, with correct shape/dtype and no collective deadlock.

Run: torchrun --standalone --nproc_per_node 4 slurm/test_sharded_bcast.py
"""
import torch
import torch.distributed as dist


def main():
    dist.init_process_group("gloo")
    world = dist.get_world_size()
    rank = dist.get_rank()
    dev = torch.device("cpu")

    # 977-ish names across 13 "shards", round-robin owner = shard_idx % world.
    n_shards = 13
    names = [f"blocks.{i}.w" for i in range(50)] + [f"blocks.{i}.norm" for i in range(50)]
    shard_of = {nm: (idx % n_shards) for idx, nm in enumerate(names)}
    owner = {nm: (shard_of[nm] % world) for nm in names}
    # mixed dtype + shape per name
    dtype_of = {nm: (torch.float32 if "norm" in nm else torch.bfloat16) for nm in names}
    shape_of = {nm: ((256,) if "norm" in nm else (128, 64)) for nm in names}

    # owner truth value = owner_rank + 1 (distinctive); non-owners hold garbage -999
    store = {}
    for nm in names:
        if owner[nm] == rank:
            store[nm] = torch.full(shape_of[nm], float(owner[nm] + 1), dtype=dtype_of[nm])
        else:
            store[nm] = torch.full(shape_of[nm], -999.0, dtype=dtype_of[nm])

    with torch.no_grad():
        for nm in sorted(names):
            src = owner[nm]
            if rank == src:
                g = store[nm].to(device=dev, dtype=dtype_of[nm])
            else:
                g = torch.empty(shape_of[nm], dtype=dtype_of[nm], device=dev)
            dist.broadcast(g, src=src)
            store[nm] = g

    # verify: every tensor equals owner+1 everywhere
    bad = 0
    for nm in names:
        expect = float(owner[nm] + 1)
        if not torch.allclose(store[nm].float(), torch.full(shape_of[nm], expect)):
            bad += 1
    allbad = [torch.tensor(0) for _ in range(world)]
    dist.all_gather(allbad, torch.tensor(bad))
    if rank == 0:
        total = int(torch.stack(allbad).sum())
        print(f"SHARDED_BCAST_TEST {'PASS' if total == 0 else 'FAIL'} "
              f"names={len(names)} world={world} bad_across_ranks={total}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
