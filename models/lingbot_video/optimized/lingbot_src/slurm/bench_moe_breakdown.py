"""MoE block internal breakdown + megafusion (torch.compile) measurement.
Uses the REAL LingBotVideoSparseMoeBlock. Times router / reorder / grouped_mm /
restore / shared, then compares full-eager vs full-compiled (inductor fuses the
pointwise glue -> the 'GEMM + epilogue' megafusion of approach 1).
"""
import os, sys, time
os.environ.setdefault("LINGBOT_MOE_EXPERT_BACKEND", "grouped_mm")
os.environ.setdefault("LINGBOT_MOE_PAD_BACKEND", "vectorized")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from lingbot_video.transformer_lingbot_video import LingBotVideoSparseMoeBlock

dev = torch.device("cuda", 0); dt = torch.bfloat16
torch.manual_seed(0)

H, INTER, E, K, MOE_I = 2048, 6144, 128, 8, 768
block = LingBotVideoSparseMoeBlock(
    hidden_size=H, intermediate_size=INTER, num_experts=E, top_k=K,
    moe_intermediate_size=MOE_I, score_func="sigmoid", norm_topk_prob=True,
    n_group=4, topk_group=2, routed_scaling_factor=2.5, n_shared_experts=1,
).to(dev).to(dt)
with torch.no_grad():
    for p in block.parameters():
        p.data.normal_(0, 0.02)

TOK = 126480  # 1080p per-rank (CP4)
x = torch.randn(1, TOK, H, device=dev, dtype=dt) * 0.1


def t(fn, it=30, wu=8):
    for _ in range(wu): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / it


@torch.no_grad()
def parts():
    tokens = x.view(-1, H)
    ti, ts, *_ = block.router(tokens)
    perm, counts, spos, sscore, ntok, tk = block._reorder_tokens(tokens, ts, ti, block.router.num_experts)
    eo = block._run_grouped_experts(perm, counts)
    return tokens, ti, ts, perm, counts, spos, sscore, ntok, tk, eo


with torch.no_grad():
    tokens, ti, ts, perm, counts, spos, sscore, ntok, tk, eo = parts()

print(f"GPU: {torch.cuda.get_device_name(0)}  MoE block, {TOK} tokens (1080p per-rank)")
print(f"{'part':>22} | {'ms':>7} | {'% of full':>9}")
print("-" * 46)
tmr = {}
tmr["router"]  = t(lambda: block.router(tokens))
tmr["reorder"] = t(lambda: block._reorder_tokens(tokens, ts, ti, E))
tmr["grouped_mm(matmul)"] = t(lambda: block._run_grouped_experts(perm, counts))
tmr["restore(scatter)"]   = t(lambda: block._restore_tokens(eo, spos, sscore, ntok, tk))
tmr["shared_expert"]      = t(lambda: block.shared_experts(x))
full_eager = t(lambda: block(x))
for k, v in tmr.items():
    print(f"{k:>22} | {v:>7.3f} | {100*v/full_eager:>8.1f}%")
print(f"{'FULL (eager)':>22} | {full_eager:>7.3f} | {100.0:>8.1f}%")
glue = tmr["router"] + tmr["reorder"] + tmr["restore(scatter)"]
print(f"\n  matmul (grouped+shared) = {tmr['grouped_mm(matmul)']+tmr['shared_expert']:.3f} ms")
print(f"  FUSABLE glue (router+reorder+restore) = {glue:.3f} ms  ({100*glue/full_eager:.1f}% of block)")

print("\n=== megafusion: torch.compile the block (inductor fuses pointwise glue) ===")
try:
    cblock = torch.compile(block, mode="max-autotune-no-cudagraphs")
    with torch.no_grad():
        for _ in range(3): cblock(x)  # trigger compile
    full_comp = t(lambda: cblock(x), it=30, wu=5)
    print(f"  FULL eager    = {full_eager:.3f} ms")
    print(f"  FULL compiled = {full_comp:.3f} ms   ({full_eager/full_comp:.2f}x, -{100*(1-full_comp/full_eager):.0f}%)")
    # correctness
    with torch.no_grad():
        a = block(x); b = cblock(x)
    print(f"  max_abs_diff eager-vs-compiled = {(a.float()-b.float()).abs().max().item():.3e}")
except Exception as ex:
    import traceback; traceback.print_exc(); print("compile failed:", ex)
