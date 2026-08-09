"""Does grouped_mm stay saturated as we scale cards (per-rank tokens shrink)?
For each resolution, per-rank MoE assignments = (S_total/n_cards)*top_k, spread over
128 experts. Measure ns/token at 4/8/16-card token counts. If ns/token rises as
cards grow, the FFN is under-saturated -> FFN scaling < ideal. Implied FFN speedup
4->8 = 2 * t(4card)/t(8card).
"""
import torch
import torch.nn.functional as F

H, I, E, TOPK = 2048, 768, 128, 8
dev = torch.device("cuda", 0); dt = torch.bfloat16
torch.manual_seed(0)
if not hasattr(torch, "_grouped_mm"):
    print("no _grouped_mm"); raise SystemExit

S_TOTAL = {"480p": 2 * 31 * (480//16) * (832//16),
           "720p": 2 * 31 * (736//16) * (1280//16),
           "1080p": 2 * 31 * (1088//16) * (1920//16)}  # x2 = batch_cfg

w1 = torch.randn(E, I, H, device=dev, dtype=dt) * 0.02
w3 = torch.randn(E, I, H, device=dev, dtype=dt) * 0.02
w2 = torch.randn(E, H, I, device=dev, dtype=dt) * 0.02


def bench(assignments, iters=30, warmup=8):
    per = max(assignments // E, 1)
    total = per * E
    x = torch.randn(total, H, device=dev, dtype=dt) * 0.1
    offs = torch.cumsum(torch.full((E,), per, device=dev, dtype=torch.int64), 0).to(torch.int32)
    def one():
        h = F.silu(torch._grouped_mm(x, w1.transpose(-2, -1), offs=offs))
        h = h * torch._grouped_mm(x, w3.transpose(-2, -1), offs=offs)
        torch._grouped_mm(h, w2.transpose(-2, -1), offs=offs)
    for _ in range(warmup): one()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): one()
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e)/iters
    return ms, ms*1e6/total   # ms, ns/token


print(f"GPU: {torch.cuda.get_device_name(0)}  grouped_mm 128 experts, ns/token vs card count")
print(f"{'res':>6} | {'4-card':>16} | {'8-card':>16} | {'16-card':>16} | {'FFN 4->8':>9} | {'FFN 8->16':>9}")
print(f"{'':>6} | {'M/exp  ns/tok':>16} | {'M/exp  ns/tok':>16} | {'M/exp  ns/tok':>16} | {'speedup':>9} | {'speedup':>9}")
print("-"*96)
for name, S in S_TOTAL.items():
    r = {}
    for nc in (4, 8, 16):
        A = (S // nc) * TOPK
        ms, nspt = bench(A)
        r[nc] = (A // E, nspt)
    su_48 = 2 * r[4][1] / r[8][1]     # ideal 2x scaled by efficiency ratio
    su_816 = 2 * r[8][1] / r[16][1]
    print(f"{name:>6} | {r[4][0]:>6} {r[4][1]:>7.2f} | {r[8][0]:>6} {r[8][1]:>7.2f} | "
          f"{r[16][0]:>6} {r[16][1]:>7.2f} | {su_48:>8.2f}x | {su_816:>8.2f}x")
print()
print("FFN speedup 4->8 = 2 * ns/token(4card)/ns/token(8card). If <2x, FFN under-saturated at 8 cards.")
print("(This is grouped_mm compute only; ignores the all-to-all a transfeat hybrid would add.)")
