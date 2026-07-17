"""Per-stage parallel scaling, to find each stage's marginal (diminishing-returns) point.
For a fixed request, if a stage gets N cards:
- Attention (Ulysses splits heads): each card does full seq S_sample with 16/N heads.
- FFN (splits sequence): each card does S_total/N tokens through all 128 experts (grouped_mm).
Per-card time = the stage's per-layer time on N cards (they run in parallel). Efficiency
eff = (time@1 / N) / time@N  (1.0 = perfect linear scaling; <1 = diminishing returns).
Where eff drops is that stage's marginal point -> how far it's worth splitting it.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

H_MODEL, I_MOE, E, TOPK, HEADS, D = 2048, 768, 128, 8, 16, 128
dev = torch.device("cuda", 0); dt = torch.bfloat16
torch.manual_seed(0)
CUDNN = [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]

RES = {"480p": (2*31*(480//16)*(832//16)),
       "720p": (2*31*(736//16)*(1280//16)),
       "1080p": (2*31*(1088//16)*(1920//16))}  # S_total (x2 batch_cfg)

w1 = torch.randn(E, I_MOE, H_MODEL, device=dev, dtype=dt)*0.02
w3 = torch.randn(E, I_MOE, H_MODEL, device=dev, dtype=dt)*0.02
w2 = torch.randn(E, H_MODEL, I_MOE, device=dev, dtype=dt)*0.02


def t(fn, it=20, wu=6):
    for _ in range(wu): fn()
    torch.cuda.synchronize()
    s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/it


def attn_ms(S_sample, heads):
    q=torch.randn(1,heads,S_sample,D,device=dev,dtype=dt); k=torch.randn_like(q); v=torch.randn_like(q)
    def f():
        with sdpa_kernel(CUDNN,set_priority=True): F.scaled_dot_product_attention(q,k,v)
    return t(f)


def ffn_ms(tokens):
    A=max(tokens*TOPK,E); per=max(A//E,1); x=torch.randn(per*E,H_MODEL,device=dev,dtype=dt)*0.1
    offs=torch.cumsum(torch.full((E,),per,device=dev,dtype=torch.int64),0).to(torch.int32)
    def f():
        h=F.silu(torch._grouped_mm(x,w1.transpose(-2,-1),offs=offs))
        h=h*torch._grouped_mm(x,w3.transpose(-2,-1),offs=offs)
        torch._grouped_mm(h,w2.transpose(-2,-1),offs=offs)
    return t(f)


print(f"GPU: {torch.cuda.get_device_name(0)}  per-CARD time & scaling efficiency vs #cards")
for name,S_total in RES.items():
    S_sample=S_total//2
    print(f"\n===== {name}  S_total={S_total} =====")
    print(f"{'N cards':>7} | {'attn(h/card ms)':>18} {'eff':>5} | {'ffn(tok/card ms)':>19} {'eff':>5} | bottleneck")
    a1=attn_ms(S_sample,HEADS); f1=ffn_ms(S_total)
    for N in (1,2,4,8,16):
        h=HEADS//N
        am=attn_ms(S_sample,h) if h>=1 else None
        fm=ffn_ms(S_total//N)
        aeff=(a1/N)/am if am else 0
        feff=(f1/N)/fm
        bott = "ATTN" if (am or 0) > fm else "ffn"
        astr=f"{h:>2}h {am:>7.3f}" if am else "  -   -"
        print(f"{N:>7} | {astr:>18} {aeff:>5.2f} | {S_total//N:>8}t {fm:>7.3f} {feff:>5.2f} | {bott} ({(am or 0)/fm:.1f}x)")
print()
print("eff=1 perfect linear; <1 diminishing. attn scales by heads (max 16). ffn scales by tokens.")
print("Balanced disagg: pick N_attn, N_ffn so attn_ms(N_attn) ~= ffn_ms(N_ffn).")
