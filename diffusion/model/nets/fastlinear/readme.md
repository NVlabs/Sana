# a fast implementation of linear attention

## 64x64, fp16

```bash
# validate correctness
## fp16 vs fp32
python -m develop_triton_litemla attn_type=LiteMLA test_correctness=True
## triton fp16 vs fp32
python -m develop_triton_litemla attn_type=TritonLiteMLA test_correctness=True

# test performance
## fp16, forward
python -m develop_triton_litemla attn_type=LiteMLA
each step takes 10.81 ms
max memory allocated: 2.2984 GB

## triton fp16, forward
python -m develop_triton_litemla attn_type=TritonLiteMLA
each step takes 4.70 ms
max memory allocated: 1.6480 GB

## fp16, backward
python -m develop_triton_litemla attn_type=LiteMLA backward=True
each step takes 35.34 ms
max memory allocated: 3.4412 GB

## triton fp16, backward
python -m develop_triton_litemla attn_type=TritonLiteMLA backward=True
each step takes 14.25 ms
max memory allocated: 2.4704 GB
```
