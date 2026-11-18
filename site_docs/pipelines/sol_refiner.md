# SoL-Refiner

SoL-Refiner turns low-resolution videos into high-resolution outputs with a single target-resolution denoising step. A base generator (SANA-Video, WAN, Cosmos-Nano or MiniMax-H3) produces a low-resolution draft; SoL-Refiner restores detail at the target resolution. Code and full docs: [`models/sol-refiner`](https://github.com/NVlabs/Sana/tree/sol-engine/models/sol-refiner).

## Entry points

| Version | Refiner backbone | Code | Weights |
|---|---|---|---|
| Original paper version | LTX-2.3 | [LTX-2.3](https://github.com/NVlabs/Sana/tree/sol-engine/models/sol-refiner/LTX-2.3) | [One-step](https://huggingface.co/Efficient-Large-Model/SoL-Refiner-LTX-2.3-One-Step) · [Multi-step](https://huggingface.co/Efficient-Large-Model/SoL-Refiner-LTX-2.3-Multi-Step) |
| MiniMax-H3 extension | LTX-2.5 | [MiniMax-H3](https://github.com/NVlabs/Sana/tree/sol-engine/models/sol-refiner/MiniMax-H3) | [One-step](https://huggingface.co/Efficient-Large-Model/SoL-Refiner-LTX-2.5-for-MiniMax-H3) |

The LTX-2.3 entry outputs 2048x1152 by default and runs both baseline and SoL-Engine execution. The MiniMax-H3 entry refines an existing H3 video to 1920x1080 and was tested on H100 80GB.

## Project page

- [SoL-Refiner](https://nvlabs.github.io/Sana/Sol-Refiner/): results, measurement protocols and video comparisons.
