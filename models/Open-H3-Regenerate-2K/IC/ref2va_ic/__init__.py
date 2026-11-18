"""Open-H3-Regenerate-2K-IC reference implementation.

The draft (low-resolution clip) is kept in MiniMax-H3's Ref2VA context as a reference video; a LoRA on the Ref2VA
transformer (`transformer_ref`) generates the high-resolution version of the same clip.  Modules:

  media      read a draft at 24 fps onto the LQ canvas, write the result with the draft's audio
  encode     H3 VAE latents of the draft, Qwen3-VL prompt embeddings with the draft as <Video 1>
  layout     the packed Ref2VA sequence [context | reference video | silent audio | target video] and row timesteps
  lora       the IC LoRA on the 312 attention / MLP linears (unmerged, y = W x + B A x)
  hyperflow  the HyperFlow 8-step adapter merged into the base (two-time embedder + LoRA), its sigma grid
  cp         Ulysses context parallelism over the packed sequence (1, 2, 4, 7 or 8 GPUs)
  sampler    the Euler loop (CFG 1) for the supported samplers
"""
