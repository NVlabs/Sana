#!/usr/bin/env python3
"""
Tiny AutoEncoder for Hunyuan Video
(DNN for encoding / decoding videos to Hunyuan Video's latent space)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from collections import namedtuple

TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))

def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), nn.ReLU(inplace=True), conv(n_out, n_out), nn.ReLU(inplace=True), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)
    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))

class SuperMemBlock(nn.Module):
    """MemBlock variant used by the Super decoder (ConvNeXt-style: 7x7 depthwise conv + inverted bottleneck)."""
    def __init__(self, n_f):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_f*2, n_f*2, 7, padding=3, groups=n_f*2, bias=False),
            nn.Conv2d(n_f*2, n_f*4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(n_f*4, n_f, 1, bias=False),
        )
    def forward(self, x, past):
        return self.conv(torch.cat([x, past], 1)) + x

class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f*stride,n_f, 1, bias=False)
    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))

class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f*stride, 1, bias=False)
    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)

def apply_model_with_memblocks_parallel(model, x, show_progress_bar):
    """
    Apply a sequential model with memblocks to the given input,
    with parallelization over the time axis and iteration over blocks.

    Args:
    - model: nn.Sequential of blocks to apply
    - x: input data, of dimensions NTCHW
    - show_progress_bar: if True, enables tqdm progressbar display

    Returns NTCHW tensor of output data.
    """
    assert x.ndim == 5, f"TAEHV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    N, T, C, H, W = x.shape
    x = x.reshape(N*T, C, H, W)

    # parallel over input timesteps, iterate over blocks
    for b in tqdm(model, disable=not show_progress_bar):
        if isinstance(b, (MemBlock, SuperMemBlock)):
            NT, C, H, W = x.shape
            T = NT // N
            _x = x.reshape(N, T, C, H, W)
            # pad with zeros along time axis (i.e. empty memory), slice
            block_memory = F.pad(_x, (0,0,0,0,0,0,1,0), value=0)[:,:T].reshape(x.shape)
            x = b(x, block_memory)
        else:
            x = b(x)
    NT, C, H, W = x.shape
    T = NT // N
    return x.view(N, T, C, H, W)

def apply_model_with_memblocks_sequential_single_step(model, memory, work_queue, progress_bar=None):
    """
    Process the work queue (a graph traversal over blocks and timesteps)
    until an output frame is produced or the queue is empty.
    Mutates memory and work_queue in place.

    Returns N1CHW output tensor, or None if the queue needs more input.
    """
    while work_queue:
        xt, i = work_queue.pop(0)
        if progress_bar is not None and i == 0:
            progress_bar.update(1)
        if i == len(model):
            return xt.unsqueeze(1)
        b = model[i]
        if isinstance(b, (MemBlock, SuperMemBlock)):
            # mem blocks are simple since we're visiting the graph in causal order
            if memory[i] is None:
                xt_new = b(xt, xt * 0)
            else:
                xt_new = b(xt, memory[i])
            memory[i] = xt
            work_queue.insert(0, TWorkItem(xt_new, i+1))
        elif isinstance(b, TPool):
            # pool blocks accumulate inputs until they have enough to pool
            if memory[i] is None:
                memory[i] = []
            memory[i].append(xt)
            if len(memory[i]) > b.stride:
                raise ValueError(f"TPool memory overflow: {len(memory[i])} items for stride {b.stride}")
            elif len(memory[i]) == b.stride:
                N, C, H, W = xt.shape
                xt = b(torch.cat(memory[i], 1).view(N*b.stride, C, H, W))
                memory[i] = []
                work_queue.insert(0, TWorkItem(xt, i+1))
        elif isinstance(b, TGrow):
            xt = b(xt)
            NT, C, H, W = xt.shape
            for xt_next in reversed(xt.view(NT//b.stride, b.stride*C, H, W).chunk(b.stride, 1)):
                work_queue.insert(0, TWorkItem(xt_next, i+1))
        else:
            xt = b(xt)
            work_queue.insert(0, TWorkItem(xt, i+1))
    return None

def apply_model_with_memblocks_sequential(model, x, show_progress_bar):
    """
    Apply a sequential model with memblocks to the given input,
    with iteration over timesteps as well as blocks.

    Args:
    - model: nn.Sequential of blocks to apply
    - x: input data, of dimensions NTCHW
    - show_progress_bar: if True, enables tqdm progressbar display

    Returns NTCHW tensor of output data.
    """
    assert x.ndim == 5, f"TAEHV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    work_queue = [TWorkItem(xt, 0) for xt in x.unbind(1)]
    memory = [None] * len(model)
    progress_bar = tqdm(range(len(work_queue)), disable=not show_progress_bar)
    out = []
    while work_queue:
        xt = apply_model_with_memblocks_sequential_single_step(model, memory, work_queue, progress_bar)
        if xt is not None:
            out.append(xt)
    progress_bar.close()
    return torch.cat(out, 1)

def apply_model_with_memblocks(model, x, parallel, show_progress_bar):
    """
    Apply a sequential model with memblocks to the given input.
    Args:
    - model: nn.Sequential of blocks to apply
    - x: input data, of dimensions NTCHW
    - parallel: if True, parallelize over timesteps (fast but uses O(T) memory)
        if False, each timestep will be processed sequentially (slow but uses O(1) memory)
    - show_progress_bar: if True, enables tqdm progressbar display

    Returns NTCHW tensor of output data.
    """
    if parallel:
        return apply_model_with_memblocks_parallel(model, x, show_progress_bar)
    else:
        return apply_model_with_memblocks_sequential(model, x, show_progress_bar)

class TAEHV(nn.Module):
    def __init__(self, checkpoint_path="taehv.pth", encoder_time_downscale=(True, True, False), decoder_time_upscale=(False, True, True), decoder_space_upscale=(True, True, True), patch_size=1, latent_channels=16, arch_variant=None):
        """Initialize pretrained TAEHV from the given checkpoint.

        Arg:
            checkpoint_path: path to weight file to load. taehv.pth for Hunyuan, taew2_1.pth for Wan 2.1.
            encoder_time_downscale: whether temporal downsampling is enabled for each block.
            decoder_time_upscale: whether temporal upsampling is enabled for each block. upsampling can be disabled for a cheaper preview.
            decoder_space_upscale: whether spatial upsampling is enabled for each block. upsampling can be disabled for a cheaper preview.
            patch_size: input/output pixelshuffle patch-size for this model.
            latent_channels: number of latent channels (z dim) for this model.
            arch_variant: decoder architecture variant. None (base) or "super" (higher-quality, ~2x decoder params). Autodetected from filename if None.
        """
        super().__init__()
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.image_channels = 3
        if len(decoder_time_upscale) == 2:
            decoder_time_upscale = (False, *decoder_time_upscale)
        self.is_cogvideox = checkpoint_path is not None and "taecvx" in checkpoint_path
        self.is_h3 = checkpoint_path is not None and "taeh3" in checkpoint_path
        if checkpoint_path is not None and "taew2_2" in checkpoint_path:
            self.patch_size, self.latent_channels = 2, 48
        if checkpoint_path is not None and "taehv1_5" in checkpoint_path:
            self.patch_size, self.latent_channels = 2, 32
        if self.is_h3:
            self.patch_size, self.latent_channels, encoder_time_downscale = 2, 24, (True, True, False)
        if checkpoint_path is not None and "taeltx" in checkpoint_path: # same for both 2 and 2.3
            self.patch_size, self.latent_channels, encoder_time_downscale, decoder_time_upscale = 4, 128, (True, True, True), (True, True, True)
        if arch_variant is None and checkpoint_path is not None and "_super" in checkpoint_path:
            arch_variant = "super"
        assert arch_variant in (None, "super"), f"unrecognized arch_variant {arch_variant!r}"
        self.encoder = nn.Sequential(
            conv(self.image_channels*self.patch_size**2, 64), nn.ReLU(inplace=True),
            TPool(64, 2 if encoder_time_downscale[0] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 2 if encoder_time_downscale[1] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 2 if encoder_time_downscale[2] else 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            conv(64, self.latent_channels),
        )
        if arch_variant == "super":
            n_f = [512, 256, 128, 64]
            self.decoder = nn.Sequential(
                nn.Conv2d(self.latent_channels, n_f[0], 1, bias=False),
                SuperMemBlock(n_f[0]), SuperMemBlock(n_f[0]), SuperMemBlock(n_f[0]), conv(n_f[0], n_f[1]*(2 if decoder_space_upscale[0] else 1)**2), nn.ReLU(inplace=True), nn.PixelShuffle(2 if decoder_space_upscale[0] else 1), TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1),
                SuperMemBlock(n_f[1]), SuperMemBlock(n_f[1]), SuperMemBlock(n_f[1]), conv(n_f[1], n_f[2]*(2 if decoder_space_upscale[1] else 1)**2), nn.ReLU(inplace=True), nn.PixelShuffle(2 if decoder_space_upscale[1] else 1), TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1),
                SuperMemBlock(n_f[2]), SuperMemBlock(n_f[2]), SuperMemBlock(n_f[2]), conv(n_f[2], n_f[3]*(2 if decoder_space_upscale[2] else 1)**2), nn.ReLU(inplace=True), nn.PixelShuffle(2 if decoder_space_upscale[2] else 1), TGrow(n_f[3], 2 if decoder_time_upscale[2] else 1),
                conv(n_f[3], self.image_channels*self.patch_size**2),
            )
        else:
            n_f = [256, 128, 64, 64]
            self.decoder = nn.Sequential(
                Clamp(), conv(self.latent_channels, n_f[0]), nn.ReLU(inplace=True),
                MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), MemBlock(n_f[0], n_f[0]), nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1), TGrow(n_f[0], 2 if decoder_time_upscale[0] else 1), conv(n_f[0], n_f[1], bias=False),
                MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), MemBlock(n_f[1], n_f[1]), nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1), TGrow(n_f[1], 2 if decoder_time_upscale[1] else 1), conv(n_f[1], n_f[2], bias=False),
                MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), MemBlock(n_f[2], n_f[2]), nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1), TGrow(n_f[2], 2 if decoder_time_upscale[2] else 1), conv(n_f[2], n_f[3], bias=False),
                nn.ReLU(inplace=True), conv(n_f[3], self.image_channels*self.patch_size**2),
            )
        # computed properties
        self.t_downscale = 2**sum(t.stride == 2 for t in self.encoder if isinstance(t, TPool))
        self.t_upscale = 2**sum(t.stride == 2 for t in self.decoder if isinstance(t, TGrow))
        self.frames_to_trim = self.t_upscale - 1

        if checkpoint_path is not None:
            self.load_state_dict(self.patch_tgrow_layers(torch.load(checkpoint_path, map_location="cpu", weights_only=True)))

    def patch_tgrow_layers(self, sd):
        """Patch TGrow layers to use a smaller kernel if needed.

        Args:
            sd: state dict to patch
        """
        new_sd = self.state_dict()
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, TGrow):
                key = f"decoder.{i}.conv.weight"
                if sd[key].shape[0] > new_sd[key].shape[0]:
                    # take the last-timestep output channels
                    sd[key] = sd[key][-new_sd[key].shape[0]:]
        return sd

    def preprocess_input_frames(self, x):
        """Preprocess RGB input frames prior to the main encoder sequence."""
        if self.patch_size > 1: x = F.pixel_unshuffle(x, self.patch_size)
        return x

    def _encode_h3_video(self, x, parallel, show_progress_bar):
        """Match H3's 17-frame chunks and three-token drop.
        https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/fa9c8ab1eaa21c8ae25e7e40b83b2e6002f340af/FL2VA/video_vae/klvae.py#L461-L503
        """
        batch = x.shape[0]
        x = torch.cat([x, x[:, -1:].expand(-1, -x.shape[1] % 17, -1, -1, -1)], dim=1)
        x = F.pad(x.reshape(batch, -1, 17, *x.shape[2:]), (0, 0, 0, 0, 0, 0, 3, 0))
        x = self.preprocess_input_frames(x)
        if parallel:
            x = apply_model_with_memblocks(self.encoder, x.flatten(0, 1), True, show_progress_bar)
            x = x.reshape(batch, -1, *x.shape[2:])
        else:
            x = torch.cat([apply_model_with_memblocks(self.encoder, chunk, False, False)
                           for chunk in tqdm(x.unbind(1), disable=not show_progress_bar)], dim=1)
        return x[:, :-3]

    def _decode_h3_video(self, x, parallel, show_progress_bar):
        """Match H3's five-token chunks and per-chunk prefix trim.
        https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/fa9c8ab1eaa21c8ae25e7e40b83b2e6002f340af/FL2VA/video_vae/klvae.py#L678-L786
        """
        x = apply_model_with_memblocks(self.decoder, x, parallel, show_progress_bar)
        chunk_frames = 5 * self.t_upscale
        x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, -x.shape[1] % chunk_frames))
        x = x.unflatten(1, (-1, chunk_frames))[:, :, self.frames_to_trim:].flatten(1, 2)
        return self.postprocess_output_frames(x[:, :-3 * self.t_upscale])

    def encode_video(self, x, parallel=True, show_progress_bar=True):
        """Encode a sequence of frames.

        Args:
            x: input NTCHW RGB (C=3) tensor with values in [0, 1].
            parallel: if True, all frames will be processed at once.
              (this is faster but may require more memory).
              if False, frames will be processed sequentially.
        Returns NTCHW latent tensor with ~Gaussian values.
        """
        if self.is_h3:
            return self._encode_h3_video(x, parallel, show_progress_bar)
        x = self.preprocess_input_frames(x)
        if x.shape[1] % self.t_downscale != 0:
            # pad at end to multiple of self.t_downscale
            n_pad = self.t_downscale - x.shape[1] % self.t_downscale
            padding = x[:, -1:].repeat_interleave(n_pad, dim=1)
            x = torch.cat([x, padding], 1)
        return apply_model_with_memblocks(self.encoder, x, parallel, show_progress_bar)

    def postprocess_output_frames(self, x):
        """Postprocess RGB frames after the main decoder sequence."""
        if self.patch_size > 1: x = F.pixel_shuffle(x, self.patch_size)
        return x.clamp_(0, 1)

    def decode_video(self, x, parallel=True, show_progress_bar=True):
        """Decode a sequence of frames.

        Args:
            x: input NTCHW latent (C=self.latent_channels) tensor with ~Gaussian values.
            parallel: if True, all frames will be processed at once.
              (this is faster but may require more memory).
              if False, frames will be processed sequentially.
        Returns NTCHW RGB tensor with ~[0, 1] values.
        """
        if self.is_h3:
            return self._decode_h3_video(x, parallel, show_progress_bar)
        skip_trim = self.is_cogvideox and x.shape[1] % 2 == 0
        x = apply_model_with_memblocks(self.decoder, x, parallel, show_progress_bar)
        x = self.postprocess_output_frames(x)
        if skip_trim:
            # skip trimming for cogvideox to make frame counts match.
            # this still doesn't have correct temporal alignment for certain frame counts
            # (cogvideox seems to pad at the start?), but for multiple-of-4 it's fine.
            return x
        return x[:, self.frames_to_trim:]

class StreamingTAEHV(nn.Module):
    def __init__(self, taehv):
        """Streaming wrapper around TAEHV for real-time use-cases (where not all inputs are available immediately).

        Encode-decode (video-to-video) usage:
            streaming = StreamingTAEHV(taehv)
            for frame in video_frames:
                latent = streaming.encode(frame_tensor)
                decoded = streaming.decode(latent)  # feeds latent if not None, then returns next frame
                if decoded is not None:
                    display(decoded)
            for frame in streaming.flush():
                display(frame)

        Decode-only (world model) usage:
            streaming = StreamingTAEHV(taehv)
            while running:
                latent = world_model.step()       # latent represents t_upscale frames
                frame = streaming.decode(latent)   # returns first frame immediately
                while frame is not None:           # retrieve remaining frames from this latent
                    display(frame)
                    frame = streaming.decode()
        """
        super().__init__()
        self.taehv = taehv
        self.reset()

    def reset(self):
        """Reset all internal state. Call this to start encoding/decoding a new stream."""
        self.encoder_work_queue, self.encoder_memory = [], [None] * len(self.taehv.encoder)
        self.decoder_work_queue, self.decoder_memory = [], [None] * len(self.taehv.decoder)
        self.n_frames_encoded, self.n_frames_decoded = 0, 0
        self._last_encoder_input_frame = None

    def encode(self, x=None):
        """Feed an input frame (optional) and try to produce an encoder output.

        The encoder accumulates t_downscale input frames before producing one latent,
        so most calls will return None. Use flush_encoder() at end-of-stream to pad and
        drain any remaining latents.

        Args:
            x: NTCHW RGB frame tensor with values in [0, 1], or None to just process pending work.
        Returns: N1CHW latent tensor, or None if not enough input has been accumulated.
        """
        if x is not None:
            assert x.ndim == 5 and x.shape[2] == self.taehv.image_channels, f"Expected NTCHW frames but got {x.shape=}"
            self._last_encoder_input_frame = x[:, -1:]
            x = self.taehv.preprocess_input_frames(x)
            self.encoder_work_queue.extend(TWorkItem(xt, 0) for xt in x.unbind(1))
            self.n_frames_encoded += x.shape[1]
        xt = apply_model_with_memblocks_sequential_single_step(
            self.taehv.encoder, self.encoder_memory, self.encoder_work_queue)
        return xt

    def decode(self, x=None):
        """Feed a latent (optional) and try to produce a decoded frame.

        Each latent produces t_upscale output frames due to temporal upscaling. The first
        decode(latent) call returns the first of these frames; call decode() with no argument
        to retrieve the rest, one at a time. Each call does the minimum decoder work needed to
        produce one frame.

        Startup frames (the first frames_to_trim raw decoder outputs, used for causal alignment
        with the reference VAE) are consumed internally and never returned.

        Args:
            x: NTCHW latent tensor, or None to retrieve the next pending frame.
        Returns: N1CHW decoded RGB frame tensor, or None if the queue needs more input.
        """
        if x is not None:
            assert x.ndim == 5 and x.shape[2] == self.taehv.latent_channels, f"Expected NTCHW latents but got {x.shape=}"
            self.decoder_work_queue.extend(TWorkItem(xt, 0) for xt in x.unbind(1))
        while True:
            xt = apply_model_with_memblocks_sequential_single_step(
                self.taehv.decoder, self.decoder_memory, self.decoder_work_queue)
            if xt is None:
                return None
            self.n_frames_decoded += 1
            # skip startup frames (to match decode_video trim behavior)
            if not self.taehv.is_cogvideox and self.n_frames_decoded <= self.taehv.frames_to_trim:
                continue
            return self.taehv.postprocess_output_frames(xt)

    def flush_encoder(self):
        """Pad (if needed) and drain all remaining latents from the encoder.

        Returns list of N1CHW latent tensors.
        """
        latents = []
        if self._last_encoder_input_frame is not None and self.n_frames_encoded % self.taehv.t_downscale != 0:
            n_pad = self.taehv.t_downscale - self.n_frames_encoded % self.taehv.t_downscale
            for _ in range(n_pad):
                lat = self.encode(self._last_encoder_input_frame)
                if lat is not None:
                    latents.append(lat)
        while (lat := self.encode()) is not None:
            latents.append(lat)
        return latents

    def flush_decoder(self):
        """Drain all remaining decoded frames from the decoder.

        Returns list of N1CHW decoded RGB frame tensors.
        """
        frames = []
        while (frame := self.decode()) is not None:
            frames.append(frame)
        return frames

    def flush(self):
        """Flush encoder (with padding) and decoder, returning all remaining decoded frames.

        Returns list of N1CHW decoded RGB frame tensors.
        """
        frames = []
        for latent in self.flush_encoder():
            frame = self.decode(latent)
            if frame is not None:
                frames.append(frame)
        frames.extend(self.flush_decoder())
        return frames

@torch.no_grad()
def main():
    """Run TAEHV roundtrip reconstruction on the given video paths."""
    import os
    import sys
    import cv2 # no highly esteemed deed is commemorated here

    class VideoTensorReader:
        def __init__(self, video_file_path):
            self.cap = cv2.VideoCapture(video_file_path)
            assert self.cap.isOpened(), f"Could not load {video_file_path}"
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        def __iter__(self):
            return self
        def __next__(self):
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                raise StopIteration  # End of video or error
            return torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1) # BGR HWC -> RGB CHW

    class VideoTensorWriter:
        def __init__(self, video_file_path, width_height, fps=30):
            self.writer = cv2.VideoWriter(video_file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, width_height)
            assert self.writer.isOpened(), f"Could not create writer for {video_file_path}"
        def write(self, frame_tensor):
            assert frame_tensor.ndim == 3 and frame_tensor.shape[0] == 3, f"{frame_tensor.shape}??"
            self.writer.write(cv2.cvtColor(frame_tensor.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)) # RGB CHW -> BGR HWC
        def __del__(self):
            if hasattr(self, 'writer'): self.writer.release()

    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16
    checkpoint_path = os.getenv("TAEHV_CHECKPOINT_PATH", "taehv.pth")
    checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    print(f"Using device \033[31m{dev}\033[0m, dtype \033[32m{dtype}\033[0m, checkpoint \033[34m{checkpoint_name}\033[0m ({checkpoint_path})")
    taehv = TAEHV(checkpoint_path=checkpoint_path).to(dev, dtype)
    for video_path in sys.argv[1:]:
        print(f"Processing {video_path}...")
        video_in = VideoTensorReader(video_path)
        video = torch.stack(list(video_in), 0)[None]
        vid_dev = video.to(dev, dtype).div_(255.0)
        # convert to device tensor
        if video.numel() < 100_000_000:
            print(f"  {video_path} seems small enough, will process all frames in parallel")
            # convert to device tensor
            vid_enc = taehv.encode_video(vid_dev)
            print(f"  Encoded {video_path} -> {vid_enc.shape}. Decoding...")
            vid_dec = taehv.decode_video(vid_enc)
            print(f"  Decoded {video_path} -> {vid_dec.shape}")
        else:
            print(f"  {video_path} seems large, will process each frame sequentially")
            # convert to device tensor
            vid_enc = taehv.encode_video(vid_dev, parallel=False)
            print(f"  Encoded {video_path} -> {vid_enc.shape}. Decoding...")
            vid_dec = taehv.decode_video(vid_enc, parallel=False)
            print(f"  Decoded {video_path} -> {vid_dec.shape}")
        video_out_path = video_path + f".reconstructed_by_{checkpoint_name}.mp4"
        video_out = VideoTensorWriter(video_out_path, (vid_dec.shape[-1], vid_dec.shape[-2]), fps=int(round(video_in.fps)))
        for frame in vid_dec.clamp_(0, 1).mul_(255).round_().byte().cpu()[0]:
            video_out.write(frame)
        print(f"  Saved to {video_out_path}")

if __name__ == "__main__":
    main()
