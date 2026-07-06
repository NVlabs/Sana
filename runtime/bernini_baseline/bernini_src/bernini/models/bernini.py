# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import torch
import torch.nn as nn
import torch.distributed as dist
import os
from transformers import UMT5EncoderModel, Qwen2_5_VLConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .diffloss_fm import DiffLoss_FM
from .wan_diffusion import GEN_Wanx22
from .modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration


logger = logging.get_logger(__name__)


def _join_subfolder(base_subfolder, leaf):
    if base_subfolder:
        return f"{base_subfolder}/{leaf}"
    return leaf


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dtype)


class MLPConnector(nn.Module):
    """Local connector implementation used by the Bernini checkpoint.

    Keep the same parameter layout as the original training connector so the
    released checkpoint loads without the external training package.
    """

    def __init__(
        self,
        in_dim,
        num_layers_for_gen=1,
        out_dim_for_gen=4096,
        enable_gen_branch=True,
        gen_head_type="mlp",
        num_layers_for_vit=1,
        out_dim_for_vit=3584,
        enable_vit_branch=True,
    ):
        super().__init__()
        self.enable_gen_branch = enable_gen_branch
        self.enable_vit_branch = enable_vit_branch
        if enable_gen_branch:
            self.proj_gen = nn.Sequential(
                nn.Linear(in_dim, out_dim_for_gen),
                nn.GELU(),
                RMSNorm(out_dim_for_gen),
                nn.Linear(out_dim_for_gen, out_dim_for_gen),
            )
        if enable_vit_branch:
            self.pred_vit = nn.Sequential(
                nn.Linear(in_dim, out_dim_for_vit),
                nn.GELU(),
                nn.Linear(out_dim_for_vit, out_dim_for_vit),
                RMSNorm(out_dim_for_vit),
                nn.Linear(out_dim_for_vit, out_dim_for_vit),
            )

    @staticmethod
    def _run_projection(proj, x):
        param = next(proj.parameters(), None)
        if param is not None and (x.device != param.device or x.dtype != param.dtype):
            x = x.to(device=param.device, dtype=param.dtype)
        return proj(x)

    def for_gen(self, x):
        return self._run_projection(self.proj_gen, x)

    def for_vit(self, x):
        return self._run_projection(self.pred_vit, x)

def with_skip_config(config, skip_transformer_1=False, skip_transformer_2=False):
    """Create a copy of config with skip_transformer_* flags set."""
    config_copy = copy.deepcopy(config)
    config_copy.skip_transformer_1 = skip_transformer_1
    config_copy.skip_transformer_2 = skip_transformer_2
    return config_copy


def _join_if_present(base, *parts):
    if base is None:
        return None
    return os.path.join(base, *parts)


class BerniniConfig(PretrainedConfig):
    model_type = "bernini"

    def __init__(
        self,
        base_dir=None,
        mllm_config_path=None,
        mllm_subfolder=None,
        processor_config_path=None,
        processor_subfolder=None,
        mllm_attn_implementation="sdpa",
        diff_dec_config_path=None,
        transformer_config_path=None,
        transformer_2_config_path=None,
        scheduler_config_path=None,
        bernini_ckpt_subfolder=None,
        scratch_mllm=False,
        scratch=False,
        noise_tmin=0.0, # [0, 0.875] for WAN2.2 high noise; [0.875, 1.0] for WAN2.2 low noise
        noise_tmax=1.0, # [0, 1.0] for WAN2.1
        flow_shift=5,
        use_unipc=False,
        target_fps=16,
        switch_dit_boundary=0.875,
        shift=3.0,
        cotrain=False,
        # setting for clip fmmar
        num_mask_token=256,
        clip_diff_cfg=None,
        connector_cfg=None,
        mask_ratio_infer_cfg=None,
        feature_type_from_stage_one=None,
        additional_special_tokens=[],
        tie_word_embeddings=False,
        ema_decay=None,
        partial_pretrain_model=None,
        use_src_id_rotary_emb=False,
        interpolate_src_id=True,
        max_trained_src_id=5,
        max_sequence_length=512,
        # t5 embedding
        t5_text_encoder_path=None,
        t5_text_encoder_subfolder=None,
        t5_tokenizer_path=None,
        t5_tokenizer_subfolder=None,
        t5_max_sequence_length=512,
        t5_combine_type="kl_loss",
        vae_model_path=None,
        vae_subfolder=None,
        vae_config_path=None,
        wovae_task_list=['und_img', 'und_txt', 'und_vid'],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.base_dir = base_dir
        self.mllm_config_path = mllm_config_path if mllm_config_path is not None else base_dir
        self.mllm_subfolder = mllm_subfolder
        self.mllm_attn_implementation = mllm_attn_implementation
        self.processor_config_path = (
            processor_config_path if processor_config_path is not None else self.mllm_config_path
        )
        self.processor_subfolder = processor_subfolder
        self.diff_dec_config_path = diff_dec_config_path if diff_dec_config_path is not None else base_dir
        self.transformer_config_path = (
            transformer_config_path
            if transformer_config_path is not None
            else _join_if_present(base_dir, "transformer_config.json")
        )
        self.transformer_2_config_path = (
            transformer_2_config_path
            if transformer_2_config_path is not None
            else _join_if_present(base_dir, "transformer_2_config.json")
        )
        self.scheduler_config_path = scheduler_config_path or (
            os.path.join(base_dir, "scheduler")
            if base_dir is not None
            else None
        )
        self.bernini_ckpt_subfolder = bernini_ckpt_subfolder
        self.vae_model_path = vae_model_path if vae_model_path is not None else base_dir
        self.vae_subfolder = vae_subfolder
        self.vae_config_path = vae_config_path or (
            os.path.join(self.vae_model_path, _join_subfolder(self.vae_subfolder or "vae", "config.json"))
            if self.vae_model_path is not None
            else None
        )
        self.scratch = scratch
        self.scratch_mllm = scratch_mllm
        self.ema_decay = ema_decay
        self.noise_tmin = noise_tmin
        self.noise_tmax = noise_tmax
        self.flow_shift = flow_shift
        self.use_unipc = use_unipc
        self.target_fps = target_fps
        self.switch_dit_boundary = switch_dit_boundary
        self.shift = shift
        self.cotrain = cotrain
        self.use_src_id_rotary_emb = use_src_id_rotary_emb
        # When the number of conditioning segments exceeds `max_trained_src_id`
        # (the largest source_id seen in training), evenly map their ids into
        # the trained range [1, max_trained_src_id] instead of extrapolating to
        # unseen integer ids. The target segment keeps source_id 0.
        self.interpolate_src_id = interpolate_src_id
        self.max_trained_src_id = max_trained_src_id
        self.max_sequence_length = max_sequence_length
        self.wovae_task_list = wovae_task_list

        self.num_mask_token = num_mask_token
        self.clip_diff_cfg = clip_diff_cfg
        self.connector_cfg = connector_cfg
        self.mask_ratio_infer_cfg = mask_ratio_infer_cfg
        self.feature_type_from_stage_one = feature_type_from_stage_one
        self.additional_special_tokens = additional_special_tokens
        self.tie_word_embeddings = tie_word_embeddings
        self.partial_pretrain_model = partial_pretrain_model

        self.t5_text_encoder_path = t5_text_encoder_path if t5_text_encoder_path is not None else base_dir
        self.t5_text_encoder_subfolder = t5_text_encoder_subfolder
        self.t5_tokenizer_path = t5_tokenizer_path if t5_tokenizer_path is not None else base_dir
        self.t5_tokenizer_subfolder = t5_tokenizer_subfolder
        self.t5_max_sequence_length = t5_max_sequence_length
        self.t5_combine_type = t5_combine_type

        self.architectures = ["BerniniModel"]

class BerniniModel(PreTrainedModel):
    config_class = BerniniConfig
    def __init__(self, config):
        super().__init__(config)
        self.mllm = None
        self.diff_dec = None
        self.vit_decoder = None
        self.base_dir = getattr(config, "base_dir", None)
        self.mllm_config_path = config.mllm_config_path
        self.mllm_subfolder = getattr(config, "mllm_subfolder", None)
        self.diff_dec_config_path = config.diff_dec_config_path
        self.processor_config_path = config.processor_config_path
        self.feature_type_from_stage_one = config.feature_type_from_stage_one
        self.num_mask_token = config.num_mask_token
        self.use_t5_encoder = config.t5_text_encoder_path is not None

        # =============== Init MLLM ===============
        self.mllm_attn_implementation = config.mllm_attn_implementation
        logger.info(
            f"MLLM attention implement: config.mllm_attn_implementation={config.mllm_attn_implementation}"
        )
        if self.config.mllm_config_path is not None:
            mllm_config = Qwen2_5_VLConfig.from_pretrained(
                self.config.mllm_config_path,
                subfolder=self.config.mllm_subfolder,
            )
            if self.config.scratch_mllm:
                self.mllm = Qwen2_5_VLForConditionalGeneration._from_config(
                    mllm_config,
                    attn_implementation=config.mllm_attn_implementation,
                    torch_dtype=torch.bfloat16,
                )
            else:
                self.mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.config.mllm_config_path,
                    subfolder=self.config.mllm_subfolder,
                    attn_implementation=config.mllm_attn_implementation,
                )
            self.mask_tokens = nn.Parameter(torch.randn(1, self.num_mask_token, self.mllm.config.hidden_size) * 0.01)
            
        # =============== Init Diff Dec ===============
        if self.config.diff_dec_config_path:
            if getattr(self.config, "cotrain", False):
                self.diff_dec = GEN_Wanx22(with_skip_config(config, skip_transformer_1=False, skip_transformer_2=True))
                self.diff_dec_low = GEN_Wanx22(with_skip_config(config, skip_transformer_1=True, skip_transformer_2=False))
            else:
                self.diff_dec = GEN_Wanx22(config)
                self.diff_dec_low = None
       
        # =============== Init Connector ===============
        if config.connector_cfg.get('enable_gen_branch', True):
            assert self.diff_dec is not None
        if config.connector_cfg.get('enable_vit_branch', True):
            assert self.mllm is not None
        self.connector = MLPConnector(
            in_dim=self.mllm.config.hidden_size if self.mllm is not None else 3584,
            # setting for diffusion generator
            num_layers_for_gen=config.connector_cfg.get('num_layers_for_gen', 1),
            out_dim_for_gen=config.connector_cfg.get('out_dim_for_gen', 4096),
            enable_gen_branch=config.connector_cfg.get('enable_gen_branch', True),
            gen_head_type=config.connector_cfg.get('gen_head_type', 'mlp'),
            # setting for predict vit embed
            num_layers_for_vit=config.connector_cfg.get('num_layers_for_vit', 1),
            out_dim_for_vit=config.connector_cfg.get('out_dim_for_vit', 3584),
            enable_vit_branch=config.connector_cfg.get('enable_vit_branch', True),
        )

        # =============== vit decoder ===============
        self.vit_decoder = DiffLoss_FM(
            z_channels=config.clip_diff_cfg.get('z_channels', 3584),
            target_channels=config.clip_diff_cfg.get('target_channels', 3584),
            depth=config.clip_diff_cfg.get('depth', 16),
            width=config.clip_diff_cfg.get('width', 1536),
            diff_net=config.clip_diff_cfg.get("diff_net", "SimpleMLPAdaLN"),
            scheduler_type=config.clip_diff_cfg.get("scheduler_type", "FlowMatchScheduler"),
            shift=config.clip_diff_cfg.get("shift", 3.0),
            num_inference_steps=config.clip_diff_cfg.get("num_inference_steps", 100),
            extra_one_step=config.clip_diff_cfg.get("extra_one_step", True),
            diffusion_batch_mul=config.clip_diff_cfg.get("diffusion_batch_mul", 1),
            grad_checkpointing=True,
        )
        
        # =============== t5 embedding ===============
        if self.use_t5_encoder:
            logger.info(f"Initializing UMT5 encoder from {config.t5_text_encoder_path}")
            def load_t5_text_encoder():
                return UMT5EncoderModel.from_pretrained(
                    config.t5_text_encoder_path,
                    subfolder=config.t5_text_encoder_subfolder,
                    torch_dtype=torch.bfloat16,
                )

            # Stagger loading across ranks to reduce peak memory
            if dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                for r in range(world_size):
                    if r == rank:
                        self.t5_text_encoder = load_t5_text_encoder()
                    dist.barrier()
            else:
                self.t5_text_encoder = load_t5_text_encoder()
            self.t5_max_sequence_length = getattr(config, 't5_max_sequence_length', 512)
            self.t5_text_encoder.eval()
            for param in self.t5_text_encoder.parameters():
                param.requires_grad = False

    def get_t5_text_embeddings(self, input_ids, attention_mask, input_lens, pad_text_embeds=True):
        """
        Args:
            input_ids: tensor with shape (1, n) where n = sum of all sequence lengths
            attention_mask: tensor with shape (1, n)
            input_lens: tensor with shape (1, b) where b = batch_size
        Returns:
            batch_text_seqlen: list of t5_max_sequence_length repeated b times
            batch_text_embs: tensor with shape (1, b * t5_max_sequence_length, hidden_dim)
        """
        # Remove batch dim
        input_ids = input_ids.squeeze(0)  # (n,)
        attention_mask = attention_mask.squeeze(0)  # (n,)
        input_lens = input_lens.squeeze(0)  # (b,)
    
        batch_size = input_lens.size(0)
    
        # Split concatenated sequence into individual samples by lengths
        input_ids_list = torch.split(input_ids, input_lens.tolist())
        attention_mask_list = torch.split(attention_mask, input_lens.tolist())
    
        # Pad each sample to t5_max_sequence_length
        padded_input_ids = []
        padded_attention_mask = []
        for ids, mask in zip(input_ids_list, attention_mask_list):
            seq_len = ids.size(0)
            if seq_len < self.t5_max_sequence_length:
                pad_len = self.t5_max_sequence_length - seq_len
                ids = torch.cat([ids, ids.new_zeros(pad_len)])
                mask = torch.cat([mask, mask.new_zeros(pad_len)])
            else:
                ids = ids[:self.t5_max_sequence_length]
                mask = mask[:self.t5_max_sequence_length]
            padded_input_ids.append(ids)
            padded_attention_mask.append(mask)
    
        encoder_device = next(self.t5_text_encoder.parameters()).device

        # Stack to batch: (batch_size, t5_max_sequence_length)
        input_ids_batch = torch.stack(padded_input_ids, dim=0)
        attention_mask_batch = torch.stack(padded_attention_mask, dim=0)
        input_ids_batch = input_ids_batch.to(encoder_device)
        attention_mask_batch = attention_mask_batch.to(encoder_device)
    
        # Get actual sequence lengths (clamped to t5_max_sequence_length)
        seq_lens = torch.clamp(input_lens, max=self.t5_max_sequence_length)
    
        # Get embeddings
        with torch.no_grad():
            prompt_embeds = self.t5_text_encoder(
                input_ids_batch, attention_mask_batch
            ).last_hidden_state  # (batch_size, t5_max_sequence_length, hidden_dim)
    
        # Zero out padding positions
        if pad_text_embeds:
            prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
            prompt_embeds = torch.stack([
                torch.cat([u, u.new_zeros(self.t5_max_sequence_length - u.size(0), u.size(1))]) 
                for u in prompt_embeds
            ], dim=0)  # (batch_size, t5_max_sequence_length, hidden_dim)
        
            # Build return values
            batch_text_seqlen = [self.t5_max_sequence_length] * batch_size
            batch_text_embs = prompt_embeds.view(1, -1, prompt_embeds.size(-1))
            return batch_text_embs, batch_text_seqlen
        
        else:
            prompt_embeds = [u[:v].unsqueeze(0) for u, v in zip(prompt_embeds, seq_lens)]
            seq_lens = seq_lens.to(dtype=torch.long).cpu().tolist()
            batch_text_embs = torch.cat(prompt_embeds, dim=1)
            return batch_text_embs, seq_lens

    def get_t5_text_embeddings_sample(self, input_ids, attention_mask):
        encoder_device = next(self.t5_text_encoder.parameters()).device
        input_ids = input_ids.to(encoder_device)
        attention_mask = attention_mask.to(encoder_device)
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        with torch.no_grad():
            prompt_embeds = self.t5_text_encoder(
                input_ids, attention_mask).last_hidden_state
        prompt_embeds = [u[:min(v, self.t5_max_sequence_length)] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(prompt_embeds, dim=0)
        return prompt_embeds
    
    def get_ignore_modules_in_mixed_precision(self):
        from diffusers.models.embeddings import TimestepEmbedding
        from diffusers.models.normalization import FP32LayerNorm
        return (TimestepEmbedding, FP32LayerNorm)

    def post_process_input_embeds(
        self, 
        input_embeds, 
        visual_output_mask, 
        tgt_vit_mask,
        inference=False
    ):
        target_vit_embed_mask = visual_output_mask.squeeze(0)
        target_vit_embeds = input_embeds[:, target_vit_embed_mask, :]
        target_vit_embeds_gt = target_vit_embeds.clone()
        mask_token = self.mask_tokens[:, :1]
        
        if inference:
            # mask all tokens
            mask_rate = 1
            all_vit_token_num = sum(target_vit_embed_mask).detach().cpu().numpy()
            target_vit_embeds[:, :, :] = mask_token.expand(1, all_vit_token_num, -1)
            input_embeds[:, target_vit_embed_mask, :] = target_vit_embeds
            diff_loss_mask = torch.ones(all_vit_token_num).to(target_vit_embeds.device)
        
        elif tgt_vit_mask is not None:
            diff_loss_mask = tgt_vit_mask.squeeze(0).bool()
            token_num = int(diff_loss_mask.sum().detach().cpu().item())
            target_vit_embeds[:, diff_loss_mask, :] = mask_token.expand(1, token_num, -1)
            input_embeds[:, target_vit_embed_mask, :] = target_vit_embeds

        else: # tgt_vit_mask is None
            all_vit_token_num = sum(target_vit_embed_mask).detach().cpu().numpy()
            diff_loss_mask = torch.zeros(all_vit_token_num).to(target_vit_embeds.device)

        return dict(
            input_embeds=input_embeds, 
            diff_loss_mask=diff_loss_mask,
            target_vit_embeds=target_vit_embeds_gt
        )
    
    def feat_from_planner_to_renderer(
        self, 
        hidden_states, 
        tgt_vit_mask, 
        visual_output_mask, 
        inference=False
    ):
        pred_vit_embed_mask = visual_output_mask.squeeze(0)
        pred_vit_embeds = hidden_states[:, pred_vit_embed_mask, :].clone() # For calculate vit decoder loss
        txt_and_vit_token_mask = visual_output_mask.squeeze(0).logical_not()

        if not inference:
            all_idx = torch.nonzero(pred_vit_embed_mask, as_tuple=False).squeeze(-1)  # shape [N]
            cur_clip_mask = tgt_vit_mask.bool().logical_not() 
            valid_clip_idx = all_idx[cur_clip_mask]  
            pred_vit_embed_mask = torch.zeros(hidden_states.shape[1], dtype=torch.bool, device=hidden_states.device)
            pred_vit_embed_mask[valid_clip_idx] = True

        cond_embed_mask = (txt_and_vit_token_mask | pred_vit_embed_mask)
        diff_mllm_context_txt_mask = txt_and_vit_token_mask[cond_embed_mask]
        diff_mllm_context_vit_mask = pred_vit_embed_mask[cond_embed_mask]
        
        connector_param = next(self.connector.parameters())
        if connector_param.device != hidden_states.device or connector_param.dtype != hidden_states.dtype:
            self.connector.to(device=hidden_states.device, dtype=hidden_states.dtype)
        diff_mllm_contexts = hidden_states[:, cond_embed_mask, :]
        diff_mllm_contexts = self.connector.for_gen(diff_mllm_contexts)
        
        mllm_context_seqlens = []
        pred_vit_embed_seqlens = []
        pred_vit_embed_mask = visual_output_mask.squeeze(0)
        mllm_context_seqlens.append(int(cond_embed_mask.sum().item()))
        pred_vit_embed_seqlens.append(int(pred_vit_embed_mask.sum().item()))

        return dict(
            diff_mllm_contexts=diff_mllm_contexts,
            mllm_context_seqlens=mllm_context_seqlens,
            pred_vit_embeds=pred_vit_embeds,
            pred_vit_embed_seqlens=pred_vit_embed_seqlens,
            diff_mllm_context_txt_mask=diff_mllm_context_txt_mask,
            diff_mllm_context_vit_mask=diff_mllm_context_vit_mask,
        )

    def format_mllm_inputs_embeds(
        self,
        input_ids,
        visual_embeds,
        visual_input_mask,
        visual_output_mask,
    ):
        inputs_embeds = self.mllm.get_input_embeddings()(input_ids).to(dtype=torch.bfloat16)

        if visual_embeds is not None and len(visual_embeds) > 0:
            visual_mask = visual_input_mask | visual_output_mask
            n_visual_tokens = visual_mask.sum().long().item()
            n_visual_features = visual_embeds.shape[0]
            if n_visual_tokens != n_visual_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_visual_tokens}, features {n_visual_features}"
                )

            visual_mask = (
                visual_mask.unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            visual_embeds = visual_embeds.to(
                inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(
                visual_mask, visual_embeds)

        return inputs_embeds
