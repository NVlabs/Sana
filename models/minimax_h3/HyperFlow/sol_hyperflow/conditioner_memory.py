"""Drop only Qwen layers that are downstream of H3's selected hidden state."""
import gc
import torch
from PIL import Image
from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds


@torch.inference_mode()
def trim_and_verify(pipe, device):
    encoder = pipe.text_encoder.eval().to(device)
    language = encoder.model.language_model
    selected = pipe.text_encoder_layer
    original = len(language.layers)
    keep = selected + 1
    assert original >= keep and selected == 50
    processor = pipe.processor
    inputs = [processor(text='A fox walks through a snowy pine forest.', return_tensors='pt')]
    inputs.append(processor(
        text='<|vision_start|><|image_pad|><|vision_end|>Describe this portrait.',
        images=Image.new('RGB',(336,336),(96,112,128)), return_tensors='pt'))
    def embed(batch):
        vision = {k:v for k,v in batch.items() if k not in {'input_ids','attention_mask','mm_token_type_ids'}}
        return get_qwen3vl_prompt_embeds(encoder,processor,batch['input_ids'][0].tolist(),
            vision_inputs=vision,text_encoder_layer=selected,device=device,dtype=encoder.dtype).cpu()
    expected = [embed(batch) for batch in inputs]
    before = sum(p.numel()*p.element_size() for p in encoder.parameters())
    # Keep one layer beyond hidden_states[50] so that index 50 remains PRE-norm.
    language.layers = torch.nn.ModuleList(list(language.layers[:keep]))
    language.config.num_hidden_layers = keep
    encoder.config.text_config.num_hidden_layers = keep
    # Upstream H3 calls encoder.model and never evaluates the vocabulary head.
    encoder.lm_head = torch.nn.Identity()
    after = sum(p.numel()*p.element_size() for p in encoder.parameters())
    for reference,batch in zip(expected,inputs):
        actual = embed(batch)
        assert torch.equal(reference,actual), 'Conditioner trimming changed H3 hidden_states[50]'
    del expected, inputs
    gc.collect()
    torch.cuda.empty_cache()
    return dict(original_layers=original,retained_layers=keep,selected_hidden_state=selected,
                unused_lm_head_removed=True,freed_bytes=before-after,
                text_and_image_embeddings_bitwise_equal=True)
