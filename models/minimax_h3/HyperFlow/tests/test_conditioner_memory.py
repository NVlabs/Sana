from types import SimpleNamespace

import pytest
import torch

from sol_hyperflow import conditioner_memory as memory


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model=torch.nn.Module();self.model.language_model=torch.nn.Module()
        language=self.model.language_model
        language.layers=torch.nn.ModuleList([torch.nn.Linear(2,2) for _ in range(64)])
        with torch.no_grad():
            for layer in language.layers:
                layer.weight.copy_(torch.eye(2));layer.bias.fill_(.01)
        language.config=SimpleNamespace(num_hidden_layers=64)
        self.config=SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=64))
        self.lm_head=torch.nn.Linear(2,11)

    @property
    def dtype(self):return next(self.parameters()).dtype


def test_keeps_layer_50_pre_norm_for_text_and_image(monkeypatch):
    encoder=Encoder()
    def processor(**kwargs):
        data={"input_ids":torch.tensor([[1,2,3]])}
        if "images" in kwargs:data["pixel_values"]=torch.tensor([.25])
        return data
    observed=[]
    def embed(model,processor,ids,vision_inputs,text_encoder_layer,device,dtype):
        x=torch.tensor(ids,dtype=dtype)[:,None].repeat(1,2)
        if vision_inputs:x=x+vision_inputs["pixel_values"][0]
        states=[x]
        for layer in model.model.language_model.layers:
            x=layer(x);states.append(x)
        states[-1]=states[-1]+1000  # Final normalization must NOT reach hidden_states[50].
        value=states[text_encoder_layer];observed.append(value.clone());return value
    monkeypatch.setattr(memory,"get_qwen3vl_prompt_embeds",embed)
    pipe=SimpleNamespace(text_encoder=encoder,processor=processor,text_encoder_layer=50)
    result=memory.trim_and_verify(pipe,torch.device("cpu"))
    assert len(encoder.model.language_model.layers)==51
    assert encoder.config.text_config.num_hidden_layers==51
    assert encoder.model.language_model.config.num_hidden_layers==51
    assert isinstance(encoder.lm_head,torch.nn.Identity)
    assert torch.equal(observed[0],observed[2]) and torch.equal(observed[1],observed[3])
    assert not torch.equal(observed[0],observed[1])
    assert result["text_and_image_embeddings_bitwise_equal"] and result["freed_bytes"]>0


def test_unexpected_selected_layer_fails_before_pruning():
    encoder=Encoder();pipe=SimpleNamespace(text_encoder=encoder,text_encoder_layer=49)
    with pytest.raises(AssertionError):memory.trim_and_verify(pipe,torch.device("cpu"))
    assert len(encoder.model.language_model.layers)==64
