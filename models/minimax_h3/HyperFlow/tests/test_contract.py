import pytest

from infer import parse_args
from sol_hyperflow.config import DURATION_FRAMES, Request, validate_world_size
from sol_hyperflow.engine import HyperFlowInference


@pytest.mark.parametrize("task", ["t2v", "i2v", "ref2va"])
@pytest.mark.parametrize("duration", [5, 10, 15])
def test_native_frame_and_step_contract(task, duration):
    image = object() if task == "i2v" else None
    refs = (object(), object()) if task == "ref2va" else ()
    value = Request("A continuous view.", task, duration, 7, "sol_bsa", image, refs).payload()
    assert value["num_frames"] == DURATION_FRAMES[duration]
    assert value["num_frames"] % 17 == 5
    assert value["steps"] == 8 and value["performance_mode"] == "speed"
    assert (value["height"], value["width"]) == (768, 1344)
    if refs:
        assert value["task"] == "ref2v" and value["reference_images"] == list(refs)


@pytest.mark.parametrize("changes", [dict(prompt=""), dict(task="fl2va"), dict(duration=8),
    dict(seed=-1), dict(seed=True), dict(attention_backend="sol"), dict(task="i2v"),
    dict(task="ref2va"), dict(references=(object(),)), dict(image=object()),
    dict(task="ref2va", references=tuple(range(10)))])
def test_rejects_unsupported_requests(changes):
    values=dict(prompt="A continuous view."); values.update(changes)
    with pytest.raises(ValueError):
        Request(**values).payload()


def test_dense_is_an_explicit_attention_policy():
    assert Request("A view.", attention_backend="dense").payload()["performance_mode"] == "quality"


def test_validated_world_size_is_explicit():
    validate_world_size(8)
    for count in (1, 2, 4, 16):
        with pytest.raises(ValueError, match="nproc_per_node=8"):
            validate_world_size(count)


def test_cli_rejects_missing_reference_and_step_override():
    args=["--model","model","--adapter","weights","--task","ref2va","--prompt","test","--output","out.mp4"]
    with pytest.raises(SystemExit):parse_args(args)
    args[5]="t2v"
    with pytest.raises(SystemExit):parse_args(args+["--steps","4"])
    assert parse_args(args).lora_mode == "fused"


def test_transformer_partitions_cannot_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    (tmp_path/"transformer").mkdir();(tmp_path/"transformer/config.json").write_text("{}")
    (tmp_path/"transformer_ref").symlink_to(tmp_path/"transformer", target_is_directory=True)
    adapter=tmp_path/"adapter.safetensors";adapter.write_bytes(b"synthetic")
    with pytest.raises(ValueError, match="distinct upstream"):
        HyperFlowInference(tmp_path, adapter)
