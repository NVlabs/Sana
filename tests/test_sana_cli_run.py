# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import pytest

from sana.cli import run as sana_run


@pytest.mark.parametrize("conda_available", [False, True])
def test_hf_token_is_forwarded_without_cli_login(monkeypatch, conda_available):
    token = "hf_test_token"
    captured = {}

    monkeypatch.setenv("SANA_SLURM_ACCOUNT", "test-account")
    monkeypatch.setenv("SANA_SLURM_PARTITION", "test-partition")
    monkeypatch.setenv("CONDA_ENV_NAME", "test-env")
    monkeypatch.setenv("HF_TOKEN", token)
    monkeypatch.setattr(
        sys,
        "argv",
        ["sana-run", "--job-name", "test", "--mode", "ci", "echo", "ok"],
    )
    if conda_available:
        monkeypatch.setattr(sana_run.shutil, "which", lambda _: "/opt/conda/bin/conda")
        monkeypatch.setattr(sana_run.os.path, "exists", lambda _: True)
    else:
        monkeypatch.setattr(sana_run.shutil, "which", lambda _: None)

    def fake_run(command, *, env, shell):
        captured.update(command=command, env=env, shell=shell)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(sana_run.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exc_info:
        sana_run.main()

    assert exc_info.value.code == 0
    assert captured["env"]["HF_TOKEN"] == token
    assert captured["shell"] is True
    assert "hf auth login" not in captured["command"]
    assert token not in captured["command"]
