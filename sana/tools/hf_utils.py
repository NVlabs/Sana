# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

import os.path as osp

from huggingface_hub import hf_hub_download, snapshot_download


def hf_download_or_fpath(path):
    if osp.exists(path):
        return path

    if path.startswith("hf://"):
        segs = path.replace("hf://", "").split("/")
        repo_id = "/".join(segs[:2])
        filename = "/".join(segs[2:])
        return hf_download_data(repo_id, filename, repo_type="model", download_full_repo=True)


def hf_download_data(
    repo_id="Efficient-Large-Model/Sana_1600M_1024px",
    filename="checkpoints/Sana_1600M_1024px.pth",
    cache_dir=None,
    repo_type="model",
    download_full_repo=False,
):
    """
    Download dummy data from a Hugging Face repository.

    Args:
    repo_id (str): The ID of the Hugging Face repository.
    filename (str): The name of the file to download.
    cache_dir (str, optional): The directory to cache the downloaded file.

    Returns:
    str: The path to the downloaded file.
    """
    try:
        if download_full_repo:
            # download full repos to fit dc-ae
            snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                repo_type=repo_type,
            )
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            repo_type=repo_type,
        )
        return file_path
    except Exception as e:
        print(f"Error downloading file: {e}")
        return None
