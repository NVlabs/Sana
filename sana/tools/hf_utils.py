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

import os
import os.path as osp

from huggingface_hub import hf_hub_download, snapshot_download

HF_URI_SCHEME = "hf://"


def resolve_hf_path(path):
    """Resolve a possibly ``hf://``-prefixed path to a local filesystem path.

    Accepts either:

    * a local path (returned unchanged if it exists), or
    * ``hf://<owner>/<repo>[/<subpath>]`` — snapshot-downloads the (sub)tree
      from the Hub and returns the absolute path to the file or directory.

    Downloads are scoped via ``allow_patterns`` so only the requested subtree
    is materialised; unrelated artefacts in the same repo are skipped.
    """
    if not isinstance(path, str) or not path:
        return path
    if osp.exists(path):
        return path
    if not path.startswith(HF_URI_SCHEME):
        return path

    parts = path[len(HF_URI_SCHEME) :].split("/", 2)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid HF path {path!r}; expected hf://<owner>/<repo>[/<subpath>].")
    repo_id = f"{parts[0]}/{parts[1]}"
    subpath = parts[2] if len(parts) > 2 else ""

    allow_patterns = None
    if subpath:
        # Cover both ``subpath`` being a file and being a directory prefix.
        allow_patterns = [subpath, f"{subpath}/*", f"{subpath}/**"]

    local_root = snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns)
    return os.path.join(local_root, subpath) if subpath else local_root


def hf_download_or_fpath(path):
    """Backwards-compatible alias for :func:`resolve_hf_path`."""
    return resolve_hf_path(path)


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
