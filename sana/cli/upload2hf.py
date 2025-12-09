# Copyright 2024 NVIDIA CORPORATION & AFFILIATES (authored by @Lyken17)
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

import argparse
import os.path as osp
import re
import time
from hashlib import sha1, sha256

from huggingface_hub import HfApi
from huggingface_hub.hf_api import CommitOperationAdd
from termcolor import colored

MAX_UPLOAD_FILES_PER_COMMIT = 64
MAX_UPLOAD_SIZE_PER_COMMIT = 64 * 1024 * 1024 * 1024  # 64 GiB
MAX_UPLOAD_SIZE_PER_SINGLE_FILE = 45 * 1024 * 1024 * 1024  # 45 GiB


def compute_git_hash(filename):
    with open(filename, "rb") as f:
        data = f.read()
    s = "blob %u\0" % len(data)
    h = sha1()
    h.update(s.encode("utf-8"))
    h.update(data)
    return h.hexdigest()


def compute_lfs_hash(filename):
    with open(filename, "rb") as f:
        data = f.read()
    h = sha256()
    h.update(data)
    return h.hexdigest()


def main():
    import os

    os.environ["CURL_CA_BUNDLE"] = ""

    parser = argparse.ArgumentParser()
    parser.add_argument("local_folder", type=str)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--repo-type", type=str, choices=["model", "dataset"])
    parser.add_argument("--repo-org", type=str, default="Efficient-Large-Model")
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--root-dir", type=str, default=None)
    parser.add_argument("--token", type=str, default=None)

    parser.add_argument("-e", "--exclude", action="append", default=[r"checkpoint-[\d]*/.*", ".git/.*", "wandb/.*"])
    parser.add_argument("--fast-check", action="store_true")
    parser.add_argument("--sleep-on-error", action="store_true")

    args = parser.parse_args()

    if args.token is None:
        api = HfApi()
    else:
        print("initing using token from cmd args.")
        api = HfApi(token=args.token)

    repo_type = args.repo_type

    local_folder = args.local_folder

    if args.repo_id is not None:
        repo = args.repo_id
    else:
        # remove last /
        if local_folder[-1] == "/":
            local_folder = local_folder[:-1]

        if args.model_name is None:
            model_name = osp.basename(local_folder).replace("+", "-")
        else:
            model_name = args.model_name
        repo = osp.join(args.repo_org, model_name)

    local_folder = os.path.expanduser(local_folder)
    root_dir = local_folder if args.root_dir is None else args.root_dir
    print(f"uploading {local_folder} to {repo}")
    if not api.repo_exists(repo, repo_type=repo_type):
        api.create_repo(
            repo_id=repo,
            private=True,
            repo_type=repo_type,
        )

    BASE_URL = "https://hf.co"
    if args.repo_type == "dataset":
        BASE_URL = "https://hf.co/datasets"
    print(colored(f"Uploading {osp.join(BASE_URL, repo)}", "green"))

    ops = []
    commit_description = ""
    commit_size = 0
    for root, dirs, files in os.walk(local_folder, topdown=True):
        dirs.sort()
        for name in files:
            fpath = osp.join(root, name)
            rpath = osp.relpath(fpath, osp.abspath(root_dir))

            exclude_flag = False
            matched_pattern = None
            for pattern in args.exclude:
                if re.search(pattern, rpath):
                    exclude_flag = True
                    matched_pattern = pattern
            if exclude_flag:
                print(colored(f"""[regex filter-out][{matched_pattern}]: {rpath}, skipping""", "yellow"))
                continue

            if osp.getsize(fpath) > MAX_UPLOAD_SIZE_PER_SINGLE_FILE:
                print(
                    colored(
                        f"Huggingface only supports filesize less than {MAX_UPLOAD_SIZE_PER_SINGLE_FILE}, skipping",
                        "red",
                    )
                )
                continue

            if api.file_exists(repo_id=repo, filename=rpath, repo_type=repo_type):
                if args.fast_check:
                    print(
                        colored(
                            f"Already uploaded {rpath}, fast check pass, skipping",
                            "green",
                        )
                    )
                    continue
                else:
                    hf_meta = api.get_paths_info(repo_id=repo, paths=rpath, repo_type=repo_type)[0]

                    if hf_meta.lfs is not None:
                        # hash_type = "lfs-sha256"
                        hf_hash = hf_meta.lfs["sha256"]
                        git_hash = compute_lfs_hash(fpath)
                    else:
                        # hash_type = "git-sha1"
                        hf_hash = hf_meta.blob_id
                        git_hash = compute_git_hash(fpath)

                    if hf_hash == git_hash:
                        print(
                            colored(
                                f"Already uploaded {rpath}, hash check pass, skipping",
                                "green",
                            )
                        )
                        continue
                    else:
                        print(
                            colored(
                                f"{rpath} is not same as local version, re-uploading...",
                                "red",
                            )
                        )

            operation = CommitOperationAdd(
                path_or_fileobj=fpath,
                path_in_repo=rpath,
            )
            print(colored(f"Commiting {rpath}", "green"))
            ops.append(operation)
            commit_size += operation.upload_info.size
            commit_description += f"Upload {rpath}\n"
            if len(ops) <= MAX_UPLOAD_FILES_PER_COMMIT and commit_size <= MAX_UPLOAD_SIZE_PER_COMMIT:
                continue

            commit_message = "Upload files with vila-upload."
            result = None
            while result is None:
                try:
                    commit_info = api.create_commit(
                        repo_id=repo,
                        repo_type=repo_type,
                        operations=ops,
                        commit_message=commit_message,
                        commit_description=commit_description,
                    )
                except RuntimeError as e:
                    print(e)
                    if not args.sleep_on_error:
                        raise e
                    else:
                        print("sleeping for one hour then re-try")
                        time.sleep(1800)
                        continue
                result = "success"

            commit_description = ""
            ops = []
            commit_size = 0

            print(colored(f"Finish {commit_info}", "yellow"))

    # upload residuals
    commit_message = "Upload files with `sana-upload`."
    commit_info = api.create_commit(
        repo_id=repo,
        repo_type=repo_type,
        operations=ops,
        commit_message=commit_message,
        commit_description=commit_description,
    )


if __name__ == "__main__":
    main()
