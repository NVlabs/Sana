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

# This file is copied from https://github.com/NVlabs/VILA/tree/main/llava/wids
import fcntl
import os
import shutil
import sys
import time
from collections import deque
from datetime import datetime
from urllib.parse import urlparse

recent_downloads = deque(maxlen=1000)

open_objects = {}
max_open_objects = 100


class ULockFile:
    """A simple locking class. We don't need any of the third
    party libraries since we rely on POSIX semantics for linking
    below anyway."""

    def __init__(self, path):
        self.lockfile_path = path
        self.lockfile = None

    def __enter__(self):
        self.lockfile = open(self.lockfile_path, "w")
        fcntl.flock(self.lockfile.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        fcntl.flock(self.lockfile.fileno(), fcntl.LOCK_UN)
        self.lockfile.close()
        self.lockfile = None
        try:
            os.unlink(self.lockfile_path)
        except FileNotFoundError:
            pass


def pipe_download(remote, local):
    """Perform a download for a pipe: url."""
    assert remote.startswith("pipe:")
    cmd = remote[5:]
    cmd = cmd.format(local=local)
    assert os.system(cmd) == 0, "Command failed: %s" % cmd


def copy_file(remote, local):
    remote = urlparse(remote)
    assert remote.scheme in ["file", ""]
    # use absolute path
    remote = os.path.abspath(remote.path)
    local = urlparse(local)
    assert local.scheme in ["file", ""]
    local = os.path.abspath(local.path)
    if remote == local:
        return
    # check if the local file exists
    shutil.copyfile(remote, local)


verbose_cmd = int(os.environ.get("WIDS_VERBOSE_CMD", "0"))


def vcmd(flag, verbose_flag=""):
    return verbose_flag if verbose_cmd else flag


default_cmds = {
    "posixpath": copy_file,
    "file": copy_file,
    "pipe": pipe_download,
    "http": "curl " + vcmd("-s") + " -L {url} -o {local}",
    "https": "curl " + vcmd("-s") + " -L {url} -o {local}",
    "ftp": "curl " + vcmd("-s") + " -L {url} -o {local}",
    "ftps": "curl " + vcmd("-s") + " -L {url} -o {local}",
    "gs": "gsutil " + vcmd("-q") + " cp {url} {local}",
    "s3": "aws s3 cp {url} {local}",
}


def download_file_no_log(remote, local, handlers=default_cmds):
    """Download a file from a remote url to a local path.
    The remote url can be a pipe: url, in which case the remainder of
    the url is treated as a command template that is executed to perform the download.
    """

    if remote.startswith("pipe:"):
        schema = "pipe"
    else:
        schema = urlparse(remote).scheme
    if schema is None or schema == "":
        schema = "posixpath"
    # get the handler
    handler = handlers.get(schema)
    if handler is None:
        raise ValueError("Unknown schema: %s" % schema)
    # call the handler
    if callable(handler):
        handler(remote, local)
    else:
        assert isinstance(handler, str)
        cmd = handler.format(url=remote, local=local)
        assert os.system(cmd) == 0, "Command failed: %s" % cmd
    return local


def download_file(remote, local, handlers=default_cmds, verbose=False):
    start = time.time()
    try:
        return download_file_no_log(remote, local, handlers=handlers)
    finally:
        recent_downloads.append((remote, local, time.time(), time.time() - start))
        if verbose:
            print(
                "downloaded",
                remote,
                "to",
                local,
                "in",
                time.time() - start,
                "seconds",
                file=sys.stderr,
            )


def download_and_open(remote, local, mode="rb", handlers=default_cmds, verbose=False):
    with ULockFile(local + ".lock"):
        if os.path.exists(remote):
            # print("enter1", remote, local, mode)
            result = open(remote, mode)
        else:
            # print("enter2", remote, local, mode)
            if not os.path.exists(local):
                if verbose:
                    print("downloading", remote, "to", local, file=sys.stderr)
                download_file(remote, local, handlers=handlers)
            else:
                if verbose:
                    print("using cached", local, file=sys.stderr)
            result = open(local, mode)

        # input()

        if open_objects is not None:
            for k, v in list(open_objects.items()):
                if v.closed:
                    del open_objects[k]
            if len(open_objects) > max_open_objects:
                raise RuntimeError("Too many open objects")
            current_time = datetime.now().strftime("%Y%m%d%H%M%S")
            key = tuple(str(x) for x in [remote, local, mode, current_time])
            open_objects[key] = result
        return result
