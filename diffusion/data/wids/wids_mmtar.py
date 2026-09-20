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
import collections
import fcntl
import io
import mmap
import os
import struct

TarHeader = collections.namedtuple(
    "TarHeader",
    [
        "name",
        "mode",
        "uid",
        "gid",
        "size",
        "mtime",
        "chksum",
        "typeflag",
        "linkname",
        "magic",
        "version",
        "uname",
        "gname",
        "devmajor",
        "devminor",
        "prefix",
    ],
)


def parse_tar_header(header_bytes):
    header = struct.unpack("!100s8s8s8s12s12s8s1s100s6s2s32s32s8s8s155s", header_bytes)
    return TarHeader(*header)


def next_header(offset, header):
    block_size = 512
    size = header.size.decode("utf-8").strip("\x00")
    if size == "":
        return -1
    size = int(size, 8)
    # compute the file size rounded up to the next block size if it is a partial block
    padded_file_size = (size + block_size - 1) // block_size * block_size
    return offset + block_size + padded_file_size


# TODO(ligeng): support gzip stream
class MMIndexedTar:
    def __init__(self, fname, index_file=None, verbose=True, cleanup_callback=None):
        self.verbose = verbose
        self.cleanup_callback = cleanup_callback
        if isinstance(fname, str):
            self.stream = open(fname, "rb")
            self.fname = fname
        elif isinstance(fname, io.IOBase):
            self.stream = fname
            self.fname = None
        self.mmapped_file = mmap.mmap(self.stream.fileno(), 0, access=mmap.ACCESS_READ)
        if cleanup_callback:
            cleanup_callback(fname, self.stream.fileno(), "start")
        self._build_index()

    def close(self, dispose=False):
        if self.cleanup_callback:
            self.cleanup_callback(self.fname, self.stream.fileno(), "end")
        self.mmapped_file.close()
        self.stream.close()

    def _build_index(self):
        self.by_name = {}
        self.by_index = []
        offset = 0
        while offset >= 0 and offset < len(self.mmapped_file):
            header = parse_tar_header(self.mmapped_file[offset : offset + 500])
            name = header.name.decode("utf-8").strip("\x00")
            typeflag = header.typeflag.decode("utf-8").strip("\x00")
            if name != "" and name != "././@PaxHeader" and typeflag in ["0", ""]:
                try:
                    size = int(header.size.decode("utf-8")[:-1], 8)
                except ValueError as exn:
                    print(header)
                    raise exn
                self.by_name[name] = offset
                self.by_index.append((name, offset, size))
            offset = next_header(offset, header)

    def names(self):
        return self.by_name.keys()

    def get_at_offset(self, offset):
        header = parse_tar_header(self.mmapped_file[offset : offset + 500])
        name = header.name.decode("utf-8").strip("\x00")
        start = offset + 512
        end = start + int(header.size.decode("utf-8")[:-1], 8)
        return name, self.mmapped_file[start:end]

    def get_at_index(self, index):
        name, offset, size = self.by_index[index]
        return self.get_at_offset(offset)

    def get_by_name(self, name):
        offset = self.by_name[name]
        return self.get_at_offset(offset)

    def __iter__(self):
        for name, offset, size in self.by_index:
            yield name, self.mmapped_file[offset + 512 : offset + 512 + size]

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.get_at_index(key)
        else:
            return self.get_by_name(key)

    def __len__(self):
        return len(self.by_index)

    def get_file(self, i):
        fname, data = self.get_at_index(i)
        return fname, io.BytesIO(data)


def keep_while_reading(fname, fd, phase, delay=0.0):
    """This is a possible cleanup callback for cleanup_callback of MIndexedTar.

    It assumes that as long as there are some readers for a file,
    more readers may be trying to open it.

    Note that on Linux, unlinking the file doesn't matter after
    it has been mmapped. The contents will only be deleted when
    all readers close the file. The unlinking merely makes the file
    unavailable to new readers, since the downloader checks first
    whether the file exists.
    """
    assert delay == 0.0, "delay not implemented"
    if fd < 0 or fname is None:
        return
    if phase == "start":
        fcntl.flock(fd, fcntl.LOCK_SH)
    elif phase == "end":
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.unlink(fname)
        except FileNotFoundError:
            # someone else deleted it already
            pass
        except BlockingIOError:
            # we couldn't get an exclusive lock, so someone else is still reading
            pass
    else:
        raise ValueError(f"Unknown phase {phase}")
