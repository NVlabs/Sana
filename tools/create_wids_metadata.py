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

import json
import os
import sys
import tarfile
from glob import glob

from tqdm.contrib.concurrent import process_map

"""
python tools/create_wids_metadata.py /path/to/tar/dir > /path/to/wids-meta.json
"""

d = sys.argv[1]


def process(t):
    d = {}
    with tarfile.open(t, "r") as tar:
        for f in tar:
            n, e = os.path.splitext(f.name)
            if e == ".jpg" or e == ".jpeg" or e == ".png" or e == ".json" or e == ".npy":
                if n in d:
                    d[n] = 1
                else:
                    d[n] = 0
    s = os.path.getsize(t)
    i = sum(d.values())
    t = os.path.basename(t)
    return {"url": t, "nsamples": i, "filesize": s}


print(
    json.dumps(
        {
            "name": "sana-dev",
            "__kind__": "SANA-WebDataset",
            "wids_version": 1,
            "shardlist": sorted(
                process_map(process, glob(f"{d}/*.tar"), chunksize=1, max_workers=os.cpu_count()),
                key=lambda x: x["url"],
            ),
        },
        indent=4,
    ),
    end="",
)
