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
import io
import json
import os
import tempfile
from urllib.parse import urlparse, urlunparse

from .wids_dl import download_and_open


def urldir(url):
    """Return the directory part of a url."""
    parsed_url = urlparse(url)
    path = parsed_url.path
    directory = os.path.dirname(path)
    return parsed_url._replace(path=directory).geturl()


def urlmerge(base, url):
    """Merge a base URL and a relative URL.

    The function fills in any missing part of the url from the base,
    except for params, query, and fragment, which are taken only from the 'url'.
    For the pathname component, it merges the paths like os.path.join:
    an absolute path in 'url' overrides the base path, otherwise the paths are merged.

    Parameters:
    base (str): The base URL.
    url (str): The URL to merge with the base.

    Returns:
    str: The merged URL.
    """
    # Parse the base and the relative URL
    parsed_base = urlparse(base)
    parsed_url = urlparse(url)

    # Merge paths using os.path.join
    # If the url path is absolute, it overrides the base path
    if parsed_url.path.startswith("/"):
        merged_path = parsed_url.path
    else:
        merged_path = os.path.normpath(os.path.join(parsed_base.path, parsed_url.path))

    # Construct the merged URL
    merged_url = urlunparse(
        (
            parsed_url.scheme or parsed_base.scheme,
            parsed_url.netloc or parsed_base.netloc,
            merged_path,
            parsed_url.params,  # Use params from the url only
            parsed_url.query,  # Use query from the url only
            parsed_url.fragment,  # Use fragment from the url only
        )
    )

    return merged_url


def check_shards(l):
    """Check that a list of shards is well-formed.

    This checks that the list is a list of dictionaries, and that
    each dictionary has a "url" and a "nsamples" key.
    """
    assert isinstance(l, list)
    for shard in l:
        assert isinstance(shard, dict)
        assert "url" in shard
        assert "nsamples" in shard
    return l


def set_all(l, k, v):
    """Set a key to a value in a list of dictionaries."""
    if v is None:
        return
    for x in l:
        if k not in x:
            x[k] = v


def load_remote_dsdesc_raw(source):
    """Load a remote or local dataset description in JSON format."""
    if isinstance(source, str):
        with tempfile.TemporaryDirectory() as tmpdir:
            dlname = os.path.join(tmpdir, "dataset.json")
            with download_and_open(source, dlname) as f:
                dsdesc = json.load(f)
    elif isinstance(source, io.IOBase):
        dsdesc = json.load(source)
    else:
        # FIXME: use gopen
        import requests

        jsondata = requests.get(source).text
        dsdesc = json.loads(jsondata)
    return dsdesc


def rebase_shardlist(shardlist, base):
    """Rebase the URLs in a shardlist."""
    if base is None:
        return shardlist
    for shard in shardlist:
        shard["url"] = urlmerge(base, shard["url"])
    return shardlist


def resolve_dsdesc(dsdesc, *, options=None, base=None):
    """Resolve a dataset description.

    This rebases the shards as necessary and loads any remote references.

    Dataset descriptions are JSON files. They must have the following format;

    {
        "wids_version": 1,
        # optional immediate shardlist
        "shardlist": [
            {"url": "http://example.com/file.tar", "nsamples": 1000},
            ...
        ],
        # sub-datasets
        "datasets": [
            {"source_url": "http://example.com/dataset.json"},
            {"shardlist": [
                {"url": "http://example.com/file.tar", "nsamples": 1000},
                ...
            ]}
            ...
        ]
    }
    """
    if options is None:
        options = {}
    assert isinstance(dsdesc, dict)
    dsdesc = dict(dsdesc, **options)
    shardlist = rebase_shardlist(dsdesc.get("shardlist", []), base)
    assert shardlist is not None
    set_all(shardlist, "weight", dsdesc.get("weight"))
    set_all(shardlist, "name", dsdesc.get("name"))
    check_shards(shardlist)
    assert "wids_version" in dsdesc, "No wids_version in dataset description"
    assert dsdesc["wids_version"] == 1, "Unknown wids_version"
    for component in dsdesc.get("datasets", []):
        # we use the weight from the reference to the dataset,
        # regardless of remote loading
        weight = component.get("weight")
        # follow any source_url dsdescs through remote loading
        source_url = None
        if "source_url" in component:
            source_url = component["source_url"]
            component = load_remote_dsdesc_raw(source_url)
        assert "source_url" not in component, "double indirection in dataset description"
        assert "shardlist" in component, "no shardlist in dataset description"
        # if the component has a base, use it to rebase the shardlist
        # otherwise use the base from the source_url, if any
        subbase = component.get("base", urldir(source_url) if source_url else None)
        if subbase is not None:
            rebase_shardlist(component["shardlist"], subbase)
        l = check_shards(component["shardlist"])
        set_all(l, "weight", weight)
        set_all(l, "source_url", source_url)
        set_all(l, "dataset", component.get("name"))
        shardlist.extend(l)
    assert len(shardlist) > 0, "No shards found"
    dsdesc["shardlist"] = shardlist
    return dsdesc


def load_dsdesc_and_resolve(source, *, options=None, base=None):
    if options is None:
        options = {}
    dsdesc = load_remote_dsdesc_raw(source)
    return resolve_dsdesc(dsdesc, base=base, options=options)
