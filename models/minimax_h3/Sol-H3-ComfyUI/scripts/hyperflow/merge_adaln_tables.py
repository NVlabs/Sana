"""Merge per-task AdaLN tables (built one pipeline at a time to stay under the host RAM cap).
Usage: merge_adaln_tables.py OUT.pt IN.pt [IN.pt ...]
"""
import sys

import torch

out, parts = sys.argv[1], [torch.load(p, map_location="cpu", weights_only=True) for p in sys.argv[2:]]
keys, rows = [], [[] for _ in parts[0]["tables"]]
for part in parts:
    start = 0
    for key in part["keys"]:
        n = len(key)
        if key not in keys:
            keys.append(key)
            for b, table in enumerate(part["tables"]):
                rows[b].append(table[start:start + n])
        start += n
torch.save({"keys": keys, "tables": [torch.cat(r) for r in rows]}, out)
print("MERGED", len(keys), "forwards", int(sum(len(k) for k in keys)), "rows")
