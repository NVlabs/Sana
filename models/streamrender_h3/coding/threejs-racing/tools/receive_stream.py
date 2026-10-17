#!/usr/bin/env python3
"""Receive ordered PNG frames; reuse iter_frames() in a streaming H3 input adapter."""
import argparse
import json
from pathlib import Path
from urllib.request import urlopen


def iter_frames(url):
    # A slow consumer is disconnected by the server rather than silently losing frames.
    # Caller must reset H3 history when session changes or sequence continuity fails.
    previous = None
    with urlopen(url, timeout=120) as response:
        while True:
            boundary = response.readline()
            if not boundary:
                raise EOFError('Semantic stream disconnected; reset model history before reconnecting')
            if boundary.strip() != b'--semantic':
                continue
            headers = {}
            while True:
                line = response.readline().decode('ascii').strip()
                if not line:
                    break
                key, value = line.split(':', 1)
                headers[key.lower()] = value.strip()
            frame = int(headers['x-frame'])
            session = headers['x-session']
            if previous and session == previous[0] and frame != previous[1] + 1:
                raise ValueError('Non-contiguous frames; reset model history')
            png = response.read(int(headers['content-length']))
            if len(png) != int(headers['content-length']):
                raise EOFError('Truncated frame')
            previous = session, frame
            yield {'session': session, 'frame': frame, 'timestamp': float(headers['x-timestamp'])}, png


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:5186/api/stream')
    p.add_argument('--output', type=Path, default=Path('received'))
    p.add_argument('--frames', type=int, default=719)
    args = p.parse_args()
    for index, (meta, png) in enumerate(iter_frames(args.url)):
        directory = args.output / meta['session']
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f'{meta["frame"]:06d}.png').write_bytes(png)
        with (directory / 'frames.jsonl').open('a') as f:
            f.write(json.dumps(meta) + '\n')
        if index + 1 >= args.frames:
            break
