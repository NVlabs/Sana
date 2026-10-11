"""Feed a prerecorded semantic example to an already warm HTTP worker."""
import argparse
import io
import json
from pathlib import Path
import threading
import time

import av
import requests
from prepare_example import ROOT, DEFAULT, requests as example_requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--example', default=DEFAULT)
    parser.add_argument('--media-root', default=str(ROOT))
    parser.add_argument('--server', default='http://127.0.0.1:8765')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--output', default=str(ROOT / 'outputs/replay'))
    args = parser.parse_args()
    row = next(r for r in example_requests() if r['name'] == args.example)
    source = Path(args.media_root) / row['reference_video_path']
    server = args.server.rstrip('/')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    response = requests.post(server + '/session', json={k: row[k] for k in ('fps', 'height', 'width')}, timeout=30)
    response.raise_for_status()
    session = response.json()['id']
    stop = threading.Event()
    errors = []
    count = [0]

    def receive():
        try:
            with requests.get(server + '/stream/' + session, stream=True, timeout=(30, 120)) as stream:
                stream.raise_for_status()
                while not stop.is_set():
                    line = stream.raw.readline()
                    if not line:
                        break
                    if not line.startswith(b'--render'):
                        continue
                    headers = {}
                    while True:
                        line = stream.raw.readline()
                        if not line or line == b'\r\n':
                            break
                        key, value = line.decode().split(':', 1)
                        headers[key.lower()] = value.strip()
                    size = int(headers['content-length'])
                    image = stream.raw.read(size)
                    if len(image) != size:
                        raise IOError('Truncated feedback frame')
                    (output / f'{session}_{count[0]:06d}.jpg').write_bytes(image)
                    count[0] += 1
        except Exception as exc:
            if not stop.is_set():
                errors.append(exc)

    viewer = threading.Thread(target=receive, daemon=True)
    viewer.start()
    sent = 0
    try:
        deadline = time.monotonic() + 1200
        while True:
            response = requests.get(server + '/status/' + session, timeout=30)
            response.raise_for_status()
            status = response.json()
            if status['error']:
                raise RuntimeError(status['error'])
            if status['ready']:
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Session preparation timed out')
            time.sleep(.5)
        with av.open(str(source)) as container:
            limit = args.max_frames or row['num_frames']
            for frame in container.decode(video=0):
                if sent >= limit:
                    break
                image = frame.to_image().resize((row['width'], row['height']), resample=0)
                payload = io.BytesIO()
                image.save(payload, format='PNG')
                deadline = time.monotonic() + 120
                while True:
                    if errors:
                        raise errors[0]
                    response = requests.post(f'{server}/frame/{session}/{sent}', data=payload.getvalue(),
                                             headers={'Content-Type': 'image/png'}, timeout=60)
                    if response.status_code != 429:
                        response.raise_for_status()
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError('Input backpressure timed out')
                    time.sleep(.05)
                sent += 1
        # Drain completed chunks; an incomplete scheduling tail is discarded on close.
        last_count = count[0]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if errors:
                raise errors[0]
            if count[0] != last_count:
                last_count = count[0]
                deadline = time.monotonic() + 10
            time.sleep(.2)
        if not count[0]:
            raise RuntimeError('No rendered feedback received')
        print(json.dumps({'session': session, 'example': row['name'], 'input_frames': sent,
                          'saved_rgb_frames': count[0], 'output': str(output.resolve())}))
    finally:
        stop.set()
        requests.post(server + '/close/' + session, data=b'{}', timeout=30)
        viewer.join(timeout=2)


if __name__ == '__main__':
    main()
