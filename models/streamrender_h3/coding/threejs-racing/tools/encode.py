#!/usr/bin/env python3
"""Package a completed capture as RGB-lossless H.264 for H3's video reader."""
import argparse
import json
import subprocess
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('capture', type=Path)
p.add_argument('--ffmpeg', default='ffmpeg')
args = p.parse_args()
m = json.loads((args.capture / 'manifest.json').read_text())
if not m.get('complete'):
    raise SystemExit('Capture is incomplete')
if not m.get('h3CompatibleFrameCount'):
    raise SystemExit('Current H3 expects 5 + 17*k frames')
subprocess.run([args.ffmpeg, '-y', '-framerate', str(m['fps']), '-start_number', '0',
                '-i', str(args.capture / 'semantic/%06d.png'), '-frames:v', str(m['frames']),
                '-c:v', 'libx264rgb', '-crf', '0', '-preset', 'fast', '-pix_fmt', 'rgb24',
                str(args.capture / 'semantic.mp4')], check=True)
print(args.capture / 'semantic.mp4')
