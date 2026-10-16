#!/usr/bin/env python3
"""Original, speech-free cinematic underscore for the Sol-H3 in ComfyUI promo (no samples).

Slow pads in D minor with a rising pulse that lands on the speed chart, then resolves.
Usage: music.py OUT.wav SECONDS [--hit SECONDS]
"""
import argparse
import wave

import numpy as np

RATE = 48000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("seconds", type=float)
    parser.add_argument("--hit", type=float, default=19.0, help="time the pulse peaks (speed chart)")
    args = parser.parse_args()
    rng = np.random.default_rng(20261014)
    n = round(args.seconds * RATE)
    t_all = np.arange(n) / RATE
    mix = np.zeros((n, 2))

    def add(sig, start, gain=1.0, pan=0.0):
        b = round(start * RATE)
        k = min(len(sig), n - b)
        if k <= 0:
            return
        a = (pan + 1) * np.pi / 4
        mix[b:b + k, 0] += sig[:k] * gain * np.cos(a)
        mix[b:b + k, 1] += sig[:k] * gain * np.sin(a)

    def hz(m):
        return 440 * 2 ** ((m - 69) / 12)

    def pad(midi, dur, attack=1.2, release=1.4):
        t = np.arange(round(dur * RATE)) / RATE
        f = hz(midi)
        w = sum(np.sin(2 * np.pi * f * (1 + d) * t + p) for d, p in ((-.003, 0), (0, 1.1), (.004, 2.3)))
        w += 0.25 * np.sin(4 * np.pi * f * t)
        env = np.minimum(t / attack, 1) * np.minimum(1, np.maximum(0, dur - t) / release)
        return w * env / 3

    # Dm9 - Bbmaj7 - F - C/E, four bars each at 84 BPM.
    beat = 60 / 84
    bar = 4 * beat
    chords = ((50, 57, 60, 64, 65), (46, 53, 57, 62, 65), (41, 53, 57, 60, 65), (40, 52, 55, 60, 64))
    for i in range(int(args.seconds / bar) + 1):
        start = i * bar
        for j, m in enumerate(chords[i % 4]):
            add(pad(m, bar + 1.2), start, 0.10 if j else 0.16, pan=(-.5, .5)[j % 2] * (j > 0))
    # Sub swell and pulse that build to the hit, then fall away.
    build = np.clip((t_all - (args.hit - 8)) / 8, 0, 1) * np.clip((args.hit + 6 - t_all) / 6, 0, 1)
    sub = np.sin(2 * np.pi * hz(26) * t_all) * build * 0.22
    mix += sub[:, None]
    for k in range(int(args.seconds / (beat / 2))):
        s = k * beat / 2
        g = float(np.clip((s - (args.hit - 8)) / 8, 0, 1) * np.clip((args.hit + 6 - s) / 6, 0, 1))
        if g <= 0:
            continue
        tt = np.arange(round(.12 * RATE)) / RATE
        tick = rng.standard_normal(len(tt)) * np.exp(-tt * 60) * 0.5 + np.sin(2 * np.pi * 70 * tt) * np.exp(-tt * 25)
        add(tick, s, 0.18 * g, pan=(-.3, .3)[k % 2])
    # Soft impact on the hit.
    tt = np.arange(round(3.5 * RATE)) / RATE
    boom = np.sin(2 * np.pi * (55 - 25 * np.minimum(tt / .4, 1)) * tt) * np.exp(-tt * 1.6)
    add(boom, args.hit, 0.55)
    # Gentle fade in/out and level.
    fade = np.minimum(t_all / 1.5, 1) * np.minimum(1, (args.seconds - t_all) / 2.5)
    mix *= fade[:, None]
    mix *= 0.5 / max(1e-9, np.max(np.abs(mix)))
    pcm = (np.clip(mix, -1, 1) * 32767).astype("<i2")
    with wave.open(args.out, "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(pcm.tobytes())


if __name__ == "__main__":
    main()
