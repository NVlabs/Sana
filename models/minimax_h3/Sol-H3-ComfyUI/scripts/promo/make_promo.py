"""Compose the "Sol-H3 in ComfyUI" promo from generated clips + motion-graphics overlays.

Structure: hero shot with title, short montage with captions, the question, a per-clip latency chart
(request → MP4 for each pipeline and device), the Studio film (first frame to video, pipeline choice and the
gallery of showcase clips; rendered from the real web/studio.js by motion/render_film.mjs), and an end card over
the Studio's own gallery, blurred. Overlays are rendered by motion/render.mjs; every number shown comes from --timing (engine
reports). Music is synthesized by music.py.

Usage: make_promo.py --clips DIR --studio-films B.mp4 --timing timing.json --out promo.mp4
timing.json: {"rows": [{"kind": "base|lora|sol", "device": "rtx5090|spark", "name", "where", "s"}, ...],
              "speedup_from": NAME, "speedup_to": NAME}  (speedup = rows[from].s / rows[to].s, rounded)
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
FFMPEG = os.environ.get("PROMO_FFMPEG", "ffmpeg")
W, H, FPS = 1920, 1080, 30
ENC = ["-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-r", str(FPS)]
FIT = "scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d,setsar=1,fps=%d" % (W, H, W, H, FPS)


def ff(*args):
    subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


def duration(path):
    """Container duration in seconds, read from ffmpeg's banner (no ffprobe needed)."""
    info = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    h, m, sec = info.split("Duration: ", 1)[1].split(",", 1)[0].split(":")
    return round(int(h) * 3600 + int(m) * 60 + float(sec), 2)


def overlay_frames(scene, seconds, params, work):
    out = work / ("ov-%s-%d" % (scene, len(list(work.glob("ov-*")))))
    subprocess.run(["node", str(HERE / "motion/render.mjs"), scene, str(seconds), str(out), json.dumps(params)],
                   check=True, env=dict(os.environ, FPS=str(FPS)))
    return out


def segment(work, name, seconds, background, scene=None, params=None, start=0.0, blur=False, speed=1.0,
            darken=0.0, slots=(), slot_start=0.0, slot_speed=1.0, slot_fade_in=0.0):
    """One silent video segment: background clip (fitted, optionally blurred/sped/darkened) + overlay."""
    out = work / (name + ".mp4")
    inputs = ["-ss", "%.2f" % start, "-i", str(background)]
    chain = "[0:v]setpts=PTS/%.3f,%s" % (speed, FIT)
    if blur:
        chain += ",gblur=sigma=36"
    if darken:
        chain += ",eq=brightness=%.2f" % -darken
    chain += ",trim=duration=%.2f,setpts=PTS-STARTPTS[bg]" % seconds
    graph, last = [chain], "bg"
    for i, (clip, x, y, w, h) in enumerate(slots, start=1):
        inputs += ["-ss", "%.2f" % slot_start, "-i", str(clip)]
        graph.append("[%d:v]setpts=PTS/%.3f,scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d,fps=%d,"
                     "trim=duration=%.2f,setpts=PTS-STARTPTS,format=yuva420p,fade=t=in:st=%.2f:d=0.35:alpha=1[s%d]"
                     % (i, slot_speed, w, h, w, h, FPS, seconds, slot_fade_in, i))
        graph.append("[%s][s%d]overlay=%d:%d[v%d]" % (last, i, x, y, i))
        last = "v%d" % i
    if scene:
        frames = overlay_frames(scene, seconds, params or {}, work)
        inputs += ["-framerate", str(FPS), "-i", str(frames / "f%05d.png")]
        idx = len(slots) + 1
        graph.append("[%s][%d:v]overlay=0:0:format=auto[out]" % (last, idx))
        last = "out"
    ff(*inputs, "-filter_complex", ";".join(graph), "-map", "[%s]" % last, "-t", "%.2f" % seconds, "-an",
       *ENC, str(out))
    return out


END_S = 5.0


def gallery_tail(work, film, start):
    """The film's last END_S seconds with blur and darkening ramping in over the first 0.8 s."""
    out = work / "gallery-tail.mp4"
    graph = ("[0:v]trim=start=%.3f,setpts=PTS-STARTPTS,split[a][b];"
             "[b]gblur=sigma=28,eq=brightness=-0.30:saturation=0.8,format=yuva420p,fade=t=in:st=0:d=0.8:alpha=1[blur];"
             "[a][blur]overlay=0:0:format=auto[out]" % start)
    ff("-i", str(film), "-filter_complex", graph, "-map", "[out]", "-t", "%.2f" % END_S, "-an", *ENC, str(out))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--studio-films", type=Path, nargs="+", required=True,
                        help="full-frame Studio film parts from motion/render_film.mjs, in order")
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    timing = json.loads(args.timing.read_text())
    c = {p.stem: p for p in args.clips.glob("*.mp4")}
    work = Path(tempfile.mkdtemp(prefix="promo-"))
    segs = []
    # 1. Hero.
    segs.append(segment(work, "01-hero", 5.0, c["whale-sky-temple"], "title", start=0.0))
    # 2. Montage with captions.
    for i, (key, text) in enumerate([("astronaut-ring", "Ringed world"), ("glass-forest-stag", "Glass forest"),
                                      ("underwater-cathedral", "Sunken cathedral")]):
        segs.append(segment(work, "02-m%d" % i, 2.25, c[key], "caption", {"text": text, "sub": "First frame → video"},
                            start=0.8))
    # 3. Question.
    segs.append(segment(work, "03-question", 3.0, c["desert-caravan"], "question", start=0.5, darken=0.12))
    # 4. Latency chart over a blurred clip: one clip, request → MP4, per pipeline and device.
    rows = {r["name"] + " · " + r["where"]: r for r in timing["rows"]}
    ratio = rows[timing["speedup_from"]]["s"] / rows[timing["speedup_to"]]["s"]
    lat = {"title": "One clip, request to MP4.", "sub": "1344×768 · 121 frames with audio · model loaded",
           "scale": 200, "stagger": 0.35, "growS": 1.6, "rows": timing["rows"],
           "speedup": "%d×" % round(ratio), "speedupText": timing.get("speedup_text", "faster than the original model")}
    lat_s = 7.5
    segs.append(segment(work, "04-latency", lat_s, c["koi-lotus"], "lat", lat, blur=True, darken=0.3))
    # 5. Studio film: first frame, pipeline choice, then the gallery (already full-frame 1920x1080). Its last
    # END_S seconds (the whole Gallery wall) become the end card's background.
    studio_s = 0.0
    for i, film in enumerate(args.studio_films):
        seconds = duration(film) - (END_S if i == len(args.studio_films) - 1 else 0)
        segs.append(segment(work, "05-studio-%d" % i, seconds, film))
        studio_s += seconds
    # 6. End card over the Studio's own Gallery: the film continues, blurring and darkening in.
    segs.append(segment(work, "06-end", END_S, gallery_tail(work, args.studio_films[-1], studio_s), "end"))
    total = 5.0 + 3 * 2.25 + 3.0 + lat_s + studio_s + END_S
    music = work / "music.wav"
    # The pulse peaks as the speedup lands under the bars.
    hit = 5.0 + 3 * 2.25 + 3.0 + 0.45 + (len(timing["rows"]) - 1) * lat["stagger"] + 0.15 + lat["growS"] + 0.2
    subprocess.run(["python3", str(HERE / "music.py"), str(music), "%.2f" % (total + 0.5), "--hit", "%.2f" % hit],
                   check=True)
    inputs = sum((["-i", str(s)] for s in segs), [])
    graph = "".join("[%d:v]" % i for i in range(len(segs))) + "concat=n=%d:v=1:a=0[v]" % len(segs)
    ff(*inputs, "-i", str(music), "-filter_complex", graph, "-map", "[v]", "-map", "%d:a" % len(segs),
       *ENC, "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(args.out))
    print(json.dumps({"out": str(args.out), "segments": len(segs), "seconds": round(total, 2), "hit": hit}))
    if not args.keep:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
