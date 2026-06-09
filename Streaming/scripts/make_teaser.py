#!/usr/bin/env python3
from pathlib import Path
import os
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(os.environ.get("SANA_STREAMING_SOURCE_ROOT", "/Users/yyzhao/Downloads/sana-streaming-example"))
WORK = ROOT / "assets" / "teaser-work"
OUT_DIR = ROOT / "assets" / "teaser"
OUT = OUT_DIR / "sana-streaming-teaser.mp4"

FPS = 24
WIDTH = 1280
HEIGHT = 704
TITLE_DURATION = 3
EXAMPLE_DURATION = 10
EXAMPLE_SOURCE_DURATION = 6
EXAMPLE_WIPE_DURATION = 1
GRID_SOURCE_DURATION = 3
GRID_WIPE_DURATION = 1
GRID_DURATION = 8

TITLE_FRAME = ROOT / "assets" / "logos" / "teaser-sana-streaming-five-institutions.png"

EXAMPLES = [
    {
        "video": SOURCE_ROOT / "ready-to-use" / "long-select" / "005356. Replace the subject's white button-up shirt with a24dfe94d60.mp4",
        "prompt": "Replace the shirt with a navy silk blouse.",
    },
    {
        "video": SOURCE_ROOT / "ready-to-use" / "long-select" / "005810. Replace the background with a cinematic, rain-stre2eccb1022c.mp4",
        "prompt": "Replace the background with a rainy city window.",
    },
    {
        "video": SOURCE_ROOT / "ready-to-use" / "long-select" / "001657. Re-imagine the entire scene as an ancient fresco p0f677feae6.mp4",
        "prompt": "Re-imagine the scene as an ancient fresco.",
    },
]

SHORT_DIR = SOURCE_ROOT / "ready-to-use" / "selected_short_videos"

GRID_PAIRS = [
    {
        "source": SHORT_DIR / "short-local_0111_local_change_Replace_the_green_mu__original.mp4",
        "edit": SHORT_DIR / "short-local_0111_local_change_Replace_the_green_mu__edited.mp4",
    },
    {
        "source": SHORT_DIR / "short-local_0072_local_change_Replace_the_middle_a__original.mp4",
        "edit": SHORT_DIR / "short-local_0072_local_change_Replace_the_middle_a__edited.mp4",
    },
    {
        "source": SHORT_DIR / "short-local_0228_local_remove_Remove_the_woman_wit__original.mp4",
        "edit": SHORT_DIR / "short-local_0228_local_remove_Remove_the_woman_wit__edited.mp4",
    },
    {
        "source": SHORT_DIR / "short-bg_0131_background_change_Transform_the_backgr__original.mp4",
        "edit": SHORT_DIR / "short-bg_0131_background_change_Transform_the_backgr__edited.mp4",
    },
    {
        "source": SHORT_DIR / "short-bg_0142_background_change_Replace_the_backgrou__original.mp4",
        "edit": SHORT_DIR / "short-bg_0142_background_change_Replace_the_backgrou__edited.mp4",
    },
    {
        "source": SHORT_DIR / "short-bg_0161_background_change_Create_a_dynamic_cel__original.mp4",
        "edit": SHORT_DIR / "short-bg_0161_background_change_Create_a_dynamic_cel__edited.mp4",
    },
    {
        "source": SHORT_DIR / "short-style_0005_global_style_Apply_the_Aesthetic__original.mp4",
        "edit": SHORT_DIR / "short-style_0005_global_style_Apply_the_Aesthetic__edited.mp4",
    },
    {
        "source": SHORT_DIR / "short-style_0047_global_style_Apply_the_Watercolor__original.mp4",
        "edit": SHORT_DIR / "short-style_0047_global_style_Apply_the_Watercolor__edited.mp4",
    },
    {
        "source": SHORT_DIR / "short-style_0048_global_style_Apply_the_Chinese_In__original.mp4",
        "edit": SHORT_DIR / "short-style_0048_global_style_Apply_the_Chinese_In__edited.mp4",
    },
]


def run(cmd):
    print(" ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def ass_time(seconds):
    centiseconds = round(seconds * 100)
    h = centiseconds // 360000
    centiseconds %= 360000
    m = centiseconds // 6000
    centiseconds %= 6000
    s = centiseconds // 100
    cs = centiseconds % 100
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ass_escape(text):
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def write_search_ass(path, prompt):
    start = 1.0
    end = EXAMPLE_SOURCE_DURATION
    type_duration = 2.3
    step = type_duration / len(prompt)
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {WIDTH}",
        f"PlayResY: {HEIGHT}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Search,Helvetica,29,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,218,0,62,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for i in range(1, len(prompt) + 1):
        t0 = start + (i - 1) * step
        t1 = start + i * step
        text = ass_escape(prompt[:i] + "|")
        lines.append(f"Dialogue: 0,{ass_time(t0)},{ass_time(t1)},Search,,0,0,0,,{text}")

    lines.append(f"Dialogue: 0,{ass_time(start + type_duration)},{ass_time(end)},Search,,0,0,0,,{ass_escape(prompt)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_title_clip(path):
    run([
        "ffmpeg", "-y",
        "-loop", "1",
        "-framerate", str(FPS),
        "-t", str(TITLE_DURATION),
        "-i", str(TITLE_FRAME),
        "-vf", f"scale={WIDTH}:{HEIGHT},format=yuv420p",
        "-an",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-r", str(FPS),
        str(path),
    ])


def make_example_clip(index, item, path):
    ass_path = WORK / f"example_{index}.ass"
    write_search_ass(ass_path, item["prompt"])
    ass_filter = str(ass_path).replace("\\", "\\\\").replace(":", "\\:")
    wipe_end = EXAMPLE_SOURCE_DURATION + EXAMPLE_WIPE_DURATION
    filters = (
        f"[0:v]trim=0:{EXAMPLE_DURATION},setpts=PTS-STARTPTS,crop={WIDTH}:{HEIGHT}:0:0,fps={FPS},format=yuv420p[src];"
        f"[0:v]trim=0:{EXAMPLE_DURATION},setpts=PTS-STARTPTS,crop={WIDTH}:{HEIGHT}:0:{HEIGHT},fps={FPS},format=yuv420p[edit];"
        f"[src][edit]blend=all_expr='if(lt(T,{EXAMPLE_SOURCE_DURATION}),A,if(gte(T,{wipe_end}),B,if(gte(X,(T-{EXAMPLE_SOURCE_DURATION})/{EXAMPLE_WIPE_DURATION}*W),A,B)))',"
        f"drawbox=x='(t-{EXAMPLE_SOURCE_DURATION})/{EXAMPLE_WIPE_DURATION}*1280':y=0:w=3:h=704:color=white@0.9:t=fill:enable='between(t,{EXAMPLE_SOURCE_DURATION},{wipe_end})',"
        f"trim=0:{EXAMPLE_DURATION},setpts=PTS-STARTPTS[base];"
        "[base]"
        f"drawbox=x=160:y=42:w=960:h=74:color=black@0.72:t=fill:enable='between(t,0.5,{EXAMPLE_SOURCE_DURATION})',"
        f"drawbox=x=160:y=42:w=960:h=74:color=white@0.34:t=2:enable='between(t,0.5,{EXAMPLE_SOURCE_DURATION})',"
        f"drawtext=font=Helvetica:text='Edit instruction':x=185:y=21:fontsize=20:fontcolor=white@0.58:enable='between(t,0.5,{EXAMPLE_SOURCE_DURATION})',"
        f"subtitles='{ass_filter}'[v]"
    )
    run([
        "ffmpeg", "-y",
        "-i", str(item["video"]),
        "-filter_complex", filters,
        "-map", "[v]",
        "-an",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-r", str(FPS),
        "-pix_fmt", "yuv420p",
        str(path),
    ])


def make_grid_tile(index, item, path):
    tile_w = 404
    tile_h = 212
    wipe_end = GRID_SOURCE_DURATION + GRID_WIPE_DURATION
    filters = (
        f"[0:v]trim=0:{GRID_DURATION},setpts=PTS-STARTPTS,fps={FPS},"
        f"scale={tile_w}:{tile_h}:force_original_aspect_ratio=increase,crop={tile_w}:{tile_h},format=yuv420p[src];"
        f"[1:v]trim=0:{GRID_DURATION},setpts=PTS-STARTPTS,fps={FPS},"
        f"scale={tile_w}:{tile_h}:force_original_aspect_ratio=increase,crop={tile_w}:{tile_h},format=yuv420p[edit];"
        f"[src][edit]blend=all_expr='if(lt(T,{GRID_SOURCE_DURATION}),A,if(gte(T,{wipe_end}),B,if(gte(X,(T-{GRID_SOURCE_DURATION})/{GRID_WIPE_DURATION}*W),A,B)))',"
        f"drawbox=x='(t-{GRID_SOURCE_DURATION})/{GRID_WIPE_DURATION}*{tile_w}':y=0:w=2:h={tile_h}:color=white@0.9:t=fill:enable='between(t,{GRID_SOURCE_DURATION},{wipe_end})',"
        f"trim=0:{GRID_DURATION},setpts=PTS-STARTPTS,format=yuv420p[v]"
    )
    run([
        "ffmpeg", "-y",
        "-stream_loop", "-1",
        "-i", str(item["source"]),
        "-stream_loop", "-1",
        "-i", str(item["edit"]),
        "-filter_complex", filters,
        "-map", "[v]",
        "-an",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-r", str(FPS),
        "-pix_fmt", "yuv420p",
        str(path),
    ])


def make_grid_clip(tile_paths, path):
    pad = 24
    gap = 10
    tile_w = 404
    tile_h = 212
    inputs = []
    for tile in tile_paths:
        inputs.extend(["-i", str(tile)])

    overlays = [f"color=c=black:s={WIDTH}x{HEIGHT}:d={GRID_DURATION}:r={FPS}[base]"]
    last = "base"
    for i in range(9):
        col = i % 3
        row = i // 3
        x = pad + col * (tile_w + gap)
        y = pad + row * (tile_h + gap)
        out = f"g{i}"
        overlays.append(f"[{last}][{i}:v]overlay=x={x}:y={y}:shortest=1[{out}]")
        last = out

    run([
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", ";".join(overlays),
        "-map", f"[{last}]",
        "-an",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-r", str(FPS),
        "-pix_fmt", "yuv420p",
        str(path),
    ])


def concat_clips(paths):
    concat_file = WORK / "concat.txt"
    concat_file.write_text("".join(f"file '{p}'\n" for p in paths), encoding="utf-8")
    run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-an",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-r", str(FPS),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(OUT),
    ])


def main():
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clips = []
    title = WORK / "00_title.mp4"
    make_title_clip(title)
    clips.append(title)

    for i, item in enumerate(EXAMPLES, start=1):
        clip = WORK / f"{i:02d}_example.mp4"
        make_example_clip(i, item, clip)
        clips.append(clip)

    tile_paths = []
    for i, item in enumerate(GRID_PAIRS):
        tile = WORK / f"grid_tile_{i:02d}.mp4"
        make_grid_tile(i, item, tile)
        tile_paths.append(tile)

    grid = WORK / "04_grid.mp4"
    make_grid_clip(tile_paths, grid)
    clips.append(grid)

    concat_clips(clips)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
