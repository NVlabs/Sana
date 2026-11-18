"""Request → MP4 timing through a running ComfyUI + Sol-H3 (same graph the Studio submits).

Submits SolH3Generate → SaveVideo over ComfyUI's /prompt API, waits for /history, and reports
  client_s : POST /prompt → history shows the saved video (what a caller waits for),
  e2e_s    : the engine's own request → MP4 clock (includes Qwen),
plus Qwen and generate/phase splits. The first request is the warmup (cold load); later ones are formal.
PROMPT_FILE may also be a JSON list of request objects {"prompt", "mode"?, "image"?, "resolution"?, "name"?} (image = a
filename in ComfyUI's input folder): each runs once after a warmup on the first.
Usage: e2e_comfy.py PROFILE PROMPT_FILE OUT.json [runs=2] [seed=42] [port=8190]
"""
import json
import sys
import time
import urllib.request

profile, prompt_file, out = sys.argv[1], sys.argv[2], sys.argv[3]
runs = int(sys.argv[4]) if len(sys.argv) > 4 else 2
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42
base = "http://127.0.0.1:%s" % (sys.argv[6] if len(sys.argv) > 6 else "8190")
text = open(prompt_file).read().strip()
# A plain text file is one prompt; a JSON list is one formal request per prompt after a warmup on the first.
prompts = json.loads(text) if text.startswith("[") else [text]
prompts = [p if isinstance(p, dict) else {"prompt": p} for p in prompts]
prompt = prompts[0]


def call(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


graph = {
    "1": {"class_type": "SolH3Generate", "inputs": {"profile": profile, "prompt": prompt, "seed": seed,
                                                    "keep_loaded": True, "mode": "text", "image": "", "backend": "local"}},
    "2": {"class_type": "SaveVideo", "inputs": {"video": ["1", 0], "filename_prefix": "SolH3/e2e",
                                                "format": "auto", "codec": "auto"}},
}
rows = []
schedule = [prompts[0]] + prompts * max(1, runs - 1) if len(prompts) > 1 else [prompt] * runs
for i, prompt in enumerate(schedule):
    graph["1"]["inputs"].update(prompt=prompt["prompt"], mode=prompt.get("mode", "text"),
                                image=prompt.get("image", ""), resolution=prompt.get("resolution", "768p"))
    t0 = time.monotonic()
    pid = call("/prompt", {"prompt": graph, "client_id": "e2e"})["prompt_id"]
    while True:
        hist = call("/history/" + pid).get(pid)
        if hist and hist.get("status", {}).get("completed") is not None:
            break
        time.sleep(0.2)
    client_s = time.monotonic() - t0
    status = hist["status"]
    report = json.loads(hist["outputs"]["1"]["text"][0]) if "1" in hist.get("outputs", {}) else {}
    row = {"run": "warmup" if i == 0 else "formal", "name": prompt.get("name"), "prompt": prompt["prompt"][:60],
           "mode": prompt.get("mode", "text"), "resolution": prompt.get("resolution", "768p"), "client_s": round(client_s, 2),
           "error": None if status.get("status_str") == "success" else str(status.get("messages"))[-600:],
           "status": status.get("status_str"),
           "e2e_s": report.get("e2e_s"), "qwen_s": report.get("qwen_s"), "h3_phases_s": report.get("h3_phases_s"),
           "startup_and_warmup_s": report.get("startup_and_warmup_s"), "cold_start": report.get("cold_start"),
           "video": hist.get("outputs", {}).get("2")}
    rows.append(row)
    print("RUN", json.dumps(row), flush=True)
json.dump({"profile": profile, "seed": seed, "prompts": prompts, "runs": rows}, open(out, "w"), indent=2)
