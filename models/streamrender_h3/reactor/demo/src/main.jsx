import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ReactorProvider,
  ReactorView,
  useReactor,
  useReactorMessage,
} from "@reactor-team/js-sdk";
import { Logo } from "@reactor-team/ui";
import { sourceFile, validateSelection } from "./inputs.js";
import "./style.css";

function usePreview(source) {
  const [url, setUrl] = useState("");
  useEffect(() => {
    if (!source) {
      setUrl("");
      return;
    }
    const value =
      typeof source === "string" ? source : URL.createObjectURL(source);
    setUrl(value);
    return () => {
      if (typeof source !== "string") URL.revokeObjectURL(value);
    };
  }, [source]);
  return url;
}

function Workspace() {
  const status = useReactor((s) => s.status),
    connect = useReactor((s) => s.connect),
    disconnect = useReactor((s) => s.disconnect);
  const send = useReactor((s) => s.sendCommand),
    upload = useReactor((s) => s.uploadFile),
    lastError = useReactor((s) => s.lastError);
  const [examples, setExamples] = useState([]),
    [catalogError, setCatalogError] = useState("");
  const [selected, setSelected] = useState(""),
    [video, setVideo] = useState(null),
    [image, setImage] = useState(null);
  const [prompt, setPrompt] = useState(""),
    [seed, setSeed] = useState(20260830),
    [limit, setLimit] = useState(0);
  const [world, setWorld] = useState(null),
    [error, setError] = useState(""),
    [busy, setBusy] = useState("");
  const videoUrl = usePreview(video),
    imageUrl = usePreview(image);
  const epoch = useRef(0),
    inCommand = useRef(false);
  const active = Boolean(world?.active),
    locked = active || Boolean(busy);
  useReactorMessage((message) => {
    if (message.type === "state_update") setWorld(message.data);
  });
  useEffect(() => {
    if (status !== "ready") {
      epoch.current++;
      setWorld(null);
    }
  }, [status]);
  useEffect(() => {
    const controller = new AbortController();
    fetch("/examples/index.json", { signal: controller.signal })
      .then((response) => {
        if (!response.ok)
          throw Error(
            "Official examples are not installed. Run npm run examples as described in the README. Custom uploads remain available.",
          );
        return response.json();
      })
      .then(setExamples)
      .catch((e) => {
        if (e.name !== "AbortError") setCatalogError(e.message);
      });
    return () => controller.abort();
  }, []);
  async function command(fn) {
    if (inCommand.current) return;
    inCommand.current = true;
    setError("");
    try {
      await fn();
    } catch (e) {
      setError(String(e));
    } finally {
      inCommand.current = false;
      setBusy("");
    }
  }
  async function toggleConnection() {
    await command(async () => {
      setBusy(status === "ready" ? "Disconnecting…" : "Connecting…");
      if (status === "ready") {
        await disconnect();
        setWorld(null);
      } else await connect();
    });
  }
  function chooseExample(name) {
    setSelected(name);
    setError("");
    const item = examples.find((row) => row.name === name);
    setVideo(item?.video ?? null);
    setImage(item?.image ?? null);
    setPrompt(item?.prompt ?? "");
    setLimit(item?.max_frames ?? 0);
  }
  async function render() {
    await command(async () => {
      validateSelection({ video, image, prompt, seed, limit });
      if (status !== "ready") throw Error("Connect before rendering.");
      const token = epoch.current;
      const check = () => {
        if (token !== epoch.current)
          throw Error("Connection changed. Reconnect and start again.");
      };
      setBusy("Loading selected input files…");
      const [videoFile, imageFile] = await Promise.all([
        sourceFile(video, "video"),
        sourceFile(image, "image"),
      ]);
      check();
      setBusy(
        `Uploading semantic video (${(videoFile.size / 1048576).toFixed(1)} MiB)…`,
      );
      const videoUpload = await upload(videoFile);
      check();
      setBusy("Uploading reference image…");
      const imageUpload = await upload(imageFile);
      check();
      setBusy("Starting renderer…");
      const answer = await send("start", {
        video: videoUpload,
        image: imageUpload,
        prompt,
        seed,
        max_frames: limit,
      });
      check();
      if (!Number.isInteger(answer?.data?.world_id))
        throw Error("Missing start acknowledgement.");
      setWorld((current) =>
        current?.world_id === answer.data.world_id ? current : answer.data,
      );
    });
  }
  async function stop() {
    await command(async () => {
      setBusy("Stopping…");
      const answer = await send("stop", {});
      setWorld(answer.data);
    });
  }
  const progress =
    busy ||
    (world?.error
      ? "Input decoding stopped."
      : world?.complete
        ? "Rendering complete. Buffered video may still be playing."
        : active
          ? world.ready
            ? "Rendering the next chunk…"
            : "Preparing image and text conditioning…"
          : "Choose inputs, then Start rendering.");
  const message = error || world?.error || String(lastError ?? "");
  return (
    <div className="shell">
      <header>
        <a
          className="brand"
          href="https://reactor.inc"
          target="_blank"
          rel="noreferrer"
          aria-label="Reactor"
        >
          <Logo variant="symbol" color="white" height={20} />
        </a>
        <i />
        <strong>StreamRender-H3</strong>
        <span className="tag">RECORDED VIDEO RENDERER</span>
        <div className="connection">
          <b className={`dot ${status === "ready" ? "online" : ""}`} />
          <span>{status === "ready" ? "Connected" : status}</span>
          <button
            onClick={toggleConnection}
            disabled={
              Boolean(busy) || status === "connecting" || status === "waiting"
            }
          >
            {status === "ready" ? "Disconnect" : "Connect"}
          </button>
        </div>
      </header>
      <main className="workspace">
        <aside className="sidebar">
          <span className="eyebrow">01 / CHOOSE YOUR INPUTS</span>
          <h1>
            A recorded drive,
            <br />a new look.
          </h1>
          <p className="intro">
            Render a semantic video with its opening reference and prompt.
            Choose an official example or upload your own.
          </p>
          <label className="field">
            Official examples
            <select
              aria-label="Official example"
              value={selected}
              onChange={(e) => chooseExample(e.target.value)}
              disabled={locked}
            >
              <option value="">Custom inputs / choose an example</option>
              {["A", "B"].map((group) => (
                <optgroup
                  key={group}
                  label={
                    group === "A"
                      ? "A · 20 scenes"
                      : "B · 10 trajectories × 4 appearances"
                  }
                >
                  {examples
                    .filter((x) => x.name.startsWith(group))
                    .map((x) => (
                      <option key={x.name} value={x.name}>
                        {x.name.replaceAll("_", " · ")}
                      </option>
                    ))}
                </optgroup>
              ))}
            </select>
          </label>
          <p className="hint">
            30 recorded trajectories · 60 paired references and original
            prompts. Selecting an example fills all three inputs. Nothing starts
            automatically.
          </p>
          {catalogError && <p className="notice">{catalogError}</p>}
          <label className="upload">
            <span>01</span>
            <div>
              <strong>
                {video
                  ? typeof video === "string"
                    ? video.split("/").at(-1)
                    : video.name
                  : "Upload semantic video"}
              </strong>
              <small>24 FPS · MP4 / MOV / MKV / WebM · ≤256 MiB</small>
            </div>
            <input
              aria-label="Semantic video"
              type="file"
              accept=".mp4,.mov,.mkv,.webm"
              disabled={locked}
              onChange={(e) => {
                if (e.target.files?.[0]) {
                  setVideo(e.target.files[0]);
                  setSelected("");
                  setLimit(0);
                }
                e.target.value = "";
              }}
            />
          </label>
          <label className="upload">
            <span>02</span>
            <div>
              <strong>
                {image
                  ? typeof image === "string"
                    ? image.split("/").at(-1)
                    : image.name
                  : "Upload reference image"}
              </strong>
              <small>PNG / JPEG · aligned opening view · ≤16 MiB</small>
            </div>
            <input
              aria-label="Reference image"
              type="file"
              accept="image/png,image/jpeg"
              disabled={locked}
              onChange={(e) => {
                if (e.target.files?.[0]) {
                  setImage(e.target.files[0]);
                  setSelected("");
                }
                e.target.value = "";
              }}
            />
          </label>
          <label className="field">
            03 / Rendering prompt
            <textarea
              aria-label="Rendering prompt"
              rows={7}
              maxLength={32768}
              disabled={locked}
              placeholder="Describe how the semantic video should look…"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
            />
          </label>
          <div className="numbers">
            <label className="field">
              Seed
              <input
                aria-label="Seed"
                type="number"
                min={0}
                max={2147483647}
                value={seed}
                disabled={locked}
                onChange={(e) => setSeed(Number(e.target.value))}
              />
            </label>
            <label className="field">
              Source frame limit
              <input
                aria-label="Source frame limit"
                type="number"
                min={0}
                max={14400}
                value={limit}
                disabled={locked}
                onChange={(e) => setLimit(Number(e.target.value))}
              />
            </label>
          </div>
          <p className="hint">
            0 reads the full video, up to 10 minutes. Official examples request
            1450 frames. Image and prompt stay fixed for each render.
          </p>
          <button
            className="primary"
            onClick={active ? stop : render}
            disabled={
              Boolean(busy) ||
              status !== "ready" ||
              (!active && (!video || !image || !prompt.trim()))
            }
          >
            {active
              ? "Stop rendering"
              : world?.complete
                ? "Render again"
                : "Start rendering"}
          </button>
          <p className="progress" role="status">
            {progress}
          </p>
          {world?.complete && (
            <p className="hint">
              {world.output_frames} output frames · {world.unused_tail_frames}{" "}
              source frames left in the incomplete final chunk.
            </p>
          )}
          {message && (
            <div className="error" role="alert">
              {message}
              {message.includes("orphaned") && (
                <p>
                  The previous connection is closing. Wait about a minute before
                  reconnecting.
                </p>
              )}
            </div>
          )}
        </aside>
        <section className="stage-column">
          <div className="stage-heading">
            <div>
              <span className="eyebrow">02 / GENERATED OUTPUT</span>
              <h2>The drive, re-rendered.</h2>
            </div>
            <span className="tag">1344 × 768 · 24 FPS playback</span>
          </div>
          <div className="stage">
            <ReactorView
              track="main_video"
              muted
              videoObjectFit="contain"
              className="video"
            />
            {!world?.output_frames && (
              <div className="placeholder">
                <span className="eyebrow">
                  {active ? "PREPARING FIRST FRAMES" : "READY WHEN YOU ARE"}
                </span>
                <h2>
                  {active
                    ? "Conditioning your render"
                    : "Choose inputs. Press Start."}
                </h2>
                <p>
                  Cold initialization may take 1–2 minutes.
                  <br />
                  The source video is decoded on the server in frame order.
                </p>
              </div>
            )}
            <div className="stage-badge">
              Run {world?.world_id ?? "—"} · {world?.completed_chunks ?? 0}{" "}
              chunks · {world?.next_frame ?? 0}
              {world?.total_frames ? ` / ${world.total_frames}` : ""} source
              frames
              {world?.last_chunk_seconds != null &&
                ` · ${world.last_chunk_seconds.toFixed(2)} s/chunk`}
            </div>
          </div>
          <div className="input-previews">
            <section>
              <h3>Semantic input</h3>
              {videoUrl ? (
                <video
                  aria-label="Semantic input preview"
                  src={videoUrl}
                  controls
                  muted
                  preload="metadata"
                />
              ) : (
                <div className="empty">Choose a recorded trajectory</div>
              )}
              <p className="hint">
                Independent source preview. Rendering always starts at frame
                zero.
              </p>
            </section>
            <section>
              <h3>Appearance reference</h3>
              {imageUrl ? (
                <img src={imageUrl} alt="Selected appearance reference" />
              ) : (
                <div className="empty">Choose an aligned opening image</div>
              )}
              <p className="hint">
                Use the reference and prompt paired with this trajectory.
              </p>
            </section>
          </div>
          <footer>
            StreamRender-H3 · NVlabs / Sana<span>Built on Reactor</span>
          </footer>
        </section>
      </main>
    </div>
  );
}

createRoot(document.querySelector("#root")).render(
  <ReactorProvider
    modelName="streamrender-h3"
    local
    apiUrl={import.meta.env.VITE_REACTOR_URL || "http://localhost:8080"}
    modelTracks={[{ name: "main_video", kind: "video", direction: "recvonly" }]}
    connectOptions={{ autoConnect: false }}
  >
    <Workspace />
  </ReactorProvider>,
);
