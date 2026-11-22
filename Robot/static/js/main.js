/* SANA-Robot project page: charts, the memory timeline and the small interactive widgets.
   No dependencies. Numbers come from data.js (hand-entered, with sources) and bench_data.js (generated). */
(function () {
  "use strict";
  const S = window.SITE, B = window.BENCH;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const el = (tag, cls, html) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  };
  const name = (n) => n.replace(/π(0(?:\.5)?)/g, "π<sub>$1</sub>");
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ───────── horizontal bars ───────── */
  // rows: [{name, value, text, cls, tag}], opts: {max, reserve (px kept for the value label), ref, refLabel}
  function renderBars(box, rows, opts) {
    const reserve = opts.reserve || 64;
    box.innerHTML = "";
    rows.forEach((r, i) => {
      const row = el("div", "bar " + (r.cls || ""));
      const label = el("span", "name", name(r.name) + (r.tag ? ` <small style="display:inline;font-size:11px;letter-spacing:.04em">${r.tag}</small>` : ""));
      label.title = r.name;
      const track = el("div", "track");
      const fill = el("div", "fill");
      const val = el("span", "val", r.text);
      const frac = Math.max(0, r.value / opts.max);
      // A value just below the reference line gets its label to the right of the line, not under it.
      const labelFrac = opts.ref != null && r.value < opts.ref ? opts.ref / opts.max : frac;
      fill.dataset.w = `calc((100% - ${reserve}px) * ${frac.toFixed(4)})`;
      val.dataset.w = `calc((100% - ${reserve}px) * ${labelFrac.toFixed(4)})`;
      track.append(fill, val);
      if (opts.ref != null) {
        const ref = el("div", "ref", i === 0 && opts.refLabel ? `<span>${opts.refLabel}</span>` : "");
        ref.style.left = `calc((100% - ${reserve}px) * ${(opts.ref / opts.max).toFixed(4)})`;
        track.append(ref);
      }
      row.append(label, track);
      box.append(row);
    });
    if (box.dataset.seen || reduced) grow(box); else seen.observe(box);
  }
  function grow(box) {
    box.dataset.seen = "1";
    requestAnimationFrame(() => requestAnimationFrame(() => {
      $$(".fill", box).forEach((f) => (f.style.width = f.dataset.w));
      $$(".val", box).forEach((v) => (v.style.left = v.dataset.w));
    }));
  }
  const seen = new IntersectionObserver((entries) => {
    entries.forEach((e) => { if (e.isIntersecting) { grow(e.target); seen.unobserve(e.target); } });
  }, { threshold: 0.25 });

  const pct = (v, d = 1) => v.toFixed(d);

  /* ───────── real-world task bars ───────── */
  $$("[data-bars]").forEach((box) => {
    const rows = S.real[box.dataset.bars].map(([n, v, ours]) => ({ name: n, value: v, text: pct(v), cls: ours ? "ours" : "" }));
    const ref = box.dataset.ref ? parseFloat(box.dataset.ref) : null;
    renderBars(box, rows, { max: 100, reserve: 52, ref, refLabel: box.dataset.refLabel });
  });
  $$("[data-lowbit]").forEach((box) => {
    const rows = S.lowbit[box.dataset.lowbit].map(([n, v]) => ({ name: n, value: v, text: pct(v, v % 1 ? 1 : 0) }));
    renderBars(box, rows, { max: 100, reserve: 52 });
  });

  /* ───────── benchmark explorer ───────── */
  const BENCHES = [
    { id: "robodojo", label: "RoboDojo", pair: true, col: "Avg.", max: "auto", dec: 2,
      note: "42 bimanual simulation tasks in five capability dimensions. The score credits partial progress. With 4.5B parameters, SANA-Robot-AR achieves the best average, 34.31 / 27.09%, and the best scores on the memory, precision and long-horizon dimensions. All results, including SANA-Robot-AR, are from the official leaderboard." },
    { id: "robotwin", label: "RoboTwin 2.0", col: "Avg.", max: 100, dec: 1,
      note: "50 bimanual tasks under the Clean2Random protocol: post-trained only on the 2,500 demonstrations collected in clean scenes, then evaluated in clean scenes and in randomized scenes with changed backgrounds, lighting, clutter and table height." },
    { id: "libero_plus", label: "LIBERO-Plus", col: "Total", max: 100, dec: 1,
      note: "Zero-shot success rate under seven kinds of perturbation, for policies trained on the LIBERO dataset." },
    { id: "libero_pro", label: "LIBERO-Pro", col: "Avg.", max: 100, dec: 2,
      note: "Zero-shot success rate for policies trained on the LIBERO dataset. The gain comes largely from position perturbations, under which SANA-Robot-AR succeeds in 47.7% of the episodes, against at most 39.3% for the other methods." },
    { id: "libero_max", label: "LIBERO-Max", col: "Dynamic", max: 100, dec: 1,
      note: "Zero-shot success rate in base and dynamic rollouts, for policies trained on the LIBERO dataset." },
  ];
  const GROUP = { VLA: "VLA", WAM: "WAM", "LLM as policy": "LLM", Ours: "" };
  const TOP = 8;
  const bench = { i: 0, col: 0, metric: 0 };

  function benchSetup() {
    const tabs = $("#benchTabs");
    BENCHES.forEach((b, i) => {
      const t = el("button", "", b.label);
      t.type = "button"; t.setAttribute("role", "tab");
      t.onclick = () => { bench.i = i; bench.col = B[b.id].columns.indexOf(b.col); benchRender(); };
      tabs.append(t);
    });
    bench.col = B[BENCHES[0].id].columns.indexOf(BENCHES[0].col);
    benchRender();
  }
  function benchRender() {
    const b = BENCHES[bench.i], t = B[b.id];
    $$("#benchTabs button").forEach((x, i) => x.setAttribute("aria-selected", i === bench.i));
    const cols = $("#benchCols"); cols.innerHTML = "";
    t.columns.forEach((c, i) => {
      const p = el("button", "", c);
      p.type = "button"; p.setAttribute("aria-pressed", i === bench.col);
      p.onclick = () => { bench.col = i; benchRender(); };
      cols.append(p);
    });
    const metric = $("#benchMetric"); metric.innerHTML = "";
    if (b.pair) ["Score", "Success rate"].forEach((m, i) => {
      const p = el("button", "", m);
      p.type = "button"; p.setAttribute("aria-pressed", i === bench.metric);
      p.onclick = () => { bench.metric = i; benchRender(); };
      metric.append(p);
    });
    const value = (r) => (b.pair ? r.cells[bench.col][bench.metric] : r.cells[bench.col]);
    const text = (r) => (b.pair ? r.cells[bench.col].map((v) => v.toFixed(b.dec)).join(" / ") : r.cells[bench.col].toFixed(b.dec));
    const sorted = t.rows.slice().sort((x, y) => value(y) - value(x));
    const baselines = sorted.filter((r) => !r.ours);
    const keep = new Set(baselines.slice(0, TOP).map((r) => r.name));
    const rows = sorted.filter((r) => r.ours || keep.has(r.name))
      .map((r) => ({ name: r.name, value: value(r), text: text(r), cls: r.ours ? "ours" : "", tag: GROUP[r.group] }));
    const top = Math.max(...rows.map((r) => r.value));
    const max = b.max === "auto" ? Math.ceil(top / 10) * 10 : b.max;
    renderBars($("#benchBars"), rows, { max, reserve: b.pair ? 112 : 58 });
    const unit = b.pair ? "Each bar label reads average score / success rate (%). " : "Success rate (%). ";
    $("#benchNote").innerHTML = unit + b.note + ` The top ${Math.min(TOP, baselines.length)} of ${baselines.length} baselines for this column are shown; the full tables are in the paper.`;
  }

  /* ───────── svg helpers ───────── */
  const NS = "http://www.w3.org/2000/svg";
  function svg(tag, attrs, text) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }
  const C = { ink: "#12150c", muted: "#565c4c", line: "#e2e5da", green: "#76b900", greenInk: "#436b00", gray: "#9aa08d" };

  /* ───────── speed against memory ───────── */
  function tradeoff() {
    const W = 440, H = 330, m = { l: 48, r: 18, t: 18, b: 46 };
    const x0 = 50, x1 = 800, y1 = 60;
    const X = (v) => m.l + (Math.log(v / x0) / Math.log(x1 / x0)) * (W - m.l - m.r);
    const Y = (v) => H - m.b - (v / y1) * (H - m.t - m.b);
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Latency per action chunk on an RTX 5090 against the RoboDojo memory score" });
    [0, 20, 40, 60].forEach((v) => {
      s.append(svg("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: C.line }));
      s.append(svg("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end", "font-size": 12, fill: C.muted }, v));
    });
    [50, 100, 200, 400, 800].forEach((v) => {
      s.append(svg("line", { x1: X(v), x2: X(v), y1: H - m.b, y2: H - m.b + 5, stroke: C.gray }));
      s.append(svg("text", { x: X(v), y: H - m.b + 19, "text-anchor": "middle", "font-size": 12, fill: C.muted }, v));
    });
    s.append(svg("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", "font-size": 12.5, fill: C.muted }, "Latency per chunk on RTX 5090 (ms, log scale)"));
    s.append(svg("text", { x: 14, y: (m.t + H - m.b) / 2, "text-anchor": "middle", "font-size": 12.5, fill: C.muted, transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})` }, "RoboDojo memory score"));
    S.tradeoff.forEach((p) => {
      const left = p.ms > 400;
      s.append(svg("circle", { cx: X(p.ms), cy: Y(p.score), r: p.ours ? 9 : 6, fill: p.ours ? C.green : C.gray, stroke: "#fff", "stroke-width": 2 }));
      const t = svg("text", { x: X(p.ms) + (left ? -12 : 13), y: Y(p.score) + (p.ours ? -12 : 4), "text-anchor": left ? "end" : "start",
        "font-size": p.ours ? 14 : 12.5, "font-weight": p.ours ? 700 : 400, fill: p.ours ? C.ink : C.muted }, p.name);
      s.append(t);
      if (p.ours) s.append(svg("text", { x: X(p.ms) + 13, y: Y(p.score) + 5, "font-size": 12.5, fill: C.muted }, `${p.ms} ms · ${p.score.toFixed(2)}`));
    });
    $("#tradeoffChart").append(s);
  }

  /* ───────── DROID policy evaluation ───────── */
  function droid() {
    const W = 520, H = 400, m = { l: 52, r: 16, t: 16, b: 48 };
    const X = (v) => m.l + (v / 100) * (W - m.l - m.r), Y = (v) => H - m.b - (v / 100) * (H - m.t - m.b);
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Success rate in the world model against the real robot for 21 policy and task pairs" });
    [0, 25, 50, 75, 100].forEach((v) => {
      s.append(svg("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: C.line }));
      s.append(svg("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end", "font-size": 12, fill: C.muted }, v));
      s.append(svg("text", { x: X(v), y: H - m.b + 18, "text-anchor": "middle", "font-size": 12, fill: C.muted }, v));
    });
    s.append(svg("text", { x: (m.l + W - m.r) / 2, y: H - 8, "text-anchor": "middle", "font-size": 12.5, fill: C.muted }, "Success rate on the real robot (%)"));
    s.append(svg("text", { x: 14, y: (m.t + H - m.b) / 2, "text-anchor": "middle", "font-size": 12.5, fill: C.muted, transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})` }, "Success rate in the world model (%)"));
    s.append(svg("line", { x1: X(0), y1: Y(0), x2: X(100), y2: Y(100), stroke: C.ink, "stroke-dasharray": "5 5", "stroke-width": 1.2 }));
    s.append(svg("text", { x: X(96), y: Y(86), "text-anchor": "middle", "font-size": 12, fill: C.ink }, "y = x"));
    // Fits reported in the paper: ours y = 0.93x + 9.4, Ctrl-World y = 0.82x - 10.8.
    const fit = (a, b, color) => {
      const xa = Math.max(0, -b / a), xb = Math.min(100, (100 - b) / a);
      s.append(svg("line", { x1: X(xa), y1: Y(a * xa + b), x2: X(xb), y2: Y(a * xb + b), stroke: color, "stroke-width": 2, opacity: 0.7 }));
    };
    fit(0.82, -10.8, C.gray); fit(0.93, 9.4, C.green);
    S.droid.forEach(([task, policy, real, ctrl, ours]) => {
      const a = svg("circle", { cx: X(real), cy: Y(ctrl), r: 5, fill: "#fff", stroke: C.gray, "stroke-width": 2 });
      a.append(svg("title", {}, `${task}, ${policy}: real ${real}%, Ctrl-World ${ctrl}%`));
      const b = svg("circle", { cx: X(real), cy: Y(ours), r: 6, fill: C.green, stroke: "#fff", "stroke-width": 1.5 });
      b.append(svg("title", {}, `${task}, ${policy}: real ${real}%, SANA-Robot ${ours}%`));
      s.append(a, b);
    });
    const lx = m.l + 14, ly = m.t + 12;
    s.append(svg("circle", { cx: lx, cy: ly, r: 6, fill: C.green }));
    s.append(svg("text", { x: lx + 12, y: ly + 4, "font-size": 13, "font-weight": 700, fill: C.ink }, "SANA-Robot · y = 0.93x + 9.4"));
    s.append(svg("circle", { cx: lx, cy: ly + 22, r: 5, fill: "#fff", stroke: C.gray, "stroke-width": 2 }));
    s.append(svg("text", { x: lx + 12, y: ly + 26, "font-size": 13, fill: C.muted }, "Ctrl-World · y = 0.82x − 10.8"));
    $("#droidChart").append(s);
  }

  /* ───────── latency per device ───────── */
  function latency() {
    const box = $("#latency");
    S.latency.forEach((d) => {
      const card = el("div", "card lat", `<h4>${d.gpu}</h4>`);
      const bars = el("div", "bars");
      const x = (v) => `<em>${(d.eager / v).toFixed(1)}×</em>`;
      renderBars(bars, [
        { name: "Eager", value: d.eager, text: `${d.eager.toFixed(1)} ms` },
        { name: "Optimized", value: d.opt, text: `${d.opt.toFixed(1)} ms ${x(d.opt)}`, cls: "mid" },
        { name: "+ FP8", value: d.fp8, text: `${d.fp8.toFixed(1)} ms ${x(d.fp8)}`, cls: "mid" },
        { name: "+ NVFP4", value: d.nvfp4, text: `${d.nvfp4.toFixed(1)} ms ${x(d.nvfp4)}`, cls: "best" },
      ], { max: d.eager, reserve: 118 });
      card.append(bars); box.append(card);
    });
  }

  /* ───────── synchronous against asynchronous execution, to scale ───────── */
  function lanes() {
    const exec = (S.chunk.actions / S.chunk.hz) * 1000;        // 1066.7 ms to execute one chunk
    const L = S.hockey.sync.stallMs;                            // 181.3 ms per request
    const total = 3 * (exec + L);
    const d = 8, period = ((S.chunk.actions - d) / S.chunk.hz) * 1000, lead = (d / S.chunk.hz) * 1000;
    const seg = (cls, a, b) => `<i class="${cls}" style="left:${(a / total * 100).toFixed(3)}%;width:${((Math.min(b, total) - a) / total * 100).toFixed(3)}%"></i>`;
    let sync = "", robot = "", infer = "";
    for (let k = 0; k < 3; k++) { const t = k * (exec + L); sync += seg("stall", t, t + L) + seg("exec", t + L, t + L + exec); }
    robot += seg("exec", 0, lead);
    for (let t = lead; t < total; t += period) robot += seg("exec", t, t + period);
    for (let t = 0; t + L < total; t += period) infer += seg("infer", t, t + L);
    $("#lanes").innerHTML = `<div class="laneviz">
      <div class="row"><b>Synchronous</b><div class="t">${sync}</div></div>
      <div class="row gap"><b>Asynchronous</b><div class="t">${robot}</div></div>
      <div class="row"><span>inference</span><div class="t">${infer}</div></div>
      <div class="legend"><span><i class="sw" style="background:var(--green)"></i>robot executes actions</span>
      <span><i class="sw" style="background:var(--red)"></i>robot idles during a request</span>
      <span><i class="sw" style="background:var(--blue)"></i>request runs while the robot moves</span></div></div>`;
    const a = S.hockey.sync, b = S.hockey.async;
    $("#deltas").innerHTML =
      `<div><span>Stall before each chunk</span><b><s>${a.stallMs} ms</s> → ≈ ${b.stallMs} ms</b></div>
       <div><span>Effective control frequency</span><b><s>${a.hz} Hz</s> → ${b.hz} Hz</b></div>
       <div><span>Pucks hit</span><b><s>${a.hit}%</s> → ${b.hit}%</b></div>`;
  }

  /* ───────── unified model ───────── */
  function unified() {
    const TOK = [
      ["o", "Observation", "multi-view video latent"],
      ["s", "Proprioception", "one token"],
      ["z", "Future video", "latent of the next 32-step chunk"],
      ["a", "Actions", "32 action tokens"],
    ];
    const MODES = [
      { label: "Policy", noised: "za", formula: "p(z, a | o, s)",
        desc: "Predicts the future video and the actions from the current observation and proprioception. The video and the actions of a chunk share one noise level and are denoised jointly. This mode drives manipulation." },
      { label: "Forward dynamics (FDM)", noised: "z", formula: "p(z | o, s, a)",
        desc: "Simulates the video that an action chunk causes. This mode serves as a world simulator for policy evaluation." },
      { label: "Inverse dynamics (IDM)", noised: "a", formula: "p(a | o, z, s)",
        desc: "Recovers the actions that explain a given video, which makes the video and the actions predicted by the policy more consistent." },
    ];
    const tabs = $("#uniTabs"), toks = $("#uniTokens");
    TOK.forEach(([k, t, d]) => toks.append(el("div", "tok", `<b>${k}</b><span>${t}<br><small>${d}</small></span><em></em>`)));
    const show = (i) => {
      $$("button", tabs).forEach((b, j) => b.setAttribute("aria-selected", i === j));
      $$(".tok", toks).forEach((t, j) => {
        const noised = MODES[i].noised.includes(TOK[j][0]);
        t.classList.toggle("noised", noised);
        $("em", t).textContent = noised ? "noised · target" : "clean · condition";
        $("small", t).style.color = noised ? "#d8e6b8" : "";
      });
      $("#uniFormula").textContent = MODES[i].formula;
      $("#uniDesc").textContent = MODES[i].desc;
    };
    MODES.forEach((m, i) => {
      const b = el("button", "", m.label);
      b.type = "button"; b.setAttribute("role", "tab"); b.onclick = () => show(i);
      tabs.append(b);
    });
    show(0);
  }

  /* ───────── memory timeline ───────── */
  function memlab() {
    const video = $("#memVideo"), strip = $("#memStrip");
    const TC = S.chunk.actions / S.chunk.hz;                    // seconds per chunk
    const SPR = { cols: 10, rows: 2, src: "https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets/resolve/main/Robot/img/shell_chunks.jpg" };   // tools/build_chunk_thumbs.py
    const DESIGNS = {
      hybrid: { label: "Hybrid memory (ours)",
        desc: "Every past chunk is encoded and written into the memory once. The GDN state covers the whole episode, and the softmax window keeps the latest chunks exactly." },
      vpp2: { label: "VPP2 memory",
        desc: "Keeps the first frame and 8 frames sampled backward from the current step at intervals of 1 s. All other history is lost, and the kept frames are encoded and forwarded again at every request." },
      interndelta: { label: "InternW0-Δ memory",
        desc: "Keeps the first frame and a single frame 32 steps back. All other history is lost." },
    };
    const st = { design: "hybrid", N: 3, c: -1, key: "" };              // the Shell Game policy uses N = 3
    const dur = () => (isFinite(video.duration) && video.duration > 0 ? video.duration : 21.2);
    const samples = (t) => (st.design === "vpp2" ? [1, 2, 3, 4, 5, 6, 7, 8].map((k) => t - k) : [t - TC]);

    const thumb = (chunk, label, cls, w, h) => {
      const d = el("div", "thumb " + (cls || ""));
      const im = el("i");
      if (cls && cls.includes("sink")) { im.style.backgroundImage = "url(https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets/resolve/main/Robot/img/shell_first.jpg)"; im.style.backgroundSize = "cover"; }
      else if (chunk >= 0) {
        im.style.backgroundImage = `url(${SPR.src})`;
        im.style.backgroundSize = `${SPR.cols * w}px ${SPR.rows * h}px`;
        im.style.backgroundPosition = `-${(chunk % SPR.cols) * w}px -${Math.floor(chunk / SPR.cols) * h}px`;
      } else d.classList.add("ghost");
      d.append(im, el("span", "", label));
      return d;
    };

    // tabs, window selector, score bars
    const tabs = $("#memTabs");
    Object.keys(DESIGNS).forEach((id) => {
      const b = el("button", "", DESIGNS[id].label);
      b.type = "button"; b.setAttribute("role", "tab"); b.dataset.id = id;
      b.onclick = () => { st.design = id; st.key = ""; frame(true); scores(); };
      tabs.append(b);
    });
    const seg = $("#memN .seg");
    [1, 2, 3, 4].forEach((n) => {
      const b = el("button", "", String(n));
      b.type = "button"; b.onclick = () => { st.N = n; st.key = ""; frame(true); };
      seg.append(b);
    });
    function scores() {
      renderBars($("#memBars"), S.memoryDesigns.map((d) => ({
        name: d.name, value: d.success, text: d.success.toFixed(1),
        cls: (d.id === "hybrid" ? "ours " : "") + (d.id === st.design ? "" : "dim"),
      })), { max: 60, reserve: 48 });
    }

    function build(c, t) {
      $$("button", tabs).forEach((b) => b.setAttribute("aria-selected", b.dataset.id === st.design));
      $$("button", seg).forEach((b, i) => b.setAttribute("aria-pressed", i + 1 === st.N));
      $("#memN").classList.toggle("off", st.design !== "hybrid");
      $("#memDesc").textContent = DESIGNS[st.design].desc;
      const lanes = $("#memLanes"); lanes.innerHTML = "";
      const hybrid = st.design === "hybrid";

      const a = el("div", "lane");
      const row = el("div", "thumbs");
      row.append(thumb(0, hybrid ? "first frame" : "first", "sink" + (hybrid ? "" : " small"), 84, 47));
      const sw = (style, txt) => `<span><i class="sw" style="${style}"></i>${txt}</span>`;
      const lastN = `the last ${st.N === 1 ? "chunk" : st.N + " chunks"}`;
      if (hybrid) {
        a.append(el("div", "lane-head", `<b><i class="sw" style="background:var(--blue)"></i>Softmax layers · sliding window</b><span>exact latents of the first frame and ${lastN}</span>`));
        for (let k = st.N; k >= 1; k--) row.append(thumb(c - k, c - k >= 0 ? `chunk ${c - k + 1}` : "not yet", "", 84, 47));
      } else {
        a.append(el("div", "lane-head", `<b>Kept frames</b><span>${st.design === "vpp2" ? "the first frame and 8 frames, 1 s apart" : "the first frame and one frame 32 steps back"}</span>`));
        samples(t).forEach((s) => row.append(thumb(s >= 0 ? Math.floor(s / TC) : -1, s >= 0 ? `−${(t - s).toFixed(1)} s` : "not yet", "small", 48, 27)));
      }
      a.append(row); lanes.append(a);

      const b = el("div", "lane" + (hybrid ? "" : " off"));
      if (hybrid) {
        b.append(el("div", "lane-head", `<b><i class="sw" style="background:var(--green)"></i>GDN layers · fixed-size state</b><span>the whole episode, compressed</span>`));
        const g = el("div", "gdn"), box = el("div", "gdn-state");
        for (let i = 0; i < 24; i++) { const u = el("u"); u.style.opacity = c <= 0 ? 0.12 : 0.25 + 0.75 * (((i * 7 + c * 5) % 11) / 10); box.append(u); }
        g.append(box, el("div", "gdn-text", `<b>${Math.max(c, 0)} ${c === 1 ? "chunk" : "chunks"}</b> absorbed so far<br>the state keeps the same size`));
        b.append(g);
      } else {
        b.append(el("div", "lane-head", `<b>No state over the whole episode</b><span>fixed capacity</span>`));
        b.append(el("div", "gdn-text", "Sparsely sampled frames are coarse: they may miss the key motions or object states between two samples."));
      }
      lanes.append(b);

      // timeline: two rows over the same time axis, as in the overview video. Hybrid memory: the softmax layers hold
      // the first frame and the last N chunks, the GDN layers every chunk written so far. The other designs keep a few
      // frames (upper row) and have no state over the whole episode (lower row, empty).
      strip.innerHTML = "";
      const top = el("div", "srow"), low = el("div", "srow g" + (hybrid ? "" : " none"));
      const grid = () => { const g = el("div", "grid"); g.style.backgroundSize = `${(TC / dur()) * 100}% 100%`; return g; };
      if (hybrid) {
        top.append(el("div", "win-fill"), grid());
        low.append(el("div", "gdn-fill"), grid());
      } else {
        top.append(el("div", "past-lost"));
        samples(t).forEach(() => top.append(el("div", "tick")));
      }
      top.append(el("div", "sinkmark"));
      strip.append(top, low, el("div", "head"));
      $("#memRows").innerHTML = hybrid
        ? `<span><i class="sw" style="background:var(--blue)"></i>Softmax <em>layers</em></span><span><i class="sw" style="background:var(--green)"></i>GDN <em>layers</em></span>`
        : `<span>Kept <em>frames</em></span><span class="none">No global <em>state</em></span>`;
      $("#memLegend").innerHTML = hybrid
        ? sw("background:var(--ink)", "first frame") + sw("background:var(--blue)", `softmax window: the first frame and ${lastN}, exact`) + sw("background:var(--green)", "GDN state: every past chunk, compressed")
        : sw("background:var(--ink)", "first frame") + sw("background:var(--blue)", "kept frame") + sw("background:repeating-linear-gradient(135deg,#e3e5dc 0 4px,#c9ccbf 4px 8px)", "not in the memory");
    }

    function frame(force) {
      const D = dur(), t = Math.min(video.currentTime || 0, D), c = Math.floor(t / TC);
      const hybrid = st.design === "hybrid";
      const key = `${st.design}|${st.N}|${c}`;
      if (force || key !== st.key) { st.key = key; build(c, t); }
      const P = (v) => `${Math.max(0, Math.min(1, v / D)) * 100}%`;
      $(".head", strip).style.left = P(t);
      if (hybrid) {
        $(".gdn-fill", strip).style.width = P(c * TC);
        const w0 = Math.max(0, (c - st.N) * TC);
        const win = $(".win-fill", strip); win.style.left = P(w0); win.style.width = P(c * TC - w0);
      } else {
        $(".past-lost", strip).style.width = P(t);
        const ss = samples(t);
        $$(".tick", strip).forEach((k, i) => { k.style.left = P(ss[i]); k.style.display = ss[i] >= 0 ? "" : "none"; });
      }
      $("#memClock").textContent = `Chunk ${c + 1} · ${t.toFixed(1)} s into the episode`;
      $("#memCover").textContent = hybrid ? `The memory covers all ${(c * TC).toFixed(1)} s observed so far`
        : st.design === "vpp2" ? `The memory covers 8 frames from the last ${Math.min(8, t).toFixed(1)} s`
        : `The memory covers 1 frame, ${Math.min(TC, t).toFixed(1)} s back`;
      strip.setAttribute("aria-valuenow", Math.round(t));
      strip.setAttribute("aria-valuemax", Math.round(D));
    }

    const loop = () => { frame(false); if (!video.paused) requestAnimationFrame(loop); };
    video.addEventListener("play", () => { $("#memPlay").textContent = "❚❚"; $("#memPlay").setAttribute("aria-label", "Pause the episode"); loop(); });
    video.addEventListener("pause", () => { $("#memPlay").textContent = "▶"; $("#memPlay").setAttribute("aria-label", "Play the episode"); });
    ["seeked", "loadedmetadata", "timeupdate"].forEach((e) => video.addEventListener(e, () => frame(e === "loadedmetadata")));
    $("#memPlay").onclick = (e) => {
      e.stopPropagation();
      if (video.paused) { delete video.dataset.userPaused; video.play(); } else { video.dataset.userPaused = "1"; video.pause(); }
    };
    const seek = (ev) => { const r = strip.getBoundingClientRect(); video.currentTime = Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width)) * dur(); frame(false); };
    let drag = false;
    strip.addEventListener("pointerdown", (ev) => { drag = true; strip.setPointerCapture(ev.pointerId); seek(ev); });
    strip.addEventListener("pointermove", (ev) => drag && seek(ev));
    strip.addEventListener("pointerup", () => (drag = false));
    strip.addEventListener("keydown", (ev) => {
      const step = ev.key === "ArrowRight" ? TC : ev.key === "ArrowLeft" ? -TC : 0;
      if (step) { ev.preventDefault(); video.currentTime = Math.max(0, Math.min(dur() - 0.1, video.currentTime + step)); }
    });
    const phases = () => {
      $("#memPhases").innerHTML = S.shellPhases.map((p) =>
        `<span style="left:${(p.from / dur()) * 100}%;width:${((p.to - p.from) / dur()) * 100}%">${p.label}</span>`).join("");
    };
    video.addEventListener("loadedmetadata", phases);
    scores(); phases(); frame(true);
    $("#memPlay").textContent = "▶";
  }


  /* ───────── hero reel: the chip names the task on screen ───────── */
  /* ───────── architecture ───────── */
  function arch() {
    const G = { g: "#76b900", gs: "#eaf4d3", gd: "#3f6400", b: "#2a78d6", bs: "#dbe9fa", bd: "#1c5fb0", line: "#dfe3d6" };
    const grid = (x, y, cols, rows, size, gap, attrs) => {
      let o = "";
      for (let r = 0; r < rows; r++) for (let i = 0; i < cols; i++)
        o += `<rect x="${x + i * (size + gap)}" y="${y + r * (size + gap)}" width="${size}" height="${size}" rx="2.5" ${attrs}/>`;
      return o;
    };
    const clean = (c) => `fill="${c === "g" ? G.gs : G.bs}" stroke="${c === "g" ? G.g : G.b}" stroke-width="1.2"`;
    const noisy = (c) => `fill="url(#arch-n${c})" class="a-noisy"`;
    const card = (x, y, w, h, tag) =>
      `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="14" fill="#fff" stroke="${G.line}" stroke-width="1.5"/><text x="${x + 16}" y="${y + 25}" class="a-tag">${tag}</text>`;
    const flow = (d) => `<path d="${d}" class="a-flow" marker-end="url(#arch-ah)"/>`;
    const img = (href, x, y, w, h, cls) =>
      `<image href="https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets/resolve/main/Robot/img/${href}" x="${x}" y="${y}" width="${w}" height="${h}" preserveAspectRatio="xMidYMid slice" style="clip-path:inset(0 round 6px)" class="${cls || ""}"/>`;
    const v = (x, y, ch) => `<text x="${x}" y="${y}" class="a-var">${ch}</text>`;
    const pill = (y, kind, i) => {
      const gdn = kind === "GDN";
      return `<rect x="690" y="${y}" width="142" height="40" rx="10" fill="${gdn ? G.gs : G.bs}" stroke="${gdn ? G.g : G.b}" stroke-width="1.5"/>
        <rect x="690" y="${y}" width="142" height="40" rx="10" fill="${gdn ? G.g : G.b}" class="a-sweep" style="animation-delay:${i * 0.45}s"/>
        <text x="706" y="${y + 25}" class="a-t a-b" style="fill:${gdn ? G.gd : G.bd}">${kind}</text>
        <text x="${gdn ? 746 : 770}" y="${y + 25}" class="a-s">${gdn ? "linear attn." : "exact"}</text>`;
    };
    let state = "";
    for (let r = 0; r < 2; r++) for (let i = 0; i < 12; i++)
      state += `<rect x="${449 + i * 12}" y="${540 + r * 10.5}" width="9" height="8" rx="2" fill="${G.g}" class="a-cell" style="animation-delay:${((i * 5 + r * 3) % 12) * 0.25}s"/>`;

    $("#archChart").innerHTML = `<svg viewBox="0 0 1200 612" role="img" aria-label="Architecture of SANA-Robot: a text prompt, multi-view video, proprioception and actions are embedded into one token sequence, processed by a hybrid diffusion transformer of 8 blocks with three GDN layers and one softmax layer each, and denoised into the future video and the actions. SANA-Robot-AR reads a hybrid memory.">
      <defs>
        <marker id="arch-ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="${C.ink}"/></marker>
        <pattern id="arch-ng" width="5" height="5" patternUnits="userSpaceOnUse"><rect width="5" height="5" fill="${G.gd}"/><circle cx="1.5" cy="1.5" r=".9" fill="#c4e07d"/></pattern>
        <pattern id="arch-nb" width="5" height="5" patternUnits="userSpaceOnUse"><rect width="5" height="5" fill="${G.bd}"/><circle cx="1.5" cy="1.5" r=".9" fill="#b5d0f4"/></pattern>
      </defs>

      <!-- inputs -->
      ${card(16, 16, 262, 142, "PROMPT")}
      <text class="a-s" x="32" y="64">Embodiment</text><text class="a-t" x="118" y="64">dual-arm robot</text>
      <text class="a-s" x="32" y="85">Action</text><text class="a-t" x="118" y="85">end-effector pose, joints</text>
      <text class="a-s" x="32" y="106">Views</text><text class="a-t" x="118" y="106">camera description</text>
      <text class="a-s" x="32" y="127">Task</text><text class="a-t" x="118" y="127">zip up the green jacket,</text><text class="a-t" x="118" y="145">leave it on the tabletop</text>

      ${card(16, 174, 262, 196, "MULTI-VIEW VIDEO")}
      ${img("arch_view1.jpg", 32, 208, 112, 70)}${img("arch_view2.jpg", 150, 208, 112, 70)}
      ${img("arch_view3.jpg", 32, 284, 112, 70)}${img("arch_view4.jpg", 150, 284, 112, 70)}

      ${card(16, 386, 262, 92, "ROBOT STATE AND ACTIONS")}
      <rect x="32" y="424" width="11" height="11" rx="2.5" ${clean("b")}/><text class="a-t" x="52" y="434">proprioception</text>
      ${grid(32, 448, 4, 1, 11, 3, clean("b"))}<text class="a-t" x="96" y="458">action chunk, 32 steps</text>

      <!-- encoders -->
      ${flow("M278 272 H298")}
      <path d="M300 222 L364 240 V304 L300 322 Z" fill="#fff" stroke="${C.ink}" stroke-width="1.5" stroke-linejoin="round"/>
      <text class="a-t a-b" x="332" y="270" text-anchor="middle">VAE</text><text class="a-xs" x="332" y="287" text-anchor="middle">32×32×8</text>
      ${flow("M278 432 H298")}
      <rect x="300" y="412" width="64" height="40" rx="9" fill="#fff" stroke="${C.ink}" stroke-width="1.5"/>
      <text class="a-t a-b" x="332" y="437" text-anchor="middle">Linear</text>

      <!-- tokens -->
      ${flow("M364 256 C381 256 377 240 394 240")}${flow("M364 290 C381 290 377 302 394 302")}
      ${flow("M364 424 C381 424 377 410 394 410")}${flow("M364 440 C381 440 377 446 394 446")}
      ${grid(396, 228, 6, 2, 11, 3, clean("g"))}${v(492, 247, "o")}<text class="a-s" x="512" y="245">observation</text>
      ${grid(396, 290, 6, 2, 11, 3, noisy("g"))}${v(492, 309, "z")}<text class="a-s" x="512" y="307">future video</text>
      <rect x="396" y="404.5" width="11" height="11" rx="2.5" ${clean("b")}/>${v(492, 416, "s")}<text class="a-s" x="512" y="414">proprioception</text>
      ${grid(396, 440.5, 6, 1, 11, 3, noisy("b"))}${v(492, 452, "a")}<text class="a-s" x="512" y="450">32 actions</text>
      <path d="M606 224 q9 0 9 11 V329 q0 11 9 11 q-9 0 -9 11 V445 q0 11 -9 11" fill="none" stroke="${C.muted}" stroke-width="1.5"/>
      ${flow("M628 340 H646")}
      <text x="396" y="482" class="a-seq">x = [o; s; z; a]</text><text class="a-s" x="506" y="481">one token sequence</text>

      <!-- prompt to cross-attention -->
      ${flow("M278 87 H646")}
      <text class="a-s" x="462" y="78" text-anchor="middle">read by every token through cross-attention</text>

      <!-- hybrid DiT -->
      <rect x="648" y="28" width="276" height="560" rx="18" fill="#f4f8ea" stroke="#c5db95" stroke-width="1.5"/>
      <text x="786" y="62" text-anchor="middle" class="a-h">Hybrid DiT</text>
      <text x="786" y="83" text-anchor="middle" class="a-s">4.5B parameters · single tower</text>
      <rect x="672" y="104" width="228" height="238" rx="12" fill="#fff" stroke="#9aa08d" stroke-width="1.3" stroke-dasharray="5 4"/>
      <path d="M761 160 V172 M761 212 V224 M761 264 V280" stroke="${C.muted}" stroke-width="1.5"/>
      ${pill(120, "GDN", 0)}${pill(172, "GDN", 1)}${pill(224, "GDN", 2)}${pill(280, "Softmax", 3)}
      <text x="867" y="232" text-anchor="middle" class="a-x">×8</text>
      <text x="786" y="364" text-anchor="middle" class="a-s">24 GDN layers + 8 softmax layers</text>

      <rect x="672" y="380" width="228" height="88" rx="12" fill="#fff" stroke="${G.line}" stroke-width="1.5"/>
      <text x="688" y="404" class="a-tag">3D MROPE</text>
      ${grid(688, 414, 2, 2, 9, 2, clean("g"))}<text class="a-t" x="722" y="430">2 × 2 view grid in space</text>
      <path d="M688 451 H708 M690 447 V455 M696 447 V455 M702 447 V455" stroke="${G.b}" stroke-width="1.5"/><text class="a-t" x="722" y="455">one physical clock in time</text>

      <text x="786" y="508" text-anchor="middle" class="a-t a-b">Chunk-causal attention</text>
      <text x="786" y="528" text-anchor="middle" class="a-s">each chunk reads the memory</text>
      <text x="786" y="546" text-anchor="middle" class="a-s">of the past, never the future</text>

      <!-- memory -->
      ${card(16, 500, 604, 96, "MEMORY · SANA-ROBOT-AR")}
      <rect x="32" y="537" width="104" height="24" rx="12" fill="${G.bs}" stroke="${G.b}"/><text x="84" y="553.5" text-anchor="middle" class="a-s a-b" style="fill:${G.bd}">Softmax layers</text>
      ${img("arch_mem0.jpg", 148, 536, 42, 26)}<rect x="148" y="536" width="42" height="26" rx="6" fill="none" stroke="${C.ink}" stroke-width="2"/>
      <text x="198" y="554" class="a-s">···</text>
      ${[0, 1, 2].map((i) => img(`arch_mem${i + 1}.jpg`, 220 + i * 46, 536, 42, 26) + `<rect x="${220 + i * 46}" y="536" width="42" height="26" rx="6" fill="none" stroke="${G.b}" stroke-width="2"/>`).join("")}
      <text class="a-xs" x="148" y="582">first frame + last N chunks, exact</text>
      <rect x="372" y="537" width="64" height="24" rx="12" fill="${G.gs}" stroke="${G.g}"/><text x="404" y="553.5" text-anchor="middle" class="a-s a-b" style="fill:${G.gd}">GDN</text>
      <rect x="444" y="536" width="150" height="26" rx="6" fill="${G.gs}" stroke="${G.g}" stroke-width="1.5"/>${state}
      <text class="a-xs" x="444" y="582">state S: whole episode, fixed size</text>
      ${flow("M620 548 H646")}

      <!-- outputs -->
      <text class="a-t a-b" x="952" y="150">Flow matching</text>
      <text class="a-s" x="952" y="170">video and actions are denoised</text><text class="a-s" x="952" y="188">jointly in K steps</text>
      ${flow("M924 300 C939 300 935 264 950 264")}${flow("M924 380 C939 380 935 398 950 398")}
      ${card(952, 206, 232, 116, "FUTURE VIDEO")}
      ${grid(968, 244, 6, 2, 13, 3, clean("g"))}${v(1074, 266, "z")}<text class="a-s" x="1094" y="264">predicted</text>
      <text class="a-xs" x="968" y="306">frames of the next 32 steps</text>
      ${card(952, 338, 232, 116, "ACTIONS")}
      ${grid(968, 380, 6, 1, 13, 3, clean("b"))}${v(1074, 394, "a")}<text class="a-s" x="1094" y="392">to the robot</text>
      <path d="M968 420 C1000 404 1016 436 1048 422 S1104 404 1136 418" fill="none" stroke="${G.b}" stroke-width="2" stroke-linecap="round" class="a-traj"/>
      <text class="a-xs" x="968" y="444">one chunk of 32 actions</text>

      <rect x="952" y="490" width="13" height="13" rx="2.5" ${clean("g")}/><text class="a-s" x="974" y="501">clean token: condition</text>
      <rect x="952" y="514" width="13" height="13" rx="2.5" ${noisy("g")}/><text class="a-s" x="974" y="525">noised token: target</text>
      <text class="a-xs" x="952" y="556">Policy mode shown. FDM and IDM use</text><text class="a-xs" x="952" y="573">the same network, see design 2.</text>
    </svg>`;
  }

  /* ───────── direct RTC against memory RTC, animated over the control steps ───────── */
  function rtc() {
    const Cn = S.chunk.actions;                                  // C = 32 actions per chunk
    const X0 = -Cn, X1 = 4 * Cn;                                 // one chunk of padding, then four chunks of episode
    const LAT = 5.25;                                            // one request: 175.0 ms (BF16, RTX PRO 6000) at 30 Hz
    const SPEED = 9;                                             // control steps per second of animation
    const st = { d: 16, t: 0, playing: !reduced, visible: false, last: 0, hold: 0, key: "" };
    const W = 1000, LBL = 186, PADR = 14, RH = 28;
    const X = (v) => LBL + ((v - X0) / (X1 - X0)) * (W - LBL - PADR);
    const gcd = (a, b) => (b ? gcd(b, a % b) : a);
    const K = { exec: "#76b900", soft: "#d6e9ab", dark: "#3f6400", direct: "#e58a89", directNew: "#c9403f", mem: "#2a78d6", memSoft: "#c9dbf3", grid: "#e2e5da" };
    const box = $("#rtcChart");
    let L = null;                                                // layout of the current d

    const rect = (a, b, y, fill, extra) => `<rect x="${X(a).toFixed(1)}" y="${y}" width="${Math.max(0, X(b) - X(a) - 1.5).toFixed(1)}" height="${RH}" rx="4" fill="${fill}" ${extra || ""}/>`;
    const text = (x, y, str, extra) => `<text x="${x.toFixed(1)}" y="${y}" font-size="13" fill="${C.muted}" ${extra || ""}>${str}</text>`;
    const inside = (a, b, y, long, short, fill, extra) => {
      const w = X(b) - X(a), str = w >= long.length * 7.4 + 10 ? long : w >= short.length * 7.4 + 6 ? short : "";
      return str ? text((X(a) + X(b)) / 2, y + RH / 2 + 4.5, str, `text-anchor="middle" font-weight="700" fill="${fill}" ${extra || ""}`) : "";
    };
    const label = (y, title, sub, strong) =>
      text(LBL - 14, y + (sub ? 12 : 18), title, `text-anchor="end" font-size="14" font-weight="700" fill="${strong === false ? C.muted : C.ink}"`) +
      (sub ? text(LBL - 14, y + 26, sub, `text-anchor="end" font-size="12"`) : "");
    const fade = (age) => Math.max(0, Math.min(1, age / 0.8)).toFixed(2);

    function build() {
      const d = st.d, P = Cn - d;
      const phases = Array.from(new Set(Array.from({ length: Cn / gcd(Cn, d) }, (_, i) => (i * P) % Cn))).sort((a, b) => a - b);
      const yObs = 46, yReq = 70, yExec = yReq + RH + 10, yDir = yExec + RH + 36, yMem = yDir + RH + 36;
      const yAxis = yMem + phases.length * (RH + 8) + 4, H = yAxis + 44;
      L = { d, P, phases, K: Math.floor((X1 - Cn) / P), yObs, yReq, yExec, yDir, yMem, yAxis, H };
      let g = `<defs><pattern id="rtcHatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width="7" height="7" fill="#eef0e8"/><rect width="3" height="7" fill="#cfd3c4"/></pattern></defs>`;
      for (let v = X0; v <= X1; v += 16) {
        g += `<line x1="${X(v)}" x2="${X(v)}" y1="${yObs - 4}" y2="${yAxis}" stroke="${K.grid}" stroke-width="${v % Cn === 0 ? 1.5 : 1}"/>`;
        g += text(X(v), yAxis + 17, v < 0 ? "−" + -v : v, `text-anchor="middle" font-size="12"`);
      }
      g += `<line x1="${X(X0)}" x2="${X(X1)}" y1="${yAxis}" y2="${yAxis}" stroke="${C.gray}"/>`;
      g += text(X(X1), yAxis + 36, "control step", `text-anchor="end" font-size="12.5"`);
      g += text(X(-Cn / 2), yAxis + 36, "before the episode: copies of the first frame", `text-anchor="middle" font-size="12"`);
      g += label(yObs - 8, "Observed video");
      g += label(yReq, "Requests", `first d = ${d} dropped`);
      g += label(yExec, "Robot executes", `C − d = ${P} per chunk`);
      g += text(LBL, yDir - 10, `(a) Direct RTC: one memory, chunks of ${P} steps`, `font-weight="700" fill="#a32a29" font-size="13.5"`);
      g += text(LBL, yMem - 10, `(b) Memory RTC: ${phases.length} memories, chunks of ${Cn} steps as in training`, `font-weight="700" fill="#1c5fb0" font-size="13.5"`);
      box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" style="touch-action:pan-y;cursor:ew-resize"
        aria-label="Animation of asynchronous execution with C = ${Cn} and d = ${d}: the memory written and read under direct RTC and under memory RTC">${g}<g id="rtcDyn"></g></svg>`;
      $$("#rtcD button").forEach((b) => b.setAttribute("aria-pressed", +b.dataset.d === d));
      st.key = "";
    }

    function render() {
      const { d, P, phases, yObs, yReq, yExec, yDir, yMem, yAxis } = L, t = st.t;
      const k = Math.min(Math.floor(t / P), Math.floor((X1 - 1) / P)), tk = k * P, phase = tk % Cn;
      const reading = t - tk <= LAT;                             // request k is running
      let g = "";

      // camera stream observed so far
      g += `<rect x="${X(0)}" y="${yObs}" width="${Math.max(0, X(t) - X(0))}" height="12" rx="3" fill="#cdd3bf"/>`;

      // requests: the first d actions of each chunk are dropped while its request runs
      for (let j = 0; j <= k; j++) {
        const tj = j * P, on = j === k;
        g += rect(tj, Math.min(tj + d, X1), yReq, "url(#rtcHatch)", `opacity="${on ? 1 : 0.6}"`);
        g += `<rect x="${X(tj)}" y="${yReq + 8}" width="${Math.max(0, X(Math.min(t, tj + LAT)) - X(tj))}" height="${RH - 16}" rx="3" fill="${K.mem}" opacity="${on ? 1 : 0.45}"/>`;
      }
      for (let j = 0; j * P <= X1 - 1; j++) {
        const x = X(j * P), on = j === k, r = on ? 7 : 4.5, y = yObs - 5;
        g += `<path d="M${x - r} ${y - r * 1.5} L${x + r} ${y - r * 1.5} L${x} ${y} Z" fill="${on ? C.ink : j < k ? C.muted : "#c9cdc2"}"/>`;
      }
      g += text(Math.min(Math.max(X(tk), LBL + 70), W - 80), 14, `request ${k} · step ${tk}`, `text-anchor="middle" font-weight="700" fill="${C.ink}"`);

      // execution: chunk j arrives when its request finishes and runs from t_j + d to t_j + C
      for (let j = 0; j <= k; j++) {
        const a = j * P + d, b = Math.min(j * P + Cn, X1);
        if (t < j * P + LAT || a >= X1) continue;
        g += rect(a, b, yExec, K.soft, j === k ? `stroke="${K.dark}" stroke-width="1.5"` : "");
        if (t > a) g += rect(a, Math.min(t, b), yExec, K.exec);
        g += inside(a, b, yExec, "chunk " + j, String(j), t > (a + b) / 2 ? "#fff" : K.dark);
      }

      // (a) direct RTC: request j writes the C - d steps executed since request j - 1
      g += label(yDir, "Memory", "read by request " + k);
      for (let j = 1; j <= k; j++) {
        const age = t - j * P;
        g += `<g opacity="${fade(age)}">` + rect((j - 1) * P, j * P, yDir, age < 3 ? K.directNew : K.direct) + inside((j - 1) * P, j * P, yDir, String(P), "", "#fff") + "</g>";
      }
      if (!k) g += text(X(0) + 8, yDir + 18, "only the first frame");
      if (reading && k) g += `<rect x="${X(0) - 3}" y="${yDir - 3}" width="${X(tk) - X(0) + 4.5}" height="${RH + 6}" rx="6" fill="none" stroke="#a32a29" stroke-width="1.5" stroke-dasharray="5 4"/>`;

      // (b) memory RTC: the window that ends at step e goes into the memory of phase e mod C
      phases.forEach((ph, i) => {
        const y = yMem + i * (RH + 8), on = ph === phase;
        g += label(y, `φ = ${ph}`, on ? "read by request " + k : "not read now", on);
        let first = null;
        for (let e = ph || Cn; e <= t; e += Cn) {
          const age = t - e, fresh = age < 3;
          if (first === null) first = e - Cn;
          g += `<g opacity="${fade(age)}">`;
          if (e - Cn < 0) g += rect(e - Cn, 0, y, "url(#rtcHatch)", `opacity="${on ? 1 : 0.55}"`) + inside(e - Cn, 0, y, "pad", "", C.muted);
          g += rect(Math.max(0, e - Cn), e, y, on ? K.mem : K.memSoft, fresh ? `stroke="#1c5fb0" stroke-width="2"` : "");
          g += inside(Math.max(0, e - Cn), e, y, e - Cn < 0 ? `0–${e}` : String(Cn), "", on ? "#fff" : "#5a7cab") + "</g>";
        }
        if (first === null && on) g += text(X(0) + 8, y + 18, "only the first frame");
        if (reading && on && first !== null) g += `<rect x="${X(first) - 3}" y="${y - 3}" width="${X(tk) - X(first) + 4.5}" height="${RH + 6}" rx="6" fill="none" stroke="#1c5fb0" stroke-width="1.5" stroke-dasharray="5 4"/>`;
      });

      // playhead
      g += `<line x1="${X(t)}" x2="${X(t)}" y1="${yObs - 6}" y2="${yAxis}" stroke="${C.ink}" stroke-width="1.5"/><circle cx="${X(t)}" cy="${yAxis}" r="4" fill="${C.ink}"/>`;
      $("#rtcDyn").innerHTML = g;

      const key = d + "|" + k;
      if (key !== st.key) {
        st.key = key;
        $("#rtcK").textContent = `Request ${k} of ${L.K}`;
        const e = S.asyncError, i = e.d.indexOf(d);
        $("#rtcVerdicts").innerHTML =
          `<div class="verdict bad"><div class="vnum"><b>${e.direct[i].toFixed(2)}×</b><span>position error of synchronous execution</span></div>
             <div><h4>Direct RTC</h4><p>One memory. Each request writes the ${P} steps executed since the previous request as one memory chunk, so the memory chunks span ${P} steps instead of the ${Cn} steps of training.</p></div></div>
           <div class="verdict good"><div class="vnum"><b>${e.memory[i].toFixed(2)}×</b><span>position error of synchronous execution</span></div>
             <div><h4>Memory RTC</h4><p>${phases.length} memories, one per phase. Request ${k} starts at step ${tk} and reads the memory of phase φ = ${tk} mod ${Cn} = ${phase}, whose chunks all span ${Cn} steps and end at step ${tk}, as in training.</p></div></div>`;
      }
    }

    function loop(now) {
      if (!st.playing || !st.visible) { st.last = 0; return; }
      const dt = st.last ? Math.min(0.1, (now - st.last) / 1000) : 0;
      st.last = now;
      if (st.hold > 0) { st.hold -= dt; if (st.hold <= 0) st.t = 0; }
      else { st.t += dt * SPEED; if (st.t >= X1 - 0.5) { st.t = X1 - 0.5; st.hold = 1.6; } }
      render();
      requestAnimationFrame(loop);
    }
    const play = (on) => {
      st.playing = on; st.hold = 0;
      $("#rtcPlay").textContent = on ? "❚❚" : "▶";
      $("#rtcPlay").setAttribute("aria-label", on ? "Pause the animation" : "Play the animation");
      if (on && st.visible) requestAnimationFrame(loop);
    };
    const jump = (k) => { play(false); st.t = Math.max(0, Math.min(k, L.K)) * L.P + 2.6; render(); };

    S.asyncError.d.forEach((d) => {
      const b = el("button", "", String(d));
      b.type = "button"; b.dataset.d = d;
      b.onclick = () => { st.d = d; st.t = 0; st.hold = 0; build(); render(); };
      $("#rtcD").append(b);
    });
    $("#rtcPlay").onclick = () => play(!st.playing);
    $("#rtcPrev").onclick = () => jump(Math.floor(st.t / L.P) - 1 < 0 ? L.K : Math.floor(st.t / L.P) - 1);
    $("#rtcNext").onclick = () => jump(Math.floor(st.t / L.P) + 1 > L.K ? 0 : Math.floor(st.t / L.P) + 1);
    let drag = false;
    const scrub = (ev) => {
      const svgEl = $("svg", box), r = svgEl.getBoundingClientRect(), x = ((ev.clientX - r.left) / r.width) * W;
      st.t = Math.max(0, Math.min(X1 - 0.5, X0 + ((x - LBL) / (W - LBL - PADR)) * (X1 - X0)));
      render();
    };
    box.addEventListener("pointerdown", (ev) => { drag = true; play(false); scrub(ev); });
    window.addEventListener("pointermove", (ev) => drag && scrub(ev));
    window.addEventListener("pointerup", () => (drag = false));
    new IntersectionObserver((en) => {
      st.visible = en[0].isIntersecting;
      if (st.visible && st.playing) requestAnimationFrame(loop);
    }, { threshold: 0.25 }).observe($("#rtc"));

    build();
    if (reduced) { st.t = 3 * L.P + 2.6; }
    play(st.playing); render();
  }

  /* ───────── videos: play in view, pause out of view ───────── */
  function videos() {
    const vids = $$("video[data-autoplay]");
    if (reduced) { vids.forEach((v) => (v.controls = true)); return; }
    const io = new IntersectionObserver((entries) => entries.forEach((e) => {
      const v = e.target;
      if (e.isIntersecting && !v.dataset.userPaused) v.play().catch(() => {});
      else if (!e.isIntersecting && !v.paused) { if (v.controls) v.dataset.autoPaused = "1"; v.pause(); }
    }), { threshold: 0.4 });
    vids.forEach((v) => {
      io.observe(v);
      if (v.controls) {
        // A video with its own controls (the overview video): a pause through them is the viewer's and is kept when
        // the video comes back into view; a pause because it scrolled out of view is not.
        v.addEventListener("pause", () => { if (v.dataset.autoPaused) delete v.dataset.autoPaused; else if (!v.ended) v.dataset.userPaused = "1"; });
        v.addEventListener("play", () => { delete v.dataset.userPaused; delete v.dataset.autoPaused; });
      } else if (v.id !== "memVideo") v.addEventListener("click", () => {
        if (v.paused) { delete v.dataset.userPaused; v.play(); } else { v.dataset.userPaused = "1"; v.pause(); }
      });
    });
  }

  /* ───────── nav highlight, BibTeX copy ───────── */
  function chrome() {
    const links = $$(".nav nav a");
    const io = new IntersectionObserver((entries) => entries.forEach((e) => {
      if (e.isIntersecting) links.forEach((a) => a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id));
    }), { rootMargin: "-45% 0px -50% 0px" });
    $$("main section[id]").forEach((s) => io.observe(s));
    $("#copyBib").onclick = () => navigator.clipboard.writeText($("#bibText").textContent).then(() => {
      $("#copyBib").textContent = "Copied"; setTimeout(() => ($("#copyBib").textContent = "Copy"), 1500);
    });
  }

  benchSetup(); tradeoff(); droid(); latency(); lanes(); arch(); rtc(); unified(); memlab(); videos(); chrome();
})();
