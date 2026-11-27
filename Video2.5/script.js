(() => {
  "use strict";
  const D = window.SV25 || { long: [], t2av: [], img: "media/img/" };
  const IMG = D.img || "media/img/";
  const pic = (n) => (D.imgs && D.imgs[n]) || IMG + n;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (s, r = document) => r.querySelector(s);
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };

  /* ---------------------------------------------------------- nav */
  const nav = $("[data-nav]");
  const onScroll = () => nav && nav.classList.toggle("solid", window.scrollY > 30);
  window.addEventListener("scroll", onScroll, { passive: true }); onScroll();

  /* ---------------------------------------------------------- hero: editing timeline (static when motion is reduced) */
  if (reduce) document.querySelectorAll(".hero-timeline animate, .hero-timeline animateTransform").forEach((n) => n.remove());

  /* ---------------------------------------------------------- theme: follows the system; the nav button cycles auto, light, dark.
     The SVG art is drawn with the dark palette; tone() maps those colours to light-theme equivalents. */
  const root = document.documentElement;
  const mq = window.matchMedia("(prefers-color-scheme: light)");
  const isLight = () => (root.dataset.theme ? root.dataset.theme === "light" : mq.matches);
  const LIGHT = {
    "#eef2ec": "#141a14", "#98a196": "#596257", "#677064": "#6c7569", "#3a403b": "#c6ccc3", "#a3e635": "#437800",
    "#ffffff": "#101812", "#cfc8ff": "#4b3fc0", "#b9adff": "#6d5fe6", "#d9ff8a": "#5e9e00", "#ffc2cf": "#d24a66",
    "#ffd27a": "#c98500", "#1d1a0f": "#fdf5e4", "#f6cd83": "#8f5d06", "#4a514b": "#b5bcb2", "#bfe0ff": "#24639c",
    "#8cc4ee": "#2a6aa3", "#231d0f": "#fdf3df", "#101a0c": "#eff7e3", "#7d857a": "#98a095", "#0b100c": "#ffffff",
    "#c9d1c6": "#626b60", "#0d130f": "#f2f5ef", "#d6f59a": "#5f9e12", "#050706": "#fafbf8", "#1c3b24": "#d3ebb5",
    "#3b2a12": "#f5dfb6", "#1d2244": "#d9d6fb",
  };
  const tone = (svg) => (isLight() ? svg.replace(/#[0-9a-fA-F]{6}\b/g, (h) => LIGHT[h.toLowerCase()] || h) : svg);
  const heroSvg = $(".hero-timeline"), heroSrc = heroSvg ? heroSvg.innerHTML : "";
  const redraw = [() => { if (heroSvg) heroSvg.innerHTML = tone(heroSrc); }];
  let drawnLight = false;
  const applyTheme = (force) => { const l = isLight(); if (force || l !== drawnLight) { drawnLight = l; redraw.forEach((f) => f()); } };
  const tbtn = $("[data-theme-btn]");
  const LABEL = { auto: "automatic (follows the system)", light: "light", dark: "dark" };
  const showMode = () => { if (!tbtn) return; const m = root.dataset.theme || "auto"; tbtn.dataset.mode = m; tbtn.setAttribute("aria-label", "Theme: " + LABEL[m]); tbtn.title = "Theme: " + (m === "auto" ? "automatic" : m); };
  tbtn && tbtn.addEventListener("click", () => {
    const next = { auto: "light", light: "dark", dark: "auto" }[root.dataset.theme || "auto"];
    if (next === "auto") delete root.dataset.theme; else root.dataset.theme = next;
    try { if (next === "auto") localStorage.removeItem("sv25-theme"); else localStorage.setItem("sv25-theme", next); } catch (e) {}
    showMode(); applyTheme();
  });
  mq.addEventListener ? mq.addEventListener("change", () => applyTheme()) : mq.addListener(() => applyTheme());
  showMode();

  /* ---------------------------------------------------------- hero: latency race (log scale) */
  const race = $("[data-race]");
  if (race) {
    const rows = [
      ["MiniMax-H3 · 50 steps", 647],
      ["H3-Turbo · 8 steps", 132],
      ["LTX-2.5 distilled · 8 steps", 20.8],
      ["Ours · 8 steps", 14.7, "ours"],
      ["+ Sol-Engine", 9.0, "ours"],
      ["+ 4 steps", 4.7, "ours best"],
    ];
    const lo = Math.log10(2.5), hi = Math.log10(1400);
    const pct = (x) => Math.max(2, ((Math.log10(x) - lo) / (hi - lo)) * 100);
    rows.forEach(([n, v, cls], i) => {
      const r = el("div", "race-row " + (cls || ""));
      r.innerHTML = `<span class="name"></span><span class="track"><span class="fill"></span><span class="val">${v}<small> s</small></span></span>`;
      r.querySelector(".name").textContent = n;
      r.style.setProperty("--w", pct(v).toFixed(2) + "%");
      r.style.setProperty("--d", (0.15 + i * 0.08).toFixed(2) + "s");
      race.appendChild(r);
    });
    const line = el("span", "rt-line"); race.appendChild(line);
    const place = () => {
      const t = race.querySelector(".track"); if (!t) return;
      const a = race.getBoundingClientRect(), b = t.getBoundingClientRect();
      line.style.left = (b.left - a.left + (b.width * pct(15)) / 100).toFixed(1) + "px";
    };
    place(); window.addEventListener("resize", place);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(place);
  }

  /* ---------------------------------------------------------- showcase: two clips per row, prompt on click */
  const grid = $("[data-vgrid]");
  const fmt = (t) => (isFinite(t) ? `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}` : "0:00");
  // never show more than one video pixel per screen pixel (sources are 1344x768)
  const lockSize = () => grid && grid.style.setProperty("--cw", Math.round(Math.min(590, 1344 / (window.devicePixelRatio || 1))) + "px");
  lockSize(); window.addEventListener("resize", lockSize);
  const ICON_PLAY = '<svg viewBox="0 0 24 24" aria-hidden="true"><path class="i-pause" d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z"/><path class="i-play" d="M8 5.5v13l11-6.5z"/></svg>';
  const ICON_SND = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z"/><path class="i-off" d="M16 9.5l5 5M21 9.5l-5 5"/><path class="i-on" d="M16 8.8a4.6 4.6 0 0 1 0 6.4M18.6 6.4a8 8 0 0 1 0 11.2"/></svg>';
  // One clip downloads at a time, in page order (user, 2026-11-10: load the videos one by one, front to back), so the
  // first clips get the whole bandwidth. A clip gets its src only when its turn comes. The next one starts once the
  // browser holds the whole clip, has stopped fetching with enough data to play through, or after 20 s.
  // Pressing play or sound on a later clip starts that clip at once.
  let queue = [], cur = null, timer = 0;
  const QEV = ["progress", "suspend", "canplaythrough", "error"];
  const whole = (v) => { const b = v.buffered, d = v.duration; return d > 0 && b.length > 0 && b.end(b.length - 1) >= d - 0.3; };
  const start = (v) => { if (v.dataset.src && !v.getAttribute("src")) { v.preload = "auto"; v.src = v.dataset.src; } };
  const pump = () => {
    if (cur) return;
    while (queue.length && queue[0].getAttribute("src")) queue.shift();
    const v = queue.shift(); if (!v) return;
    cur = v; start(v);
    const done = (e) => {
      if (cur !== v) return;
      if (e && e.type !== "error" && !whole(v) && !(e.type === "suspend" && v.readyState >= 4)) return;
      QEV.forEach((t) => v.removeEventListener(t, done));
      clearTimeout(timer); cur = null; pump();
    };
    QEV.forEach((t) => v.addEventListener(t, done));
    timer = setTimeout(done, 20000);
    if (v.dataset.vis && !reduce && !v.dataset.userPaused) v.play().catch(() => {});
  };
  const vio = "IntersectionObserver" in window ? new IntersectionObserver((es) => es.forEach((e) => {
    const v = e.target.querySelector("video"); if (!v) return;
    if (e.isIntersecting) {
      v.dataset.vis = "1";
      if (!reduce && !v.dataset.userPaused && v.getAttribute("src")) v.play().catch(() => {});
    } else { delete v.dataset.vis; v.pause(); }
  }), { threshold: 0.35 }) : null;
  const makeCard = (c, i) => {
    const a = el("article", "vcard");
    a.innerHTML = `<div class="vframe"><video muted loop playsinline preload="none"></video><span class="vtag">Prompt</span>
      <div class="vprompt" hidden><p class="label">Prompt · click to close</p><div class="vprompt-text"></div></div>
      <div class="controls"><button class="ctl-play paused" type="button" aria-label="Play">${ICON_PLAY}</button>
      <div class="ctl-bar"><div class="ctl-buf"></div><div class="ctl-fill"></div><input type="range" min="0" max="1000" step="1" value="0" aria-label="Seek"></div>
      <span class="ctl-time">0:00 / 0:00</span><button class="ctl-snd" type="button" aria-pressed="false" aria-label="Turn sound on">${ICON_SND}</button></div></div>
      <div class="vcap"><b></b><span></span></div>`;
    const v = a.querySelector("video"), frame = a.querySelector(".vframe"), pr = a.querySelector(".vprompt");
    const play = a.querySelector(".ctl-play"), seek = a.querySelector("input"), fill = a.querySelector(".ctl-fill"), buf = a.querySelector(".ctl-buf"), tm = a.querySelector(".ctl-time"), snd = a.querySelector(".ctl-snd");
    v.poster = c.poster; v.dataset.src = c.mp4;  // src is set by the sequential loader (pump) or by a user play
    a.querySelector(".vprompt-text").textContent = c.prompt;
    a.querySelector(".vcap b").textContent = c.title;
    a.querySelector(".vcap span").textContent = c.meta;
    let seeking = false;
    const paint = () => {
      const d = v.duration || 0, t = v.currentTime || 0, p = d ? t / d : 0;
      fill.style.width = (p * 100).toFixed(2) + "%";
      if (!seeking) seek.value = Math.round(p * 1000);
      tm.textContent = `${fmt(t)} / ${fmt(d)}`;
      if (v.buffered && v.buffered.length && d) buf.style.width = ((v.buffered.end(v.buffered.length - 1) / d) * 100).toFixed(1) + "%";
    };
    const loop = () => { paint(); if (!v.paused) requestAnimationFrame(loop); };
    const sync = () => { play.classList.toggle("paused", v.paused); play.setAttribute("aria-label", v.paused ? "Play" : "Pause"); };
    v.addEventListener("play", () => { sync(); requestAnimationFrame(loop); });
    v.addEventListener("pause", () => { sync(); paint(); });
    ["loadedmetadata", "timeupdate", "progress"].forEach((e) => v.addEventListener(e, paint));
    a.querySelector(".controls").addEventListener("click", (e) => e.stopPropagation());
    play.addEventListener("click", () => { if (v.paused) { delete v.dataset.userPaused; start(v); v.play().catch(() => {}); } else { v.dataset.userPaused = "1"; v.pause(); } });
    seek.addEventListener("input", () => { seeking = true; if (v.duration) { v.currentTime = (seek.value / 1000) * v.duration; paint(); } });
    seek.addEventListener("change", () => { seeking = false; });
    snd.addEventListener("click", () => {
      const on = v.muted;
      if (on) grid.querySelectorAll(".vcard").forEach((o) => { if (o !== a) { const ov = o.querySelector("video"); ov.muted = true; o.querySelector(".ctl-snd").setAttribute("aria-pressed", "false"); } });
      v.muted = !on; snd.setAttribute("aria-pressed", String(on)); snd.setAttribute("aria-label", on ? "Turn sound off" : "Turn sound on");
      if (on) { delete v.dataset.userPaused; start(v); v.play().catch(() => {}); }
    });
    frame.addEventListener("click", () => { pr.hidden = !pr.hidden; });
    if (vio) vio.observe(a); else if (i < 2) v.dataset.vis = "1";
    return a;
  };
  const fillGrid = (set) => {
    if (!grid) return;
    grid.querySelectorAll(".vcard").forEach((o) => { if (vio) vio.unobserve(o); const ov = o.querySelector("video"); ov.pause(); ov.removeAttribute("src"); ov.load(); });
    grid.innerHTML = "";
    (D[set] || []).forEach((c, i) => grid.appendChild(makeCard(c, i)));
    clearTimeout(timer); cur = null;  // a tab switch restarts the queue with this tab's clips, first to last
    queue = Array.from(grid.querySelectorAll("video"));
    pump();
  };
  document.querySelectorAll("[data-tabs] button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("[data-tabs] button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
    fillGrid(b.dataset.set);
  }));
  const first = document.querySelector('[data-tabs] [aria-selected="true"]');
  fillGrid(first ? first.dataset.set : "long");

  /* ---------------------------------------------------------- SVG diagrams */
  const C = { g: "#76b900", g2: "#a3e635", r: "#e0607a", v: "#9b8cff", a: "#f0b54a", b: "#5aa9e6", t: "#eef2ec", m: "#98a196", d: "#677064", line: "#3a403b" };
  const defs = `<defs>
    <marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="${C.m}"/></marker>
    <marker id="ahg" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="${C.g2}"/></marker>
    <marker id="aha" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,1 L9,5 L0,9 z" fill="${C.a}"/></marker>
    <pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="3" height="6" fill="#a3e635" fill-opacity="0.55"/></pattern>
    <linearGradient id="gS" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#8fd61a"/><stop offset="1" stop-color="#4f8a00"/></linearGradient>
    <linearGradient id="gA" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ee7d93"/><stop offset="1" stop-color="#a83a52"/></linearGradient>
  </defs>`;
  const col = (c) => { const m = /^rgba\((\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\)$/.exec(c); if (!m) return [c, 1]; return ["#" + [m[1], m[2], m[3]].map((n) => (+n).toString(16).padStart(2, "0")).join(""), +m[4]]; };
  const box = (x, y, w, h, fill, stroke, rx = 9) => { const [f, fo] = col(fill), [k, ko] = col(stroke); return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${f}" fill-opacity="${fo}" stroke="${k}" stroke-opacity="${ko}" stroke-width="1.2"/>`; };
  const txt = (x, y, s, o = {}) => `<text x="${x}" y="${y}" fill="${o.c || C.t}" font-size="${o.s || 12}" font-weight="${o.w || 500}" text-anchor="${o.a || "middle"}"${o.f ? ` font-family="${o.f}"` : ""}>${s}</text>`;

  /* SMIL helpers: every animation loops on a shared period so the diagrams stay in sync; nothing animates when motion is reduced. */
  const ANIM = !reduce;
  const r3 = (x) => +x.toFixed(4);
  const anim = (attr, values, times, T) => ANIM ? `<animate attributeName="${attr}" values="${values.join(";")}" keyTimes="${times.map((t) => r3(Math.min(Math.max(t / T, 0), 1))).join(";")}" dur="${T}s" repeatCount="indefinite"/>` : "";
  // opacity window: off, ramp up at t0, hold, ramp down
  const flash = (t0, T, peak = 1, rise = 0.18, hold = 0.25, fall = 0.45) => anim("opacity", [0, 0, peak, peak, 0, 0], [0, t0, t0 + rise, t0 + rise + hold, t0 + rise + hold + fall, T], T);
  const particle = (path, t0, t1, T, color, r = 3.4) => ANIM ? `<g opacity="0"><circle r="${r * 2.6}" fill="${color}" fill-opacity="0.22"/><circle r="${r}" fill="${color}"/>` +
    `<animateMotion path="${path}" keyPoints="0;0;1;1" keyTimes="0;${r3(t0 / T)};${r3(t1 / T)};1" calcMode="linear" dur="${T}s" repeatCount="indefinite"/>` +
    anim("opacity", [0, 0, 1, 1, 0, 0], [0, t0, t0 + 0.06, t1 - 0.06, t1, T], T) + `</g>` : "";

  const arch = () => {
    const T = 5.2, x0 = 150, w = 46, gap = 9, n = 8, vy = 92, ay = 214, h = 46;
    const at = (i) => 0.3 + i * 0.5;                       // layer i is active at this time in the cycle
    let s = `<svg viewBox="0 0 680 330" role="img" aria-label="A 14B hybrid video stream of linear and softmax layers beside a 2B softmax audio stream. Layer by layer, the two streams exchange information through time-aligned cross-attention, and both read Qwen3-VL text features.">${defs}`;
    s += box(150, 14, 380, 34, "rgba(155,140,255,0.12)", "rgba(155,140,255,0.5)") + txt(340, 36, "Qwen3-VL-4B text features · cross-attention in every layer", { c: "#cfc8ff", s: 12 });
    s += txt(18, vy + 20, "Video", { a: "start", s: 15, w: 700 }) + txt(18, vy + 38, "14B · hybrid", { a: "start", s: 11, c: C.m });
    s += txt(18, ay + 20, "Audio", { a: "start", s: 15, w: 700 }) + txt(18, ay + 38, "2B · softmax", { a: "start", s: 11, c: C.m });
    let fx = "";
    for (let i = 0; i < n; i++) {
      const x = x0 + i * (w + gap), cx = x + w / 2, soft = i % 4 === 3, t = at(i);
      // text cross-attention feed (dim line, bright pulse when the layer is active)
      s += `<line x1="${cx}" y1="48" x2="${cx}" y2="${vy}" stroke="#9b8cff" stroke-opacity="0.3" stroke-dasharray="2 3"/>`;
      fx += particle(`M${cx},50 V${vy - 2}`, t - 0.28, t, T, "#b9adff", 2.4);
      // blocks
      s += soft ? box(x, vy, w, h, "#76b900", C.g2) + txt(cx, vy + 28, "S", { s: 13, w: 700, c: "#061003" })
                : box(x, vy, w, h, "rgba(118,185,0,0.10)", "rgba(118,185,0,0.6)") + txt(cx, vy + 28, "L", { s: 13, w: 700, c: C.g2 });
      s += box(x, ay, w, h, "#d9566f", "#f29aac") + txt(cx, ay + 28, "S", { s: 13, w: 700, c: "#2a0710" });
      // activation glow on both streams at the same depth (time-aligned)
      if (ANIM) {
        fx += `<rect x="${x - 3}" y="${vy - 3}" width="${w + 6}" height="${h + 6}" rx="11" fill="none" stroke="${soft ? "#d9ff8a" : C.g2}" stroke-width="2" opacity="0">${flash(t, T)}</rect>`;
        fx += `<rect x="${x}" y="${vy}" width="${w}" height="${h}" rx="9" fill="#fffffe" opacity="0">${flash(t, T, soft ? 0.32 : 0.16)}</rect>`;
        fx += `<rect x="${x - 3}" y="${ay - 3}" width="${w + 6}" height="${h + 6}" rx="11" fill="none" stroke="#ffc2cf" stroke-width="2" opacity="0">${flash(t, T)}</rect>`;
      }
      // exchange: thin rails plus particles crossing both ways while the layer is active
      s += `<line x1="${cx - 6}" y1="${vy + h + 4}" x2="${cx - 6}" y2="${ay - 4}" stroke="${C.a}" stroke-opacity="0.35" stroke-width="1.4"/>`;
      s += `<line x1="${cx + 6}" y1="${ay - 4}" x2="${cx + 6}" y2="${vy + h + 4}" stroke="${C.a}" stroke-opacity="0.35" stroke-width="1.4"/>`;
      fx += particle(`M${cx - 6},${vy + h + 4} V${ay - 4}`, t + 0.05, t + 0.5, T, "#ffd27a", 2.8);
      fx += particle(`M${cx + 6},${ay - 4} V${vy + h + 4}`, t + 0.05, t + 0.5, T, "#ffd27a", 2.8);
      if (i < n - 1) {
        s += `<line x1="${x + w}" y1="${vy + h / 2}" x2="${x + w + gap}" y2="${vy + h / 2}" stroke="${C.line}"/>`;
        s += `<line x1="${x + w}" y1="${ay + h / 2}" x2="${x + w + gap}" y2="${ay + h / 2}" stroke="${C.line}"/>`;
      }
    }
    const xe = x0 + n * (w + gap);
    s += txt(xe + 22, vy + 28, "× 5", { s: 13, c: C.m, w: 600 }) + txt(xe + 22, ay + 28, "× 5", { s: 13, c: C.m, w: 600 });
    s += box(185, 160, 300, 26, "#1d1a0f", "rgba(240,181,74,0.55)", 13) + txt(335, 177, "time-aligned audio ⇄ video cross-attention", { c: "#f6cd83", s: 11.5, w: 600 });
    s += fx;
    s += box(150, 290, 14, 14, "rgba(118,185,0,0.10)", "rgba(118,185,0,0.6)", 3) + txt(170, 301, "gated linear attention", { a: "start", s: 11, c: C.m });
    s += box(330, 290, 14, 14, "#76b900", C.g2, 3) + txt(350, 301, "softmax attention", { a: "start", s: 11, c: C.m });
    s += txt(480, 301, "40 layers per stream", { a: "start", s: 11, c: C.m });
    return s + "</svg>";
  };

  const stages = () => {
    const st = [
      ["0a", "Text adaptation", "Qwen3-VL features", [2, 0]],
      ["0b", "720p · 5–15 s", "keyframes", [2, 0]],
      ["1", "Audio bootstrap", "video frozen + LoRA", [1, 2]],
      ["2", "AV coupling", "5–15 s · 24 fps", [1, 2]],
      ["3", "Joint AV", "all 16B jointly", [2, 2]],
    ];
    const T = 11, at = (i) => 0.4 + i * 1.5, w = 118, gap = 13, y = 26, h = 150, bw = w - 66;
    let s = `<svg viewBox="0 0 680 300" role="img" aria-label="Five training stages light up in turn: text-encoder adaptation and resolution extension train the video model, audio is bootstrapped beside the frozen video model with a LoRA, coupled at 5 to 15 seconds, then all 16B parameters train jointly. The joint model then seeds the few-step student.">${defs}`;
    let fx = "";
    st.forEach(([k, t, sub, tr], i) => {
      const x = 8 + i * (w + gap), last = i === st.length - 1, t0 = at(i);
      s += box(x, y, w, h, last ? "rgba(118,185,0,0.10)" : "rgba(255,255,255,0.03)", last ? "rgba(118,185,0,0.6)" : C.line, 12);
      if (ANIM) {
        s += `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="12" fill="#a3e635" fill-opacity="0.08" stroke="#a3e635" stroke-width="1.6" opacity="0">${flash(t0, T, 1, 0.3, 0.9, 0.6)}</rect>`;
      }
      s += txt(x + 14, y + 26, k, { a: "start", s: 13, w: 700, c: C.g2, f: "JetBrains Mono, monospace" });
      s += txt(x + 14, y + 50, t, { a: "start", s: 12.5, w: 700 });
      s += txt(x + 14, y + 68, sub, { a: "start", s: 10.5, c: C.m });
      [["video", tr[0], C.g], ["audio", tr[1], C.r]].forEach(([lab, mode, colr], j) => {
        const by = y + 92 + j * 26;
        s += txt(x + 14, by + 10, lab, { a: "start", s: 10, c: C.d });
        s += `<rect x="${x + 52}" y="${by}" width="${bw}" height="12" rx="3" fill="none" stroke="${mode ? colr : "#4a514b"}" stroke-dasharray="${mode ? "0" : "3 3"}"/>`;
        if (mode) {
          const fillv = mode === 2 ? colr : "url(#hatch)";
          // the bar fills when its stage becomes active and stays until the cycle restarts
          s += `<rect x="${x + 52}" y="${by}" width="${bw}" height="12" rx="3" fill="${fillv}">${anim("width", [0, 0, bw, bw], [0, t0 + 0.15, t0 + 0.75, T], T)}</rect>`;
        }
      });
      if (!last) {
        s += `<line x1="${x + w + 1}" y1="${y + h / 2}" x2="${x + w + gap - 1}" y2="${y + h / 2}" stroke="${C.m}" marker-end="url(#ah)"/>`;
        fx += particle(`M${x + w - 6},${y + h / 2} H${x + w + gap + 6}`, t0 + 1.1, t0 + 1.5, T, C.g2, 2.6);
      }
    });
    const jx = 8 + 4 * (w + gap) + w / 2, tb = at(4) + 1.3;
    const pB = `M${jx},${y + h} V${y + h + 58}`;
    s += `<path d="${pB}" fill="none" stroke="${C.g2}" stroke-opacity="0.55" stroke-width="1.4" marker-end="url(#ahg)"/>`;
    fx += particle(pB, tb, tb + 0.6, T, C.g2, 3);
    s += box(471, y + h + 60, 200, 40, "rgba(90,169,230,0.10)", "rgba(90,169,230,0.5)") + txt(571, y + h + 85, "4 / 8-step student", { s: 12, w: 600, c: "#bfe0ff" });
    if (ANIM) fx += `<rect x="471" y="${y + h + 60}" width="200" height="40" rx="9" fill="none" stroke="#bfe0ff" stroke-width="1.8" opacity="0">${flash(tb + 0.6, T, 1, 0.2, 0.8, 0.6)}</rect>`;
    s += `<rect x="8" y="${y + h + 70}" width="12" height="9" rx="2" fill="${C.g}"/>` + txt(25, y + h + 79, "trained", { a: "start", s: 10, c: C.m });
    s += `<rect x="8" y="${y + h + 88}" width="12" height="9" rx="2" fill="url(#hatch)" stroke="${C.g}"/>` + txt(25, y + h + 97, "LoRA", { a: "start", s: 10, c: C.m });
    return s + fx + "</svg>";
  };

  const distill = () => {
    const T = 6.4;
    let s = `<svg viewBox="0 0 680 320" role="img" aria-label="Cross-latent distillation, animated: a student sample passes through the learned latent adapter into the teacher's latent space, the frozen H3 real score and an H3 LoRA fake score give the DMD update g, and the Gauss-Newton pullback carries the matching student move back to the student. A discriminator on frozen H3 features compares the adapted sample with real H3 videos, and its GAN loss also trains the student.">${defs}`;
    s += box(6, 50, 250, 250, "rgba(90,169,230,0.06)", "rgba(90,169,230,0.25)", 16) + txt(20, 288, "Student latent space · LTX-2.5 VAE", { a: "start", s: 11, c: "#8cc4ee", w: 600 });
    s += box(266, 50, 408, 250, "rgba(118,185,0,0.05)", "rgba(118,185,0,0.25)", 16) + txt(660, 288, "Teacher latent space · MiniMax-H3, 33B", { a: "end", s: 11, c: C.g2, w: 600 });
    const pulse = (x, y, w, h, c, t0, rx = 12) => ANIM ? `<rect x="${x - 3}" y="${y - 3}" width="${w + 6}" height="${h + 6}" rx="${rx + 2}" fill="none" stroke="${c}" stroke-width="2" opacity="0">${flash(t0, T, 1, 0.15, 0.25, 0.5)}</rect>` : "";
    s += box(26, 120, 120, 64, "rgba(90,169,230,0.14)", "rgba(90,169,230,0.7)", 12) + txt(86, 146, "Student", { s: 14, w: 700 }) + txt(86, 166, "4 / 8 steps", { s: 11, c: C.m });
    s += box(196, 112, 140, 80, "#231d0f", C.a, 14) + txt(266, 142, "Latent adapter 𝒯", { s: 13.5, w: 700, c: "#f6cd83" }) + txt(266, 162, "142M · learned", { s: 11, c: C.m }) + txt(266, 178, "bridge", { s: 11, c: C.m });
    s += `<line x1="146" y1="152" x2="194" y2="152" stroke="${C.m}" marker-end="url(#ah)"/>` + txt(170, 145, "x", { s: 11, c: C.m });
    s += box(388, 86, 128, 46, "rgba(118,185,0,0.12)", "rgba(118,185,0,0.6)", 10) + txt(452, 106, "Real score", { s: 12, w: 700 }) + txt(452, 122, "frozen H3", { s: 10.5, c: C.m });
    s += box(388, 172, 128, 46, "rgba(255,255,255,0.04)", C.line, 10) + txt(452, 192, "Fake score", { s: 12, w: 700 }) + txt(452, 208, "H3 + LoRA", { s: 10.5, c: C.m });
    s += `<path d="M336,152 H360 V109 H386" fill="none" stroke="${C.m}" marker-end="url(#ah)"/><path d="M360,152 V195 H386" fill="none" stroke="${C.m}" marker-end="url(#ah)"/>`;
    s += txt(344, 172, "𝒯(x)", { s: 10.5, c: C.m, a: "start" });
    s += box(550, 124, 108, 56, "rgba(118,185,0,0.16)", C.g2, 12) + txt(604, 148, "DMD update g", { s: 12.5, w: 700 }) + txt(604, 166, "teacher's correction", { s: 10, c: C.m });
    s += `<path d="M516,109 H533 V146 H548" fill="none" stroke="${C.m}" marker-end="url(#ah)"/><path d="M516,195 H533 V158" fill="none" stroke="${C.m}"/>`;
    const back = "M604,124 V24 H86 V118";
    s += `<path d="${back}" fill="none" stroke="${C.g2}" stroke-opacity="0.45" stroke-width="2" marker-end="url(#ahg)"/>`;
    if (ANIM) s += `<path d="${back}" fill="none" stroke="${C.g2}" stroke-width="2.4" stroke-dasharray="1000" stroke-dashoffset="1000">${anim("stroke-dashoffset", [1000, 1000, 0, 0, 1000], [0, 3.3, 5.0, 5.9, T], T)}</path>`;
    s += box(176, 10, 330, 28, "#101a0c", "rgba(163,230,53,0.6)", 14) + txt(341, 29, "Gauss–Newton pullback: the move 𝒯 maps onto g", { s: 12, w: 600, c: C.g2 });
    s += box(388, 236, 128, 36, "rgba(255,255,255,0.05)", "#7d857a", 10) + txt(452, 251, "Discriminator", { s: 11.5, w: 700 }) + txt(452, 265, "frozen H3 features", { s: 9.5, c: C.m });
    s += box(548, 236, 110, 36, "rgba(255,255,255,0.03)", C.line, 10) + txt(603, 251, "H3 videos", { s: 11, w: 600 }) + txt(603, 264, "real", { s: 9.5, c: C.m });
    s += `<path d="M360,195 V226 H398 V234" fill="none" stroke="${C.m}" marker-end="url(#ah)"/>`;
    s += `<line x1="548" y1="254" x2="518" y2="254" stroke="${C.m}" marker-end="url(#ah)"/>`;
    const ganLoss = "M388,254 H110 V186";
    s += `<path d="${ganLoss}" fill="none" stroke="#c9d1c6" stroke-opacity="0.55" stroke-dasharray="5 4" stroke-width="1.5" marker-end="url(#ah)"/>`;
    s += box(214, 245, 68, 18, "#0b100c", "rgba(201,209,198,0.35)", 9) + txt(248, 258, "GAN loss", { s: 10, c: "#c9d1c6", w: 600 });
    // forward pass: sample -> adapter -> both scores -> g ; backward: pullback to the student
    s += particle("M146,152 H196", 0.2, 0.8, T, "#bfe0ff", 3);
    s += pulse(196, 112, 140, 80, C.a, 0.8, 14);
    s += particle("M336,152 H360 V109 H388", 1.2, 1.9, T, "#f6cd83", 3) + particle("M336,152 H360 V195 H388", 1.2, 1.9, T, "#f6cd83", 3);
    s += particle("M336,152 H360 V226 H398 V236", 1.2, 1.9, T, "#f6cd83", 2.6) + particle("M548,254 H518", 1.3, 1.9, T, "#eef2ec", 2.6);
    s += pulse(388, 236, 128, 36, "#c9d1c6", 1.9, 10);
    s += pulse(388, 86, 128, 46, C.g2, 1.9, 10) + pulse(388, 172, 128, 46, "#c9d1c6", 1.9, 10);
    s += particle("M516,109 H533 V146 H550", 2.3, 2.9, T, C.g2, 3) + particle("M516,195 H533 V158 H550", 2.3, 2.9, T, "#c9d1c6", 3);
    s += pulse(550, 124, 108, 56, C.g2, 2.9, 12);
    s += particle(back, 3.3, 5.0, T, C.g2, 4.2);
    s += particle("M388,254 H110 V186", 3.3, 5.0, T, "#c9d1c6", 3);
    s += pulse(26, 120, 120, 64, "#bfe0ff", 5.0, 12);
    return s + "</svg>";
  };

  const dg = { arch, stages, distill };
  const drawDiagrams = () => document.querySelectorAll("[data-diagram]").forEach((d) => { const f = dg[d.dataset.diagram]; if (f) d.innerHTML = tone(f()); });
  redraw.push(drawDiagrams);
  drawDiagrams();
  applyTheme(isLight());

  /* ---------------------------------------------------------- efficiency bars */
  const series = [
    ["H3-Turbo · 8 steps", "var(--s0)", [28.3, 75.5, 132]],  // 10 s measured 11-09 (paper b1d941f, 243 frames)
    ["LTX-2.5 distilled · 8 steps", "var(--s1)", [5.3, 12.0, 20.8]],
    ["Ours · 8 steps", "var(--s2)", [5.0, 9.5, 14.7]],
    ["Ours + Sol-Engine", "var(--s3)", [2.6, 5.7, 9.0]],
    ["Ours · 4 steps + Sol-Engine", "var(--s4)", [1.4, 3.0, 4.7]],
  ];
  const bars = $("[data-bars]");
  if (bars) {
    // true proportions on one linear axis (user, 2026-11-10): the slowest bar sets the full height, nothing is cut
    const top = Math.max(...series.flatMap((s) => s[2]).filter((v) => v != null));
    ["5 s clip", "10 s clip", "15 s clip"].forEach((x, gi) => {
      const g = el("div", "bgroup"), cols = el("div", "bcols");
      series.forEach(([n, col, vals], si) => {
        const v = vals[gi];
        const b = el("div", "bcol" + (v == null ? " nm" : ""));
        b.style.background = col;
        b.style.setProperty("--h", v == null ? "8%" : Math.max(0.6, (v / top) * 92).toFixed(2) + "%");
        b.style.setProperty("--d", (si * 0.07).toFixed(2) + "s");
        b.title = `${n}: ${v == null ? "not measured" : v + " s"}`;
        b.appendChild(el("span", "v", v == null ? "n/m" : String(v)));
        cols.appendChild(b);
      });
      g.append(cols, el("div", "bx", x));
      bars.appendChild(g);
    });
    const lg = $("[data-legend]");
    series.forEach(([n, col]) => { const e = el("span", null, `<i style="background:${col}"></i>`); e.append(n); lg.appendChild(e); });
  }
  // replay bar and meter animations when they scroll into view (static and fully drawn if this never fires)
  if ("IntersectionObserver" in window && !reduce) {
    const io = new IntersectionObserver((es) => es.forEach((e) => {
      if (!e.isIntersecting) return;
      e.target.querySelectorAll(".bcol, .meter-track i").forEach((n) => { n.style.animation = "none"; void n.offsetWidth; n.style.animation = ""; });
      io.unobserve(e.target);
    }), { threshold: 0.3 });
    document.querySelectorAll("[data-bars], [data-meter]").forEach((n) => io.observe(n));
  }

  /* ---------------------------------------------------------- citation */
  const copy = $("[data-copy]");
  copy && copy.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText($("[data-bib]").textContent); copy.textContent = "Copied"; }
    catch (e) { copy.textContent = "Select to copy"; }
    setTimeout(() => (copy.textContent = "Copy"), 1600);
  });
})();
