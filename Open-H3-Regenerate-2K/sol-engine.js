// SoL-Engine blog behaviour: navigation state, teaser scroll, demo, synchronized comparisons,
// the resolution ladder, chart grow-in and bar tooltips. No dependencies.
(() => {
  const reduce = matchMedia('(prefers-reduced-motion: reduce)');

  // Highlight the nav link of the section in view.
  const links = [...document.querySelectorAll('nav a')];
  const targets = links.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver(es => es.forEach(e => {
      if (e.isIntersecting) links.forEach(a => a.toggleAttribute('aria-current', a.getAttribute('href') === '#' + e.target.id));
    }), { rootMargin: '-20% 0px -70% 0px' });
    targets.forEach(t => io.observe(t));
  }

  // Teaser card recedes as the article scrolls in.
  const card = document.querySelector('.teaser-card'), cue = document.querySelector('.intro-cue'), copy = document.querySelector('.hero-split .hero-copy');
  if (card) {
    const mq = matchMedia('(max-width:720px),(prefers-reduced-motion:reduce)'); let raf = 0;
    const apply = () => {
      raf = 0;
      if (mq.matches) { for (const el of [card, cue, copy]) if (el) el.style.transform = el.style.opacity = ''; return; }
      const h = card.offsetHeight || 1, p = Math.min(1, Math.max(0, scrollY / (h * .8))), e = p * p * (3 - 2 * p);
      card.style.transform = `scale(${1 - e * .08})`; card.style.opacity = String(1 - e * .55);
      if (cue) cue.style.opacity = String(Math.max(0, 1 - p * 3));
      if (copy) copy.style.opacity = String(.35 + .65 * Math.min(1, p * 1.6));
    };
    const req = () => { if (!raf) raf = requestAnimationFrame(apply); };
    addEventListener('scroll', req, { passive: true }); addEventListener('resize', req); mq.addEventListener?.('change', req); apply();
  }

  const play = v => { const p = v.play(); if (p && p.catch) p.catch(() => { v.muted = true; v.play().catch(() => {}); }); };

  // Demo film: "Watch the demo" scrolls to it and plays; it plays muted while in view.
  const demo = document.querySelector('#demo video');
  if (demo) {
    document.querySelector('[data-play-demo]')?.addEventListener('click', e => {
      e.preventDefault();
      const r = demo.getBoundingClientRect();
      scrollTo({ top: scrollY + r.top - Math.max(16, (innerHeight - r.height) / 2), behavior: reduce.matches ? 'auto' : 'smooth' });
      history.replaceState(null, '', '#demo'); play(demo);
    });
    if (demo.hasAttribute('data-silent')) {
      demo.defaultMuted = true; demo.muted = true; demo.volume = 0;
    } else {
      const fig = demo.closest('.demo-media'), btn = document.createElement('button');
      btn.type = 'button'; btn.className = 'demo-unmute'; btn.textContent = 'Tap for sound'; btn.hidden = !demo.muted;
      fig.style.position = 'relative'; fig.appendChild(btn);
      btn.addEventListener('click', () => { demo.muted = false; play(demo); btn.hidden = true; });
      demo.addEventListener('volumechange', () => { btn.hidden = !demo.muted; });
    }
    if ('IntersectionObserver' in window) new IntersectionObserver(es => es.forEach(e => {
      if (reduce.matches) return;
      if (e.isIntersecting) { if (demo.paused) play(demo); } else if (!demo.paused) demo.pause();
    }), { threshold: .5 }).observe(demo);
  }

  // Posters load when a video comes near the viewport.
  const vids = [...document.querySelectorAll('video[data-poster]')];
  const setPoster = v => { if (!v.getAttribute('poster')) v.poster = v.dataset.poster; };
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { setPoster(e.target); io.unobserve(e.target); } }), { rootMargin: '600px 0px' });
    vids.forEach(v => io.observe(v));
  } else vids.forEach(setPoster);

  // Only one comparison or gallery clip plays at a time.
  const groups = new Set();
  const soloStop = except => groups.forEach(g => { if (g !== except) g.stop(); });
  document.querySelectorAll('.gallery video').forEach(v => {
    const g = { stop: () => { if (!v.paused) v.pause(); } };
    groups.add(g); v.addEventListener('play', () => soloStop(g));
  });

  const ready = v => new Promise(r => {
    if (v.readyState >= 3) return r();
    v.addEventListener('canplay', () => r(), { once: true });
    v.preload = 'auto'; if (v.networkState === 0 || v.readyState === 0) v.load();
  });
  const fmt = (t, d) => `${t.toFixed(2)} / ${d.toFixed(2)} s`;

  // Fetch a clip completely before it plays. Streaming several 2K files at once stalls on ordinary connections, and a
  // stalled lead makes synchronized panes jump; from memory they play smoothly and in step. Falls back to streaming.
  const blobs = new Map();  // url -> Promise of an object URL (or the url itself if the fetch fails)
  const fetchURL = (url, onBytes) => {
    if (!blobs.has(url)) blobs.set(url, (async () => {
      try {
        const r = await fetch(url);
        if (!r.ok || !r.body) throw new Error(String(r.status));
        const total = +r.headers.get('content-length') || 0, reader = r.body.getReader(), parts = [];
        let got = 0;
        for (;;) { const { done, value } = await reader.read(); if (done) break; parts.push(value); got += value.length; onBytes(got, total); }
        return URL.createObjectURL(new Blob(parts, { type: 'video/mp4' }));
      } catch (e) { onBytes(1, 1); return url; }
    })());
    else onBytes(1, 1);
    return blobs.get(url);
  };
  const fetchClip = (v, onBytes) => v._loaded ||= (async () => {
    v.src = await fetchURL(v.querySelector('source')?.src || v.src, onBytes);
    await ready(v);
  })();
  const progress = (vs, show) => {
    const seen = new Map();
    return v => (got, total) => {
      seen.set(v, [got, total || got]);
      let a = 0, b = 0; for (const [g, t] of seen.values()) { a += g; b += t; }
      show(Math.min(99, Math.round(a / Math.max(b, 1) * 100 * seen.size / vs.length)));
    };
  };

  // Detail zoom follows the pointer: the same relative point in every pane.
  const zoomable = (fig, area) => {
    const z = fig.querySelector('.sync-zoom'); if (!z) return;
    z.addEventListener('click', () => { const on = !fig.classList.contains('is-zoomed'); fig.classList.toggle('is-zoomed', on); z.setAttribute('aria-pressed', String(on)); });
    area.addEventListener('pointermove', e => {
      if (!fig.classList.contains('is-zoomed')) return;
      const pane = e.target.closest('.pane,.ladder-stage'); if (!pane) return;
      const r = pane.getBoundingClientRect();
      const x = Math.min(100, Math.max(0, (e.clientX - r.left) / r.width * 100)), y = Math.min(100, Math.max(0, (e.clientY - r.top) / r.height * 100));
      fig.style.setProperty('--fx', x.toFixed(1) + '%'); fig.style.setProperty('--fy', y.toFixed(1) + '%');
    });
  };

  // Synchronized figures: the lead video drives the others.
  document.querySelectorAll('[data-sync]').forEach(fig => {
    const vs = [...fig.querySelectorAll('video')], lead = fig.querySelector('video[data-lead]') || vs[0], fol = vs.filter(v => v !== lead);
    const btn = fig.querySelector('.sync-play'), seek = fig.querySelector('.sync-seek'), time = fig.querySelector('.sync-time');
    const dur = parseFloat(fig.dataset.duration); let on = false, raf = 0, primed = null, rate = 1, loading = false;
    const ui = () => { const t = Math.min(lead.currentTime, dur); time.textContent = loading ? time.textContent : fmt(t, dur); seek.value = String(Math.round(t / dur * 1000)); btn.textContent = on ? 'Pause' : 'Play'; btn.setAttribute('aria-pressed', String(on)); };
    const tick = () => {
      raf = 0; if (!on) return;
      const t = lead.currentTime;
      if (lead.readyState >= 3) for (const f of fol) { if (f.seeking || f.readyState < 3) continue; if (Math.abs(f.currentTime - t) > .1) f.currentTime = t; if (f.paused) f.play().catch(() => {}); }
      ui(); raf = requestAnimationFrame(tick);
    };
    const g = {
      stop() { on = false; vs.forEach(v => v.pause()); ui(); },
      async start() {
        soloStop(g); on = true; ui();
        if (!primed) {
          loading = true; time.textContent = 'Loading 0%';
          const track = progress(vs, pct => { if (loading) time.textContent = `Loading ${pct}%`; });
          primed = Promise.all(vs.map(v => fetchClip(v, track(v))));
        }
        await primed; loading = false; if (!on) { ui(); return; }
        const t = lead.currentTime; fol.forEach(f => { f.currentTime = t; }); vs.forEach(v => { v.playbackRate = rate; v.loop = true; });
        await Promise.all(vs.map(v => v.play().catch(() => {}))); if (!raf) raf = requestAnimationFrame(tick);
      },
    };
    groups.add(g);
    btn.addEventListener('click', () => on ? g.stop() : g.start());
    lead.addEventListener('pause', () => { if (on && !lead.ended) g.stop(); });
    lead.addEventListener('seeked', ui);
    seek.addEventListener('input', () => { const t = seek.value / 1000 * dur; vs.forEach(v => { v.currentTime = t; }); ui(); });
    fig.querySelectorAll('[data-rate]').forEach(b => b.addEventListener('click', () => {
      rate = parseFloat(b.dataset.rate); vs.forEach(v => { v.playbackRate = rate; });
      fig.querySelectorAll('[data-rate]').forEach(o => o.setAttribute('aria-pressed', String(o === b)));
    }));
    zoomable(fig, fig);
    ui();
  });

  // Resolution ladder: only the selected tier decodes; switching keeps the playback time.
  document.querySelectorAll('.ladder').forEach(fig => {
    const tiers = [...fig.querySelectorAll('.tier')], tabs = [...fig.querySelectorAll('[role=tab]')];
    const btn = fig.querySelector('.sync-play'), seek = fig.querySelector('.sync-seek'), time = fig.querySelector('.sync-time');
    const dur = parseFloat(fig.dataset.duration); let on = false, raf = 0;
    let cur = tiers.find(t => t.classList.contains('is-active')).querySelector('video');
    const ui = () => { const t = Math.min(cur.currentTime, dur); time.textContent = fmt(t, dur); seek.value = String(Math.round(t / dur * 1000)); btn.textContent = on ? 'Pause' : 'Play'; btn.setAttribute('aria-pressed', String(on)); };
    const tick = () => { raf = 0; if (!on) return; ui(); raf = requestAnimationFrame(tick); };
    const g = {
      stop() { on = false; cur.pause(); ui(); },
      async start() { soloStop(g); on = true; ui(); cur.loop = true; await load(cur); if (!on) return; await cur.play().catch(() => {}); if (!raf) raf = requestAnimationFrame(tick); },
    };
    groups.add(g);
    const select = async key => {
      const next = tiers.find(t => t.dataset.tier === key); if (!next) return;
      const v = next.querySelector('video'); if (v === cur) return;
      const t = cur.currentTime, was = on;
      tabs.forEach(b => b.setAttribute('aria-selected', String(b.dataset.tier === key)));
      if (!v.getAttribute('poster') && v.dataset.poster) v.poster = v.dataset.poster;
      await load(v); v.currentTime = t; v.loop = true;
      if (was) { v.playbackRate = cur.playbackRate; await v.play().catch(() => {}); }
      tiers.forEach(x => x.classList.toggle('is-active', x === next));
      const old = cur; cur = v; old.pause(); ui();
    };
    const stage = fig.querySelector('.ladder-stage');
    async function load(v) {
      const label = v.closest('.tier').dataset.tier.toUpperCase();
      fig.classList.add('is-loading'); stage.dataset.loading = `Loading ${label}…`;
      await fetchClip(v, (got, total) => { stage.dataset.loading = `Loading ${label} ${total ? Math.round(got / total * 100) : 0}%`; });
      fig.classList.remove('is-loading');
    }
    tabs.forEach(b => b.addEventListener('click', () => select(b.dataset.tier)));
    btn.addEventListener('click', () => on ? g.stop() : g.start());
    seek.addEventListener('input', () => { cur.currentTime = seek.value / 1000 * dur; ui(); });
    tiers.forEach(t => t.querySelector('video').addEventListener('seeked', ui));
    zoomable(fig, fig.querySelector('.ladder-stage'));
    ui();
  });

  // Deck-style comparison viewers: hidden videos drawn into a full-frame canvas with a magnifier and matched zoom canvases.
  document.querySelectorAll('.cmpv').forEach(fig => {
    const views = JSON.parse(fig.dataset.views), leadKey = fig.dataset.lead;
    const main = fig.querySelector('.cmpv-canvas'), badge = fig.querySelector('.cmpv-badge'), loadEl = fig.querySelector('.cmpv-loading');
    const viewBtn = fig.querySelector('.cmpv-view'), playBtn = fig.querySelector('.cmpv-play'), tabs = [...fig.querySelectorAll('.cmpv-tab')];
    const zooms = [...fig.querySelectorAll('.cmpv-zoom')].map(z => ({ key: z.dataset.key, c: z.querySelector('canvas') }));
    const samples = [];
    fig.querySelectorAll('.cmpv-media span[data-cx]').forEach(e => { samples[+e.dataset.i] = { cx: +e.dataset.cx, cy: +e.dataset.cy, vids: {} }; });
    fig.querySelectorAll('.cmpv-media video').forEach(v => { samples[+v.dataset.i].vids[v.dataset.key] = v; });
    let cur = 0, view = 0, z = +fig.dataset.zoom || 4, cx = samples[0].cx, cy = samples[0].cy, on = false, raf = 0;
    const dpr = () => window.devicePixelRatio || 1;
    const fit = c => { const w = Math.round(c.clientWidth * dpr()), h = Math.round(c.clientHeight * dpr()); if (w && h && (c.width !== w || c.height !== h)) { c.width = w; c.height = h; } };
    const region = v => {  // magnifier window in the video's own pixels; same relative place in every output
      const W = v.videoWidth, H = v.videoHeight, w = W / z, h = H / z;
      return [Math.max(0, Math.min(W - w, cx * W - w / 2)), Math.max(0, Math.min(H - h, cy * H - h / 2)), w, h];
    };
    const draw = () => {
      raf = 0;
      const s = samples[cur], mv = s.vids[views[view][0]], lead = s.vids[leadKey];
      if (on && lead.readyState >= 3) for (const v of Object.values(s.vids)) {
        if (v === lead || v.seeking || v.readyState < 3) continue;
        if (Math.abs(v.currentTime - lead.currentTime) > .1) v.currentTime = lead.currentTime;
        if (v.paused) v.play().catch(() => {});
      }
      if (mv && mv.readyState >= 2) {
        fit(main); const g = main.getContext('2d');
        g.drawImage(mv, 0, 0, main.width, main.height);
        const [x, y, w, h] = region(mv), sx = main.width / mv.videoWidth, sy = main.height / mv.videoHeight;
        g.lineWidth = 2 * dpr(); g.strokeStyle = '#76b900'; g.strokeRect(x * sx, y * sy, w * sx, h * sy);
      }
      for (const zz of zooms) {
        const v = s.vids[zz.key]; if (!v || v.readyState < 2) continue;
        fit(zz.c); const [x, y, w, h] = region(v);
        zz.c.getContext('2d').drawImage(v, x, y, w, h, 0, 0, zz.c.width, zz.c.height);
      }
      if (on) raf = requestAnimationFrame(draw);
    };
    const redraw = () => { if (!raf) raf = requestAnimationFrame(draw); };
    const load = i => {
      const s = samples[i];
      if (!s.ready) {
        const list = Object.values(s.vids);
        const track = progress(list, pct => { if (cur === i) loadEl.textContent = `Loading ${pct}%`; });
        s.ready = Promise.all(list.map(v => fetchClip(v, track(v)))).then(() => { s.done = true; });
      }
      loadEl.hidden = !!s.done;
      return s.ready.then(() => { if (cur === i) { loadEl.hidden = true; redraw(); } });
    };
    const vids = () => Object.values(samples[cur].vids);
    const g = {
      stop() { on = false; vids().forEach(v => v.pause()); playBtn.textContent = 'Play'; },
      async start() {
        soloStop(g); on = true; playBtn.textContent = 'Pause';
        const i = cur; await load(i); if (!on || cur !== i) return;
        const lead = samples[i].vids[leadKey];
        vids().forEach(v => { v.loop = true; if (v !== lead) v.currentTime = lead.currentTime; });
        await Promise.all(vids().map(v => v.play().catch(() => {}))); redraw();
      },
    };
    groups.add(g);
    playBtn.addEventListener('click', () => on ? g.stop() : g.start());
    const cycle = () => { view = (view + 1) % views.length; badge.textContent = views[view][1]; viewBtn.textContent = `Showing: ${views[view][1]}`; redraw(); };
    viewBtn.addEventListener('click', cycle);
    tabs.forEach(b => b.addEventListener('click', () => {
      const i = +b.dataset.i; if (i === cur) return;
      const was = on; g.stop(); cur = i; cx = samples[i].cx; cy = samples[i].cy;
      tabs.forEach(t => t.setAttribute('aria-selected', String(t === b)));
      if (was) g.start(); else load(i);
    }));
    const at = e => { const r = main.getBoundingClientRect(); cx = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)); cy = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)); redraw(); };
    let down = null;
    main.addEventListener('pointermove', e => { if (e.pointerType === 'mouse' || down) at(e); });
    main.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY, e.pointerType]; });
    main.addEventListener('pointerup', e => {
      if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 6) { if (down[2] === 'mouse') cycle(); else at(e); }
      down = null;
    });
    main.addEventListener('pointercancel', () => { down = null; });
    main.addEventListener('wheel', e => { e.preventDefault(); z = Math.min(8, Math.max(1.5, z * (e.deltaY < 0 ? 1.15 : 1 / 1.15))); redraw(); }, { passive: false });
    addEventListener('resize', redraw);
    if ('IntersectionObserver' in window) {  // fetch the first sample as it approaches; play while in view
      new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) load(cur); }), { rootMargin: '900px 0px' }).observe(fig);
      new IntersectionObserver(es => es.forEach(e => { if (reduce.matches) return; if (e.isIntersecting) { if (!on) g.start(); } else if (on) g.stop(); }), { threshold: .35 }).observe(fig);
    } else load(cur);
  });

  // Three-column viewers: input small on the left, outputs stacked in the middle, detail windows beside each output.
  document.querySelectorAll('.cmp3').forEach(fig => {
    const panes = {};
    fig.querySelectorAll('.cmp3-pane').forEach(el => { panes[el.dataset.key] = { el, v: el.querySelector('video'), box: el.querySelector('.cmp3-box') }; });
    const lead = panes[fig.dataset.lead].v, all = Object.values(panes).map(p => p.v);
    const dets = [...fig.querySelectorAll('.cmp3-zoom')].map(z => ({ v: panes[z.dataset.key].v, c: z.querySelector('canvas') }));
    const loadEl = fig.querySelector('.cmp3-loading'), playBtn = fig.querySelector('.cmpv-play'), tabs = [...fig.querySelectorAll('.cmpv-tab')];
    const samples = [];
    fig.querySelectorAll('.cmpv-media span[data-cx]').forEach(e => { samples[+e.dataset.i] = { cx: +e.dataset.cx, cy: +e.dataset.cy, label: e.dataset.label, urls: {} }; });
    fig.querySelectorAll('.cmpv-media a[data-key]').forEach(a => { samples[+a.dataset.i].urls[a.dataset.key] = a.href; });
    let cur = -1, want = 0, z = +fig.dataset.zoom || 8, cx = samples[0].cx, cy = samples[0].cy, on = false, raf = 0, rate = 1, showing = null;
    const dpr = () => window.devicePixelRatio || 1;
    const win = () => {  // magnifier window as fractions of the frame; its shape follows the detail windows
      const c = dets[0].c, a = (c.clientWidth || 8) / (c.clientHeight || 9), W = lead.videoWidth || 16, H = lead.videoHeight || 9;
      const w = 1 / z, h = Math.min(1, (W / z) / a / H);
      return [Math.max(0, Math.min(1 - w, cx - w / 2)), Math.max(0, Math.min(1 - h, cy - h / 2)), w, h];
    };
    const draw = () => {
      raf = 0;
      if (on && lead.readyState >= 3) for (const v of all) {
        if (v === lead || v.seeking || v.readyState < 3) continue;
        if (Math.abs(v.currentTime - lead.currentTime) > .1) v.currentTime = lead.currentTime;
        if (v.paused) v.play().catch(() => {});
      }
      const [x, y, w, h] = win();
      for (const p of Object.values(panes)) if (p.box) Object.assign(p.box.style, { left: x * 100 + '%', top: y * 100 + '%', width: w * 100 + '%', height: h * 100 + '%' });
      const A = (lead.videoWidth || 16) / (lead.videoHeight || 9);
      for (const d of dets) {
        if (d.v.readyState < 2) continue;
        const cw = Math.round(d.c.clientWidth * dpr()), ch = Math.round(d.c.clientHeight * dpr());
        if (cw && ch && (d.c.width !== cw || d.c.height !== ch)) { d.c.width = cw; d.c.height = ch; }
        // a source with another aspect (e.g. a 1344x768 input) is shown cover-cropped to the output frame; map the same way
        let W = d.v.videoWidth, H = d.v.videoHeight, ox = 0, oy = 0;
        if (W / H > A) { ox = (W - H * A) / 2; W = H * A; } else { oy = (H - W / A) / 2; H = W / A; }
        d.c.getContext('2d').drawImage(d.v, ox + x * W, oy + y * H, w * W, h * H, 0, 0, d.c.width, d.c.height);
      }
      if (on) raf = requestAnimationFrame(draw);
    };
    const redraw = () => { if (!raf) raf = requestAnimationFrame(draw); };
    const show = i => showing && want === i ? showing : (want = i, showing = (async () => {
      const s = samples[i], keys = Object.keys(s.urls);
      loadEl.hidden = false; loadEl.textContent = 'Loading…';
      const track = progress(keys, pct => { if (want === i) loadEl.textContent = `Loading ${pct}%`; });
      const urls = await Promise.all(keys.map(k => fetchURL(s.urls[k], track(k))));
      if (want !== i) return false;
      if (cur !== i) {
        keys.forEach((k, j) => { panes[k].v.src = urls[j]; });
        await Promise.all(keys.map(k => ready(panes[k].v)));
        if (want !== i) return false;
        cur = i; cx = s.cx; cy = s.cy;
        const badge = panes.input && panes.input.el.querySelector('.badge'); if (badge && s.label) badge.textContent = s.label;
      }
      loadEl.hidden = true; redraw(); return true;
    })());
    const g = {
      stop() { on = false; all.forEach(v => v.pause()); playBtn.textContent = 'Play'; },
      async start() {
        soloStop(g); on = true; playBtn.textContent = 'Pause';
        if (!(await show(want)) || !on) return;
        all.forEach(v => { v.loop = true; v.playbackRate = rate; if (v !== lead) v.currentTime = lead.currentTime; });
        await Promise.all(all.map(v => v.play().catch(() => {}))); redraw();
      },
    };
    groups.add(g);
    playBtn.addEventListener('click', () => on ? g.stop() : g.start());
    tabs.forEach(b => b.addEventListener('click', () => {
      const i = +b.dataset.i; if (i === want) return;
      const was = on; g.stop(); tabs.forEach(t => t.setAttribute('aria-selected', String(t === b)));
      showing = null; want = i;
      if (was) g.start(); else show(i);
    }));
    fig.querySelectorAll('[data-rate]').forEach(b => b.addEventListener('click', () => {
      rate = parseFloat(b.dataset.rate); all.forEach(v => { v.playbackRate = rate; });
      fig.querySelectorAll('[data-rate]').forEach(o => o.setAttribute('aria-pressed', String(o === b)));
    }));
    const at = (e, el) => { const r = el.getBoundingClientRect(); cx = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)); cy = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)); redraw(); };
    for (const p of Object.values(panes)) {
      const el = p.el; let down = null;
      el.addEventListener('pointermove', e => { if (e.pointerType === 'mouse' || down) at(e, el); });
      el.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
      el.addEventListener('pointerup', e => { if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 6) at(e, el); down = null; });
      el.addEventListener('pointercancel', () => { down = null; });
      // 2026-10-27 (user): fixed small magnifier box; no scroll zoom in the comparison viewers
    }
    addEventListener('resize', redraw);
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting && cur < 0) show(want); }), { rootMargin: '900px 0px' }).observe(fig);
      new IntersectionObserver(es => es.forEach(e => { if (reduce.matches) return; if (e.isIntersecting) { if (!on) g.start(); } else if (on) g.stop(); }), { threshold: .3 }).observe(fig);
    } else show(want);
  });

  // Charts grow in once when they scroll into view.
  const charts = [...document.querySelectorAll('.gfig,.hchart')];
  if (!reduce.matches && 'IntersectionObserver' in window) {
    document.documentElement.classList.add('js-anim');
    charts.forEach(c => c.classList.add('chart-pending'));
    const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.remove('chart-pending'); io.unobserve(e.target); } }), { threshold: .35 });
    charts.forEach(c => io.observe(c));
  }

  // Bar tooltips (values are also printed beside every bar).
  const tip = document.querySelector('.tip');
  if (tip) {
    document.querySelectorAll('[data-tip]').forEach(el => {
      el.addEventListener('pointermove', e => {
        tip.textContent = el.dataset.tip; tip.hidden = false;
        const w = tip.offsetWidth, h = tip.offsetHeight;
        tip.style.left = Math.min(innerWidth - w - 10, Math.max(10, e.clientX + 14)) + 'px';
        let top = e.clientY + 16; if (top + h > innerHeight - 8) top = e.clientY - h - 12;
        tip.style.top = Math.max(8, top) + 'px';
      });
      el.addEventListener('pointerleave', () => { tip.hidden = true; });
    });
  }
  // BibTeX copy button (same behaviour as the StreamRender-H3 page)
  document.querySelectorAll('.cite-copy').forEach(b => b.addEventListener('click', async () => {
    const el = document.getElementById(b.dataset.copyTarget);
    try { await navigator.clipboard.writeText(el.textContent); b.textContent = 'Copied'; } catch (e) {
      const r = document.createRange(); r.selectNodeContents(el); const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r); b.textContent = 'Press ⌘C';
    }
    setTimeout(() => { b.textContent = 'Copy all BibTeX'; }, 1800);
  }));
  window.blogReady = true;
})();
