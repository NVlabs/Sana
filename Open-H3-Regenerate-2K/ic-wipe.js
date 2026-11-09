// IC view: 384P -> 768P before/after. Each source is one side-by-side video (input | IC), split on a canvas so both sides stay frame-locked.
(() => {
  const box = document.getElementById('ic-wipe');
  if (!box) return;
  const W = 1344, H = 768;
  const v = box.querySelector('video'), cv = box.querySelector('canvas'), ctx = cv.getContext('2d');
  const handle = box.querySelector('.ic-wipe-handle'), line = document.getElementById('ic-wipe-line');
  const tabs = [...document.querySelectorAll('.ic-wipe-tab')];
  const reduce = matchMedia('(prefers-reduced-motion: reduce)');
  let split = .5, touched = false, t0 = performance.now(), visible = false, cur = -1, poster = null, raf = 0;
  const draw = () => {
    if (!touched && !reduce.matches) split = .5 + .3 * Math.sin((performance.now() - t0) / 1400);
    handle.style.left = (split * 100) + '%';
    box.setAttribute('aria-valuenow', String(Math.round(split * 100)));
    const src = v.readyState >= 2 && v.videoWidth ? v : (poster && poster.complete && poster.naturalWidth ? poster : null);
    if (!src) return;
    const sx = Math.round(split * W);
    ctx.drawImage(src, 0, 0, W, H, 0, 0, W, H);
    if (sx < W) ctx.drawImage(src, W + sx, 0, W - sx, H, sx, 0, W - sx, H);
  };
  const loop = () => { draw(); raf = visible ? requestAnimationFrame(loop) : 0; };
  const play = () => { if (reduce.matches) return; const p = v.play(); if (p && p.catch) p.catch(() => {}); };
  const select = i => {
    if (i === cur) return;
    cur = i;
    tabs.forEach((t, k) => { t.setAttribute('aria-selected', String(k === i)); t.tabIndex = k === i ? 0 : -1; });
    line.textContent = tabs[i].dataset.line;
    poster = new Image(); poster.src = tabs[i].dataset.poster;
    touched = false; t0 = performance.now(); split = .5;
    v.src = tabs[i].dataset.src; v.load();
    if (visible) play();
  };
  tabs.forEach((t, i) => {
    t.addEventListener('click', () => { select(i); });
    t.addEventListener('keydown', e => {
      const d = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 }[e.key];
      if (!d) return;
      e.preventDefault();
      const n = (cur + d + tabs.length) % tabs.length; select(n); tabs[n].focus();
    });
  });
  const setFrom = e => { const r = box.getBoundingClientRect(); split = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)); touched = true; };
  let drag = false;
  box.addEventListener('pointerdown', e => { drag = true; box.setPointerCapture(e.pointerId); setFrom(e); if (v.paused) { const p = v.play(); if (p && p.catch) p.catch(() => {}); } });
  box.addEventListener('pointermove', e => { if (drag) setFrom(e); });
  box.addEventListener('pointerup', () => { drag = false; });
  box.addEventListener('pointercancel', () => { drag = false; });
  box.addEventListener('keydown', e => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    touched = true; split = Math.min(1, Math.max(0, split + (e.key === 'ArrowLeft' ? -.05 : .05))); e.preventDefault();
  });
  new IntersectionObserver(es => es.forEach(e => {
    visible = e.isIntersecting;
    if (visible) { if (cur < 0) select(0); play(); if (!raf) raf = requestAnimationFrame(loop); } else v.pause();
  }), { rootMargin: '200px 0px' }).observe(box);
})();
