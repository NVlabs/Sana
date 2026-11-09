// Version switch (2026-11-04): Last Version (v1) | IC Version (ic). #ic or any link into a view opens that view.
(() => {
  const views = [...document.querySelectorAll('.ver-view')], groups = [...document.querySelectorAll('.nav-group')];
  const reduce = matchMedia('(prefers-reduced-motion: reduce)');
  let cur = null;
  const viewOf = el => el && el.closest('.ver-view') ? el.closest('.ver-view').dataset.view : null;
  const set = v => {
    if (v === cur) return;
    cur = v;
    views.forEach(el => { el.hidden = el.dataset.view !== v; if (el.hidden) el.querySelectorAll('video').forEach(x => x.pause()); });
    groups.forEach(el => { el.hidden = el.dataset.view !== v; });
    document.querySelectorAll('.ver-btn').forEach(b => b.setAttribute('aria-selected', String(b.dataset.view === v)));
    document.querySelectorAll('.ver-card').forEach(b => b.classList.toggle('is-active', b.dataset.view === v));
  };
  const toView = v => {
    set(v);
    history.replaceState(null, '', v === 'ic' ? '#ic' : location.pathname + location.search);
    const t = document.getElementById(v === 'ic' ? 'view-ic' : 'view-v1');
    scrollTo({ top: scrollY + t.getBoundingClientRect().top - 72, behavior: reduce.matches ? 'auto' : 'smooth' });
  };
  document.querySelectorAll('.ver-btn,.ver-card').forEach(b => b.addEventListener('click', () => toView(b.dataset.view)));
  // links into the hidden view switch to it first, then the browser scrolls to the target as usual
  document.addEventListener('click', e => {
    const a = e.target.closest('a[href^="#"]'); if (!a) return;
    let t = null; try { t = document.querySelector(a.getAttribute('href')); } catch (_) { return; }
    const v = viewOf(t); if (v && v !== cur) set(v);
  }, true);
  const pick = () => {
    const h = location.hash; let t = null; try { t = h ? document.querySelector(h) : null; } catch (_) {}
    set(viewOf(t) || (h.startsWith('#ic') ? 'ic' : 'v1'));
    if (t) requestAnimationFrame(() => t.scrollIntoView());
  };
  addEventListener('hashchange', pick); pick();

  // IC demo film: plays muted while in view; tap for sound; "Watch the demo" scrolls to it and plays
  const demo = document.querySelector('#ic-demo video');
  if (!demo) return;
  const play = () => { const p = demo.play(); if (p && p.catch) p.catch(() => { demo.muted = true; demo.play().catch(() => {}); }); };
  const fig = demo.closest('.demo-media'), btn = document.createElement('button');
  btn.type = 'button'; btn.className = 'demo-unmute'; btn.textContent = 'Tap for sound'; btn.hidden = !demo.muted;
  fig.style.position = 'relative'; fig.appendChild(btn);
  btn.addEventListener('click', () => { demo.muted = false; play(); btn.hidden = true; });
  demo.addEventListener('volumechange', () => { btn.hidden = !demo.muted; });
  document.querySelector('[data-play-ic-demo]')?.addEventListener('click', e => {
    e.preventDefault(); set('ic');
    const r = demo.getBoundingClientRect();
    scrollTo({ top: scrollY + r.top - Math.max(16, (innerHeight - r.height) / 2), behavior: reduce.matches ? 'auto' : 'smooth' }); play();
  });
  if ('IntersectionObserver' in window) new IntersectionObserver(es => es.forEach(e => {
    if (reduce.matches) return;
    if (e.isIntersecting) { if (demo.paused) play(); } else if (!demo.paused) demo.pause();
  }), { threshold: .5 }).observe(demo);
})();
