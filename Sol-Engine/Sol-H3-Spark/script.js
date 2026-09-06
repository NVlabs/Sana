/* Standalone counterparts of the blog's navigation, video and citation controls. */
(() => {
  'use strict';
  const header = document.querySelector('.site-header');
  const menu = document.querySelector('.nav-menu-toggle');
  const closeMenu = () => {
    header.classList.remove('menu-open');
    menu.setAttribute('aria-expanded', 'false');
    menu.setAttribute('aria-label', 'Open section menu');
  };
  menu.addEventListener('click', () => {
    const open = header.classList.toggle('menu-open');
    menu.setAttribute('aria-expanded', String(open));
    menu.setAttribute('aria-label', open ? 'Close section menu' : 'Open section menu');
  });
  document.querySelectorAll('.nav-links a').forEach(link => link.addEventListener('click', closeMenu));
  document.addEventListener('keydown', event => { if (event.key === 'Escape') closeMenu(); });

  const theme = document.querySelector('.theme-toggle');
  const lightSheet = document.getElementById('light-theme');
  const sun = theme.innerHTML;
  let light = false;
  theme.addEventListener('click', () => {
    light = !light;
    lightSheet.media = light ? 'all' : 'not all';
    document.documentElement.style.colorScheme = light ? 'light' : 'dark';
    theme.setAttribute('aria-label', light ? 'Switch to dark theme' : 'Switch to light theme');
    theme.innerHTML = light ? '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20.985 12.486a9 9 0 1 1-9.473-9.472c.405-.022.617.44.402.784a6.25 6.25 0 0 0 8.268 8.268c.344-.215.806-.004.803.42"/></svg>' : sun;
  });

  const hero = document.querySelector('.hero-film');
  const heroButton = document.querySelector('.spark-hero-control');
  const playIcon = heroButton.querySelector('svg').innerHTML;
  const heroState = () => {
    const paused = hero.paused;
    heroButton.querySelector('span').textContent = paused ? 'Play' : 'Pause';
    heroButton.setAttribute('aria-label', paused ? 'Play background video' : 'Pause background video');
    heroButton.querySelector('svg').innerHTML = paused ? playIcon : '<rect x="14" y="4" width="4" height="16" rx="1"/><rect x="6" y="4" width="4" height="16" rx="1"/>';
  };
  heroButton.addEventListener('click', () => { if (hero.paused) void hero.play().catch(heroState); else hero.pause(); });
  hero.addEventListener('play', heroState);
  hero.addEventListener('pause', heroState);
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const applyMotion = () => { if (reducedMotion.matches) hero.pause(); else void hero.play().catch(heroState); };
  reducedMotion.addEventListener('change', applyMotion);
  applyMotion();

  document.querySelectorAll('.spark-gallery').forEach(gallery => {
    const videos = [...gallery.querySelectorAll('video')];
    const [start, pause] = gallery.querySelectorAll('.spark-gallery-controls button');
    const status = gallery.querySelector('.playback-status');
    const startMarkup = start.innerHTML;
    let generation = 0;
    pause.addEventListener('click', () => {
      generation++;
      videos.forEach(video => video.pause());
      start.disabled = false;
      start.innerHTML = startMarkup;
    });
    start.addEventListener('click', async () => {
      const run = ++generation;
      start.disabled = true;
      start.textContent = 'Loading…';
      status.textContent = '';
      try {
        await Promise.all(videos.map(video => new Promise((resolve, reject) => {
          video.pause(); video.muted = true;
          if (video.readyState >= 2) { resolve(); return; }
          const cleanup = () => { clearTimeout(timer); video.removeEventListener('loadeddata', ready); video.removeEventListener('error', fail); };
          const ready = () => { cleanup(); resolve(); };
          const fail = () => { cleanup(); reject(new Error('Video loading failed')); };
          const timer = setTimeout(fail, 20000);
          video.addEventListener('loadeddata', ready, {once:true});
          video.addEventListener('error', fail, {once:true});
          video.preload = 'auto'; video.load();
        })));
        if (run !== generation) return;
        videos.forEach(video => { video.currentTime = 0; });
        await Promise.all(videos.map(video => video.play()));
        if (run !== generation) videos.forEach(video => video.pause());
      } catch {
        if (run === generation) {
          videos.forEach(video => video.pause());
          status.textContent = 'Some videos could not start together. Please use the individual play controls.';
        }
      } finally {
        if (run === generation) { start.disabled = false; start.innerHTML = startMarkup; }
      }
    });
  });
  const copy = document.querySelector('.copy-code');
  copy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(document.querySelector('.citation-window code').textContent);
      copy.textContent = 'Copied';
    } catch { copy.textContent = 'Select the text to copy'; }
  });
})();
