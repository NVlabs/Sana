(() => {
  const root = document.querySelector('[data-f1-gallery]');
  if (!root) return;
  const video = root.querySelector('video');
  const clips = [...root.querySelectorAll('[data-f1-clip]')];
  const modes = [...root.querySelectorAll('[data-f1-mode]')];
  const title = root.querySelector('[data-f1-title]');
  const caption = root.querySelector('[data-f1-caption]');
  const reference = root.querySelector('[data-f1-reference]');
  const status = root.querySelector('[data-f1-status]');
  const direct = root.querySelector('[data-f1-direct]');
  let index = 0, mode = 'comparison', revision = 0;
  let pendingTime = 0, pendingPlay = false;

  function select(nextIndex, nextMode) {
    const sameClip = index === nextIndex;
    const time = sameClip ? (video.readyState ? video.currentTime : pendingTime) : 0;
    const play = video.readyState ? !video.paused : pendingPlay;
    index = nextIndex;
    mode = nextMode;
    pendingTime = time;
    pendingPlay = play;
    const item = clips[index], token = ++revision;
    video.pause();
    clips.forEach((button, i) => button.setAttribute('aria-pressed', String(i === index)));
    modes.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.f1Mode === mode)));
    title.textContent = item.dataset.title;
    caption.textContent = mode === 'comparison'
      ? 'Left: engine reconstruction RGB. Middle: semantic input. Right: H3 render.'
      : 'Final H3 render, conditioned on the semantic replay and an ImageGen appearance reference.';
    reference.href = item.dataset.reference;
    reference.setAttribute('aria-label', `${item.dataset.title}: open the appearance reference`);
    direct.href = item.dataset[mode];
    video.setAttribute('aria-label', `${item.dataset.title}: ${mode === 'comparison' ? 'engine, semantic input and H3 comparison' : 'H3 render'}`);
    video.style.aspectRatio = mode === 'comparison' ? '14 / 3' : '7 / 4';
    video.width = mode === 'comparison' ? 2016 : 1344;
    video.height = mode === 'comparison' ? 432 : 768;
    video.poster = item.dataset[`${mode}Poster`];
    status.textContent = 'Loading clip…';
    video.preload = 'auto';
    video.src = item.dataset[mode];
    video.addEventListener('loadedmetadata', () => {
      if (token !== revision) return;
      video.currentTime = Math.min(time, Math.max(0, video.duration - 1 / 24));
      status.textContent = '';
      if (play) video.play().catch(() => { status.textContent = 'Press play to continue.'; });
    }, { once: true });
    video.load();
  }

  clips.forEach((button, i) => {
    button.addEventListener('click', () => { if (i !== index) select(i, mode); });
    button.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? clips.length - 1
        : (i + (event.key === 'ArrowRight' ? 1 : clips.length - 1)) % clips.length;
      clips[next].focus();
      if (next !== index) select(next, mode);
    });
  });
  modes.forEach(button => button.addEventListener('click', () => {
    if (button.dataset.f1Mode !== mode) select(index, button.dataset.f1Mode);
  }));
  video.addEventListener('error', () => { status.textContent = 'Unable to load the clip. Try the direct video link below.'; });
})();
