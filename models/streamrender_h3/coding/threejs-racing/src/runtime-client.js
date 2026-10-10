// Same-origin proxy to the persistent GPU worker; no credentials in the browser.
const panel = document.createElement('section');
panel.innerHTML = '<h2>Streaming Render</h2><small id="render-status">GPU worker status</small><img id="render-feedback" alt="Streaming H3 video feedback" style="display:none;width:100%;border-radius:8px;margin-top:10px">';
document.querySelector('main').append(panel);
const status = panel.querySelector('#render-status');
const feedback = panel.querySelector('#render-feedback');
let timer;

async function request(path, options) {
  const response = await fetch('/runtime/' + path, options);
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || response.statusText);
  return value;
}

export async function startRuntime(config) {
  const health = await request('health');
  if (!health.ready) throw new Error(health.error || 'GPU worker loading / warming up');
  const session = await request('session', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(config),
  });
  const deadline = performance.now() + 180000;
  for (;;) {
    const value = await request('status/' + session.id);
    if (value.error || value.closed) throw new Error(value.error || 'Session closed');
    if (value.ready) break;
    if (performance.now() > deadline) {
      await closeRuntime(session.id);
      throw new Error('Session setup timed out');
    }
    await new Promise(resolve => setTimeout(resolve, 250));
  }
  feedback.src = '/runtime/stream/' + session.id;
  feedback.style.display = 'block';
  clearInterval(timer);
  timer = setInterval(async () => {
    try {
      const value = await request('status/' + session.id);
      const metrics = value.metrics;
      status.textContent = value.error || (metrics.backend === 'passthrough'
        ? 'Transport diagnostic · chunk ' + metrics.round
        : metrics.rgb_ready_wall_ms == null
        ? 'Waiting for first reference chunk'
        : 'RGB-ready render ' + metrics.rgb_ready_wall_ms.toFixed(1) + ' ms · chunk ' + metrics.round);
      if (value.closed) clearInterval(timer);
    } catch (error) {
      status.textContent = error.message;
    }
  }, 500);
  return {...session, runtime: true};
}

export async function sendRuntimeFrame(session, index, blob, controls) {
  // Retry the identical frame; never skip simulation steps on backpressure.
  for (;;) {
    const response = await fetch('/runtime/frame/' + session.id + '/' + index, {
      method: 'POST', body: blob,
      headers: {'Content-Type': 'image/png', 'X-Controls': JSON.stringify(controls)},
    });
    if (response.status === 429) {
      const state = await request('status/' + session.id);
      if (state.closed || state.error) throw new Error(state.error || 'Session closed');
      await new Promise(resolve => setTimeout(resolve, 25));
      continue;
    }
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Reference frame rejected');
    return result;
  }
}

export async function closeRuntime(id) {
  clearInterval(timer);
  if (id) await request('close/' + id, {method: 'POST', body: ''});
  feedback.removeAttribute('src');
}
