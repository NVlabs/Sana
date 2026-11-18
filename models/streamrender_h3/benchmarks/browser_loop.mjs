// Actual engine -> HTTP -> H3 -> JPEG feedback acceptance test.
import {chromium} from '../coding/threejs-racing/node_modules/playwright/index.mjs';
import {spawn} from 'node:child_process';
import {mkdir, writeFile, mkdtemp, rm} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const engine = path.join(root, 'coding/threejs-racing');
const output = process.env.VALIDATION_OUTPUT || path.join(root, 'outputs/browser-validation');
const server = process.env.STREAMRENDER_SERVER || 'http://127.0.0.1:8765';
const frontendUrl = 'http://127.0.0.1:' + (process.env.FRONTEND_PORT || '5191');
await mkdir(output, {recursive: true});
const deadline = Date.now() + 20 * 60 * 1000;
for (;;) {
  try {
    const health = await (await fetch(server + '/health')).json();
    if (health.error) throw Error(health.error);
    if (health.ready) break;
  } catch (error) {
    if (Date.now() > deadline) throw error;
  }
  if (Date.now() > deadline) throw Error('GPU warmup timeout');
  await new Promise(resolve => setTimeout(resolve, 1000));
}
const frontend = spawn(process.execPath, [path.join(engine, 'node_modules/vite/bin/vite.js')],
                       {cwd: engine, stdio: 'inherit'});
let browser;
// Chromium Unix sockets need a short local path. Only ephemeral browser
// metadata uses /tmp; model, Python, CUDA and output caches stay on code storage.
const browserTmp = await mkdtemp('/tmp/streamrender-browser-');
process.env.TMPDIR = browserTmp;
try {
  for (let i = 0; i < 60; i++) {
    try { if ((await fetch(frontendUrl)).ok) break; } catch {}
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  browser = await chromium.launch({headless: true,
    executablePath: process.env.CHROMIUM_BIN || undefined,
    args: ['--no-sandbox', '--disable-dev-shm-usage']});
  const page = await browser.newPage({viewport: {width: 1440, height: 1400}});
  const errors = [];
  page.on('pageerror', error => errors.push(String(error)));
  await page.goto(frontendUrl);
  await page.waitForFunction(() => window.racing, {timeout: 60000});
  await page.evaluate(() => window.racing.setAuto(true));
  await page.locator('#live').click();
  const rounds = Number(process.env.VALIDATION_ROUNDS || 30);
  await page.waitForFunction(rounds => {
    const text = document.querySelector('#render-status')?.textContent || '';
    const match = text.match(/chunk (\d+)/);
    return match && Number(match[1]) >= rounds;
  }, rounds, {timeout: 300000});
  const result = await page.evaluate(async () => {
    const image = document.querySelector('#render-feedback');
    const id = image.src.split('/').pop();
    const response = await fetch('/runtime/status/' + id);
    return {status: await response.json(), imageWidth: image.naturalWidth,
            imageHeight: image.naturalHeight, engine: window.racing.state()};
  });
  if (result.imageWidth !== 1344 || result.imageHeight !== 768) {
    throw Error('Feedback image geometry mismatch: ' + JSON.stringify(result));
  }
  if (errors.length) throw Error(errors.join('\n'));
  await page.screenshot({path: path.join(output, 'browser.png'), fullPage: true});
  await page.locator('#live').click();
  await writeFile(path.join(output, 'result.json'), JSON.stringify(result, null, 2));
  console.log('BROWSER_LOOP_PASS', JSON.stringify(result));
} finally {
  if (browser) await browser.close();
  frontend.kill('SIGTERM');
  await rm(browserTmp, {recursive: true, force: true});
}
