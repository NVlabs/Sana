// Render one part of studio_film.html to an MP4 (1920x1080, 30 fps, silent).
// node render_film.mjs <part a|b> <out.mp4> [--still t1,t2,...]
// Needs the repository root served over http (BASE, default http://127.0.0.1:8777),
// PLAYWRIGHT_PATH if playwright is not resolvable, and ffmpeg (PROMO_FFMPEG or PATH).
import { createRequire } from 'module';
import { spawn } from 'child_process';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_PATH || 'playwright');

const [part, out, ...rest] = process.argv.slice(2);
const FPS = 30, BASE = process.env.BASE || 'http://127.0.0.1:8777';
const stillIndex = rest.indexOf('--still');
const stills = stillIndex >= 0 ? rest[stillIndex + 1].split(',').map(Number) : null;
const browser = await chromium.launch({
  executablePath: process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true, args: ['--autoplay-policy=no-user-gesture-required'] });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
const errors = [];
page.on('pageerror', e => errors.push(e.message));
page.on('console', m => { if (m.type() === 'error' && !/favicon/.test(m.location()?.url || '')) errors.push(m.text()); });
await page.goto(`${BASE}/scripts/promo/motion/studio_film.html?part=${part}&solh3_lang=en${process.env.FILM_QUERY ? "&" + process.env.FILM_QUERY : ""}`);
const D = await (await page.waitForFunction(() => window.filmReady, null, { timeout: 30000 })).jsonValue();
const times = stills || Array.from({ length: Math.round(D * FPS) }, (_, i) => i / FPS);
let ff = null;
if (!stills) {
  ff = spawn(process.env.PROMO_FFMPEG || 'ffmpeg', ['-y', '-hide_banner', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS),
    '-i', '-', '-c:v', 'libx264', '-crf', '14', '-preset', 'medium', '-pix_fmt', 'yuv420p', out], { stdio: ['pipe', 'inherit', 'inherit'] });
}
for (const t of times) {
  await page.evaluate(t => window.renderFrame(t), t);
  const png = await page.screenshot({ type: 'png' });
  if (ff) { if (!ff.stdin.write(png)) await new Promise(r => ff.stdin.once('drain', r)); }
  else require('fs').writeFileSync(`${out}-${t}.png`, png);
}
await browser.close();
if (ff) { ff.stdin.end(); await new Promise(r => ff.on('close', r)); }
if (errors.length) { console.error(errors); process.exit(1); }
console.log(JSON.stringify({ part, frames: times.length, out }));
