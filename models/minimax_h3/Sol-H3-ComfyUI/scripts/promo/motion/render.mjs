// Render one motion-graphics scene of scenes.html to a transparent PNG sequence.
// node render.mjs <scene> <seconds> <outDir> [paramsJSON] [--still t1,t2,...]
import { createRequire } from 'module';
import { mkdirSync } from 'fs';
import { pathToFileURL } from 'url';
import path from 'path';
const require = createRequire(import.meta.url);
// Set PLAYWRIGHT_PATH to a playwright install if it is not resolvable from here.
const { chromium } = require(process.env.PLAYWRIGHT_PATH || 'playwright');

const [scene, seconds, outDir, paramsJson = '{}', ...rest] = process.argv.slice(2);
const FPS = Number(process.env.FPS || 30), D = Number(seconds), params = JSON.parse(paramsJson);
const stillIndex = rest.indexOf('--still');
const stills = stillIndex >= 0 ? rest[stillIndex + 1].split(',').map(Number) : null;
mkdirSync(outDir, { recursive: true });
const browser = await chromium.launch({
  executablePath: process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  headless: true });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
const errors = [];
page.on('pageerror', e => errors.push(e.message));
await page.goto(pathToFileURL(path.resolve(path.dirname(new URL(import.meta.url).pathname), 'scenes.html')).href);
await page.evaluate(() => document.fonts.ready);
const times = stills || Array.from({ length: Math.round(D * FPS) }, (_, i) => i / FPS);
for (const [i, t] of times.entries()) {
  await page.evaluate(([n, t, p, d]) => window.renderFrame(n, t, p, d), [scene, t, params, D]);
  await page.screenshot({ path: path.join(outDir, stills ? `still-${t}.png` : `f${String(i).padStart(5, '0')}.png`),
                          omitBackground: true });
}
await browser.close();
if (errors.length) { console.error(errors); process.exit(1); }
console.log(JSON.stringify({ scene, frames: times.length, outDir }));
