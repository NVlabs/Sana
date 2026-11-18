// Render scripts/studio_preview.html headless and save docs/design/studio-*.png.
// Usage: node scripts/studio_screenshots.mjs <playwright module dir> <chrome path> [base URL]
import { createRequire } from 'node:module';
const [playwrightDir, chrome, base = 'http://localhost:8765'] = process.argv.slice(2);
const { chromium } = createRequire(import.meta.url)(playwrightDir);

const shots = [
    ['studio-1920.png', 1920, 1080, 'empty'], ['studio-1280.png', 1280, 800, 'empty'],
    ['studio-progress.png', 1920, 1080, 'progress'], ['studio-result.png', 1920, 1080, 'done'],
    ['studio-library.png', 1440, 900, 'library'], ['studio-first-frame.png', 1280, 800, 'first_frame'],
    ['studio-gallery.jpg', 1920, 1080, 'gallery'], ['studio-two-stage.png', 1440, 900, 'two_stage'],
    ['studio-zh.png', 1440, 900, 'empty', 'zh'],
];
const browser = await chromium.launch({ executablePath: chrome, headless: true });
let failures = 0;
for (const [name, width, height, state, lang = 'en'] of shots) {
    const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
    const errors = [];
    page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
    page.on('pageerror', e => errors.push(String(e)));
    await page.goto(`${base}/scripts/studio_preview.html?state=${state}&solh3_lang=${lang}`);
    await page.waitForFunction(() => window.previewReady === true);
    await page.waitForTimeout(900); // Entry animations settle.
    await page.screenshot({ path: `docs/design/${name}`, ...(name.endsWith('.jpg') ? { type: 'jpeg', quality: 82 } : {}) });
    console.log(name, errors.length ? 'ERRORS: ' + errors.join(' | ') : 'no console errors');
    failures += errors.length;
    await page.close();
}
await browser.close();
process.exit(failures ? 1 : 0);
