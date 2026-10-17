// Sol-H3 Studio: a branded full-screen <dialog> whose controls read/write the
// widgets of one SolH3Generate node; Generate = app.graphToPrompt() + api.queuePrompt().
// Pattern after FreeVideo web/studio.js (Apache-2.0). ComfyUI's `app`/`api` are injected
// so the DOM builder also runs standalone (scripts/studio_preview.html).
import { wordmark } from './wordmark.js';

const css = document.createElement('link');
css.rel = 'stylesheet'; css.href = new URL('./studio.css', import.meta.url).href;
document.head.append(css);

// ---- Language: English by default; the header switch is remembered per browser.
const LANG_KEY = 'solh3.lang';
function initialLang() {
    const override = typeof location !== 'undefined' ? new URLSearchParams(location.search).get('solh3_lang') : null;
    if (override === 'zh' || override === 'en') return override;
    try { return localStorage.getItem(LANG_KEY) === 'zh' ? 'zh' : 'en'; } catch { return 'en'; }
}
let lang = initialLang();
export const t = (en, zh) => lang === 'zh' ? zh : en;
export const language = () => lang;
export function setLanguage(next) {
    lang = next === 'zh' ? 'zh' : 'en';
    try { localStorage.setItem(LANG_KEY, lang); } catch { /* Private windows keep the session choice only. */ }
}

const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
};
const button = (label, cls, action) => {
    const b = el('button', cls, label); b.type = 'button'; if (action) b.onclick = action; return b;
};
const widget = (node, name) => node?.widgets?.find(w => w.name === name);
const value = (node, name, fallback) => widget(node, name)?.value ?? fallback;
function set(node, name, v) {
    const w = widget(node, name); if (!w) return;
    w.value = v; w.callback?.(v); node.graph?.change?.(); node.setDirtyCanvas?.(true, true);
}
const seconds = s => Number.isFinite(s) ? (s < 60 ? `${s.toFixed(1)} s` : `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`) : '—';
const randomSeed = () => Math.floor(Math.random() * 2 ** 32);
const isVideo = name => /\.(mp4|webm|mov|mkv)$/i.test(name || '');

const MODES = [
    ['text', () => t('Text', '文生视频'), () => t('Prompt only', '仅提示词')],
    ['first_frame', () => t('First frame', '首帧'), () => t('Animate an image', '让图片动起来')],
    ['reference', () => t('Reference', '参考图'), () => t('Up to 4 images', '最多 4 张')],
];
const MODE_NAME = { first_frame: () => t('first frame', '首帧'), reference: () => t('reference images', '参考图') };
const RESOLUTIONS = ['768p', '480p'];
const CANVAS = { '768p': '1344 × 768', '480p': '864 × 480' };
const PIPELINES = {
    one_stage: [() => t('One stage', '单阶段'), () => t('8-step LoRA · Sol-H3 acceleration', '8 步 LoRA · Sol-H3 加速')],
    two_stage: [() => t('Two stage', '两阶段'), () => t('H3 draft · LTX refine', 'H3 草稿 · LTX 精修')],
};
const STAGES = {
    queued: [() => t('Queued', '排队中'), .03],
    loading: [() => t('Loading model', '加载模型'), .08],
    conditioning: [() => t('Encoding prompt', '编码提示词'), .18],
    generating: [() => t('Generating', '生成中'), null],
    denoising: [() => t('Denoising', '去噪'), .25],
    refining: [() => t('Refining', '精修'), .7],
    decoding: [() => t('Decoding video', '解码视频'), .86],
    saving: [() => t('Saving', '保存中'), .95],
    done: [() => t('Done', '完成'), 1],
    error: [() => t('Failed', '失败'), 1],
};
const EXAMPLES = [
    [() => t('Lighthouse storm', '风暴灯塔'), 'A slow aerial push-in toward a lone lighthouse on black cliffs as a storm rolls in at dusk, waves exploding into white spray, the beam sweeping through rain. overall_soundscape: Roaring surf, gusting wind and distant thunder.'],
    [() => t('Neon noodle bar', '霓虹面馆'), 'Handheld close-up in a tiny Tokyo noodle bar at night, steam curling past neon signs reflected in a rain-streaked window, a chef ladling broth with practiced calm. overall_soundscape: Sizzling pans, low chatter and rain on glass.'],
    [() => t('Desert caravan', '沙漠驼队'), 'Golden-hour tracking shot of a camel caravan cresting a vast dune, long shadows rippling across the sand, fine grains lifting in the wind. overall_soundscape: Soft wind, camel bells and footsteps in sand.'],
    [() => t('Glacier kayak', '冰川皮划艇'), 'Low wide shot gliding beside a red kayak through turquoise glacial water, towering blue ice walls on both sides, a seal surfacing nearby. overall_soundscape: Paddles dipping, dripping ice and distant cracking.'],
];

/** Build the Studio around injected ComfyUI objects. Returns {openStudio, current}. */
export function createStudio({ app, api, onNodes = () => {} }) {
    let current = null;
    const view = (file, subfolder = '', type = 'output') =>
        api.apiURL('/view?' + new URLSearchParams({ filename: file, subfolder, type }));
    const thumbURL = id => api.apiURL('/solh3/library/thumbnail?' + new URLSearchParams({ id }));
    const galleryURL = name => api.apiURL('/solh3/gallery/file?' + new URLSearchParams({ name }));
    const json = async (path, options) => {
        const response = await api.fetchApi(path, { cache: 'no-store', ...options });
        if (!response.ok) throw new Error(`${path}: ${response.status}`);
        return response.json();
    };

    function openStudio(node, { tab = 'studio' } = {}) {
        if (current?.node === node && current.dialog.open) { current.dialog.focus(); return current.dialog; }
        current?.dialog.close();
        const cleanup = [];
        const dialog = el('dialog', 'sh-studio');
        dialog.lang = lang === 'zh' ? 'zh-CN' : 'en';
        dialog.setAttribute('aria-label', t('Sol-H3 Studio', 'Sol-H3 创作面板'));
        current = { node, dialog };

        // ---- Header: brand, view switch, device, language, library.
        const header = el('header', 'sh-header');
        const brand = el('div', 'sh-brand'); brand.append(wordmark(t('in ComfyUI', 'ComfyUI 版')));
        const views = el('div', 'sh-switch'); views.setAttribute('role', 'group');
        views.setAttribute('aria-label', t('View', '视图'));
        const studioTab = button(t('Studio', '创作'), 'sh-switch-item', () => showTab('studio'));
        const galleryTab = button(t('Gallery', '作品墙'), 'sh-switch-item', () => showTab('gallery'));
        const nodesTab = button(t('Nodes', '节点'), 'sh-switch-item', () => { dialog.close(); onNodes(node); });
        views.append(studioTab, galleryTab, nodesTab);
        const actions = el('div', 'sh-header-actions');
        const device = el('span', 'sh-device');
        device.append(el('i', 'sh-dot'), el('span', null, t('Local · RTX 5090', '本地 · RTX 5090')));
        const langSwitch = el('div', 'sh-lang'); langSwitch.setAttribute('role', 'group');
        langSwitch.setAttribute('aria-label', t('Language', '语言'));
        for (const [id, label] of [['en', 'EN'], ['zh', '中文']]) {
            const b = button(label, 'sh-lang-item', () => {
                if (id === lang) return;
                setLanguage(id);
                const reopen = { tab: activeTab };
                dialog.close();
                openStudio(node, reopen);
            });
            b.setAttribute('aria-pressed', String(id === lang)); b.lang = id === 'zh' ? 'zh-CN' : 'en';
            langSwitch.append(b);
        }
        const libraryButton = button(t('Library', '作品库'), 'sh-ghost', () => openLibrary());
        libraryButton.prepend(icon('grid'));
        const close = button('', 'sh-icon-button', () => dialog.close());
        close.append(icon('close')); close.title = t('Close (Esc)', '关闭 (Esc)');
        close.setAttribute('aria-label', close.title);
        actions.append(device, langSwitch, libraryButton, close);
        header.append(brand, views, actions);

        // ---- Body.
        const body = el('div', 'sh-body');
        const controls = el('div', 'sh-controls'), scroll = el('div', 'sh-controls-scroll');
        const footer = el('div', 'sh-controls-footer');
        controls.append(scroll, footer);
        const output = el('div', 'sh-output');
        body.append(controls, output);
        dialog.append(header, body);
        const section = (label, aside) => {
            const wrap = el('section', 'sh-section'), head = el('div', 'sh-label');
            head.append(el('span', null, label)); if (aside) head.append(aside);
            wrap.append(head); scroll.append(wrap); return wrap;
        };

        // Prompt.
        const count = el('span', 'sh-count');
        const promptSection = section(t('Prompt', '提示词'), count);
        const promptBox = el('div', 'sh-prompt');
        const prompt = el('textarea');
        prompt.value = value(node, 'prompt', '');
        prompt.placeholder = t('Describe the shot: subject, camera, light, motion… then the soundscape.',
            '描述镜头：主体、运镜、光线、动作……再描述声音氛围。');
        prompt.setAttribute('aria-label', t('Prompt', '提示词'));
        prompt.spellcheck = false;
        const examples = el('div', 'sh-examples');
        examples.append(el('span', 'sh-examples-label', t('Try', '试试')));
        for (const [title, text] of EXAMPLES) {
            const chip = button(title(), 'sh-example', () => { prompt.value = text; prompt.dispatchEvent(new Event('input')); prompt.focus(); });
            chip.title = text; examples.append(chip);
        }
        const galleryLink = button(t('Browse the gallery →', '浏览作品墙 →'), 'sh-link sh-examples-more', () => showTab('gallery'));
        examples.append(galleryLink);
        promptBox.append(prompt);
        promptSection.append(promptBox, examples);
        const syncPrompt = () => {
            count.textContent = t(`${prompt.value.length} chars`, `${prompt.value.length} 字`);
            set(node, 'prompt', prompt.value); update();
        };
        prompt.addEventListener('input', syncPrompt);

        // Mode + images.
        const modeSection = section(t('Mode', '模式'));
        let mode = value(node, 'mode', 'text');
        const modeRow = el('div', 'sh-segmented sh-modes'); modeRow.setAttribute('role', 'radiogroup');
        for (const [id, label, hint] of MODES) {
            const b = button('', 'sh-segment'); b.dataset.id = id;
            b.append(icon(id), el('strong', null, label()), el('small', null, hint()));
            b.setAttribute('role', 'radio');
            b.onclick = () => {
                if (b.dataset.unsupported === 'true') return;
                mode = id; set(node, 'mode', id); renderMode(); update();
            };
            modeRow.append(b);
        }
        let images = String(value(node, 'image', '') || '').split(',').map(s => s.trim()).filter(Boolean);
        const drop = el('div', 'sh-drop'), file = el('input'); file.type = 'file'; file.accept = 'image/*'; file.hidden = true;
        const dropEmpty = el('button', 'sh-drop-empty'); dropEmpty.type = 'button';
        const dropTitle = el('strong'), dropHint = el('small', null, t('PNG, JPG or WebP · stored in ComfyUI input', 'PNG、JPG 或 WebP · 存入 ComfyUI input'));
        dropEmpty.append(icon('upload'), dropTitle, dropHint);
        dropEmpty.onclick = () => file.click();
        const thumbs = el('div', 'sh-thumbs');
        const uploadNote = el('p', 'sh-note');
        drop.append(dropEmpty, thumbs, file);
        modeSection.append(modeRow, drop, uploadNote);
        const limit = () => mode === 'first_frame' ? 1 : 4;
        const inputName = name => { const split = name.lastIndexOf('/'); return [name.slice(split + 1), split < 0 ? '' : name.slice(0, split)]; };
        function renderImages() {
            thumbs.replaceChildren();
            for (const name of images) {
                const tile = el('figure', 'sh-thumb'), img = el('img');
                const [leaf, folder] = inputName(name);
                img.src = view(leaf, folder, 'input');
                img.alt = name; img.loading = 'lazy';
                const remove = button('', 'sh-thumb-remove', () => { images = images.filter(n => n !== name); saveImages(); });
                remove.append(icon('close')); remove.setAttribute('aria-label', t('Remove', '移除'));
                tile.append(img, remove); thumbs.append(tile);
            }
            if (images.length && images.length < limit()) {
                const add = button('', 'sh-thumb sh-thumb-add', () => file.click());
                add.append(icon('plus')); add.setAttribute('aria-label', t('Add image', '添加图片'));
                thumbs.append(add);
            }
            dropEmpty.hidden = images.length > 0;
            thumbs.hidden = images.length === 0;
            file.multiple = mode === 'reference';
        }
        function saveImages() { images = images.slice(0, limit()); set(node, 'image', images.join(',')); renderImages(); update(); }
        async function uploadFiles(files) {
            const names = [];
            for (const f of files) {
                const form = new FormData(); form.append('image', f);
                const response = await api.fetchApi('/upload/image', { method: 'POST', body: form });
                if (!response.ok) throw new Error(`${response.status}`);
                const saved = await response.json();
                names.push(saved.subfolder ? `${saved.subfolder}/${saved.name}` : saved.name);
            }
            return names;
        }
        async function upload(files) {
            const wanted = [...files].filter(f => f.type.startsWith('image/')).slice(0, limit() - (mode === 'first_frame' ? 0 : images.length));
            if (!wanted.length) return;
            drop.dataset.busy = 'true'; uploadNote.textContent = t('Uploading…', '上传中…'); uploadNote.dataset.error = 'false';
            try {
                const names = await uploadFiles(wanted);
                images = mode === 'first_frame' ? names : [...images, ...names];
                uploadNote.textContent = '';
                saveImages();
            } catch (error) {
                uploadNote.textContent = t('Upload failed: ', '上传失败：') + error.message; uploadNote.dataset.error = 'true';
            } finally { drop.dataset.busy = 'false'; file.value = ''; }
        }
        file.onchange = () => upload(file.files);
        for (const name of ['dragenter', 'dragover']) drop.addEventListener(name, e => { e.preventDefault(); drop.dataset.over = 'true'; });
        for (const name of ['dragleave', 'drop']) drop.addEventListener(name, () => { drop.dataset.over = 'false'; });
        drop.addEventListener('drop', e => { e.preventDefault(); upload(e.dataTransfer?.files || []); });
        function renderMode() {
            for (const b of modeRow.children) b.setAttribute('aria-checked', String(b.dataset.id === mode));
            drop.hidden = mode === 'text';
            dropTitle.textContent = mode === 'reference' ? t('Drop up to 4 images or click to upload', '拖入最多 4 张图片或点击上传')
                : t('Drop an image or click to upload', '拖入图片或点击上传');
            if (mode === 'first_frame' && images.length > 1) saveImages();
            uploadNote.textContent = mode === 'reference' ? t('Subjects, characters or style to carry into the video.', '要带入视频的主体、角色或风格。')
                : mode === 'first_frame' ? t('The video starts from this frame.', '视频从这张图开始。') : '';
            uploadNote.dataset.error = 'false';
            renderImages();
        }

        // Pipeline = profile: one stage (8-step LoRA, Sol-H3 acceleration) or two stage (H3 draft + LTX refine).
        const pipelineSection = section(t('Pipeline', '流程'));
        const pipelineRow = el('div', 'sh-pipelines'); pipelineRow.setAttribute('role', 'radiogroup');
        pipelineSection.append(pipelineRow);
        let profiles = (widget(node, 'profile')?.options?.values || []).map((name, i) => ({
            name, label: name, modes: ['text'], resolutions: ['768p'], pipeline: 'one_stage', steps: null, default: i === 0 }));
        const profile = () => profiles.find(p => p.name === value(node, 'profile')) || profiles[0];
        function renderPipelines() {
            pipelineRow.replaceChildren();
            const chosen = profile()?.name;
            for (const p of profiles) {
                const b = button('', 'sh-pipeline'); b.dataset.id = p.name;
                const [title, about] = PIPELINES[p.pipeline] || [() => p.label, () => ''];
                const head = el('span', 'sh-pipeline-head');
                head.append(el('strong', null, profiles.filter(q => q.pipeline === p.pipeline).length > 1 ? p.label : title()));
                // latency_s: one number, or {resolution: seconds} measured per resolution.
                const lat = typeof p.latency_s === 'object' && p.latency_s ? p.latency_s[resolution] ?? p.latency_s[p.resolutions?.[0]] : p.latency_s;
                if (Number.isFinite(lat)) head.append(el('b', 'sh-pipeline-time', `≈ ${Math.round(lat)} s`));
                const facts = [p.steps ? t(`${p.steps} steps`, `${p.steps} 步`) : null,
                    p.modes.length > 1 ? p.modes.map(m => MODES.find(x => x[0] === m)?.[1]()).join(' · ') : t('Text only', '仅文生视频')];
                b.append(head, el('small', null, about()), el('span', 'sh-pipeline-facts', facts.filter(Boolean).join('  ·  ')));
                b.title = p.name; b.setAttribute('role', 'radio'); b.setAttribute('aria-checked', String(p.name === chosen));
                b.onclick = () => {
                    set(node, 'profile', p.name);
                    // Keep the request valid: fall back to what the new pipeline supports.
                    if (!p.modes.includes(mode)) { mode = 'text'; set(node, 'mode', 'text'); renderMode(); }
                    if (!p.resolutions.includes(resolution)) { resolution = p.resolutions[0]; set(node, 'resolution', resolution); }
                    renderPipelines(); renderFormat(); update();
                };
                pipelineRow.append(b);
            }
        }

        // Format: resolution (per pipeline), 16:9, fixed 5 s duration.
        const formatSection = section(t('Format', '画面'));
        let resolution = value(node, 'resolution', '768p');
        const resolutionRow = el('div', 'sh-chips'); resolutionRow.setAttribute('role', 'radiogroup');
        const formatNote = el('p', 'sh-note');
        const formatGrid = el('div', 'sh-format');
        const resolutionBlock = el('div', 'sh-field'), durationBlock = el('div', 'sh-field');
        resolutionBlock.append(el('span', 'sh-field-label', t('Resolution', '分辨率')), resolutionRow);
        const duration = el('div', 'sh-chips'); const fixed = button('5 s', 'sh-chip'); fixed.setAttribute('aria-checked', 'true'); fixed.tabIndex = -1;
        duration.append(fixed);
        durationBlock.append(el('span', 'sh-field-label', t('Duration', '时长')), duration);
        formatGrid.append(resolutionBlock, durationBlock);
        formatSection.append(formatGrid, formatNote);
        function renderFormat() {
            const supported = profile()?.resolutions || ['768p'];
            if (!supported.includes(resolution)) { resolution = supported[0]; set(node, 'resolution', resolution); }
            resolutionRow.replaceChildren();
            for (const id of RESOLUTIONS) {
                const b = button('', 'sh-chip'); b.dataset.id = id; b.append(el('span', null, id));
                const ok = supported.includes(id);
                b.disabled = !ok; b.setAttribute('role', 'radio'); b.setAttribute('aria-checked', String(id === resolution));
                if (!ok) b.title = t(`${PIPELINES[profile()?.pipeline]?.[0]() || 'This pipeline'} renders ${supported.join(', ')}`,
                    `${PIPELINES[profile()?.pipeline]?.[0]() || '该流程'}仅支持 ${supported.join('、')}`);
                b.onclick = () => { resolution = id; set(node, 'resolution', id); renderFormat(); renderPipelines(); update(); };
                resolutionRow.append(b);
            }
            formatNote.textContent = t(`${CANVAS[resolution]} · 16:9 · 24 fps · 5 s with audio`, `${CANVAS[resolution]} · 16:9 · 24 fps · 5 秒 · 含音频`);
        }

        // Seed.
        const seedSection = section(t('Seed', '种子'));
        const seedRow = el('div', 'sh-seed');
        let seedMode = 'random';
        const seedInput = el('input'); seedInput.type = 'number'; seedInput.min = '0'; seedInput.step = '1';
        seedInput.value = String(value(node, 'seed', 42)); seedInput.setAttribute('aria-label', t('Seed', '种子'));
        const seedChips = el('div', 'sh-segmented sh-seed-mode'); seedChips.setAttribute('role', 'radiogroup');
        for (const [id, label] of [['random', t('Random', '随机')], ['fixed', t('Fixed', '固定')]]) {
            const b = button(label, 'sh-chip'); b.dataset.id = id; b.setAttribute('role', 'radio');
            b.setAttribute('aria-checked', String(id === seedMode));
            b.onclick = () => { seedMode = id; seedInput.disabled = id === 'random'; for (const c of seedChips.children) c.setAttribute('aria-checked', String(c === b)); };
            seedChips.append(b);
        }
        seedInput.disabled = true;
        seedInput.addEventListener('change', () => { const n = Math.max(0, Math.floor(Number(seedInput.value) || 0)); seedInput.value = String(n); set(node, 'seed', n); });
        const dice = button('', 'sh-icon-button', () => { const n = randomSeed(); seedInput.value = String(n); set(node, 'seed', n); seedMode = 'fixed'; seedInput.disabled = false; for (const c of seedChips.children) c.setAttribute('aria-checked', String(c.dataset.id === 'fixed')); });
        dice.append(icon('dice')); dice.title = t('New seed', '换一个种子'); dice.setAttribute('aria-label', dice.title);
        seedRow.append(seedChips, seedInput, dice);
        seedSection.append(seedRow);

        // Footer: Generate.
        const generate = button('', 'sh-generate');
        const generateLabel = el('span', null, t('Generate', '生成'));
        generate.append(icon('spark'), generateLabel, el('kbd', null, navigator.platform?.includes('Mac') ? '⌘ ↵' : 'Ctrl ↵'));
        const reason = el('p', 'sh-reason');
        footer.append(generate, reason);

        // ---- Output: stage + recent rail.
        const stageWrap = el('div', 'sh-stage-wrap'), stage = el('div', 'sh-stage');
        stageWrap.append(stage);
        const empty = el('div', 'sh-empty');
        const emptyNote = el('span');
        empty.append(el('div', 'sh-empty-mark'), el('strong', null, t('Your video appears here', '生成的视频会显示在这里')), emptyNote);
        const card = el('div', 'sh-progress'); card.hidden = true;
        const cardTop = el('div', 'sh-progress-top');
        const stageName = el('strong', 'sh-stage-name'), stageMeta = el('span', 'sh-stage-meta');
        const cancel = button(t('Cancel', '取消'), 'sh-ghost sh-small', cancelRun);
        const cardTitle = el('div'); cardTitle.append(el('i', 'sh-live'), stageName);
        cardTop.append(cardTitle, cancel);
        const bar = el('div', 'sh-bar'), fill = el('div', 'sh-bar-fill'); bar.append(fill);
        const cardStats = el('div', 'sh-progress-stats');
        const stepStat = el('span'), elapsedStat = el('span');
        cardStats.append(stepStat, stageMeta, elapsedStat);
        const timeline = el('ol', 'sh-timeline');
        function renderTimeline() {
            const two = profile()?.pipeline === 'two_stage';
            const steps = [['loading', t('Load', '加载')], ['conditioning', t('Encode', '编码')], ['denoising', two ? t('Draft', '草稿') : t('Denoise', '去噪')],
                ...(two ? [['refining', t('Refine', '精修')]] : []), ['decoding', t('Decode', '解码')], ['saving', t('Save', '保存')]];
            timeline.replaceChildren(...steps.map(([id, label]) => { const li = el('li', null, label); li.dataset.id = id; return li; }));
            timeline.style.setProperty('--sh-steps', String(steps.length));
        }
        card.append(cardTop, bar, cardStats, timeline);
        const player = el('div', 'sh-player'); player.hidden = true;
        const video = el('video'); video.controls = true; video.loop = true; video.playsInline = true; video.preload = 'metadata';
        const resultBar = el('div', 'sh-result-bar');
        const resultMeta = el('div', 'sh-result-meta');
        const download = el('a', 'sh-ghost sh-small'); download.append(icon('download'), el('span', null, t('Download', '下载')));
        download.setAttribute('download', '');
        resultBar.append(resultMeta, download);
        player.append(video);
        const failure = el('div', 'sh-failure'); failure.hidden = true;
        stage.append(empty, player, card, failure);
        const resultRow = el('div', 'sh-result-row'); resultRow.append(resultBar); resultRow.hidden = true;

        const rail = el('section', 'sh-rail');
        const railHead = el('div', 'sh-rail-head');
        railHead.append(el('span', 'sh-label-text', t('Recent', '最近生成')), button(t('View all', '查看全部'), 'sh-link', () => openLibrary()));
        const railList = el('div', 'sh-rail-list');
        rail.append(railHead, railList);
        output.append(stageWrap, resultRow, rail);

        // ---- State.
        let promptId = null, settled = false, running = false, submitting = false, last = null, tick = null;
        function blocked() {
            if (running || submitting) return t('Generating…', '生成中…');
            if (!prompt.value.trim()) return t('Describe your scene to start.', '先描述你的画面。');
            const p = profile();
            if (!p || /SOL_ENGINE_CONFIG/.test(p.name)) return t('Configure a Sol-H3 profile first (config.json).', '请先配置 Sol-H3 profile（config.json）。');
            if (mode !== 'text' && !p.modes.includes(mode))
                return t(`Switch pipeline to use ${MODE_NAME[mode]()}.`, `切换流程以使用${MODE_NAME[mode]()}模式。`);
            if (mode !== 'text' && !images.length) return mode === 'first_frame'
                ? t('Upload a first-frame image.', '请上传首帧图片。') : t('Upload at least one reference image.', '请至少上传一张参考图。');
            return '';
        }
        function update() {
            const why = blocked();
            generate.disabled = Boolean(why);
            generate.dataset.busy = String(running || submitting);
            generateLabel.textContent = running || submitting ? t('Generating…', '生成中…') : t('Generate', '生成');
            reason.textContent = why && !(running || submitting) ? why : '';
            const p = profile();
            for (const b of modeRow.children) {
                const supported = b.dataset.id === 'text' || Boolean(p?.modes.includes(b.dataset.id));
                b.dataset.unsupported = String(!supported);
                b.setAttribute('aria-disabled', String(!supported));
                const other = profiles.find(q => q.modes.includes(b.dataset.id));
                b.querySelector('small').textContent = supported ? MODES.find(m => m[0] === b.dataset.id)[2]()
                    : other ? t(`With ${PIPELINES[other.pipeline]?.[0]() || other.label}`, `需切换到${PIPELINES[other.pipeline]?.[0]() || other.label}`)
                    : t('Not in this setup', '当前配置不支持');
            }
            emptyNote.textContent = t(`5 s · ${CANVAS[resolution]} · sound included`, `5 秒 · ${CANVAS[resolution]} · 含声音`);
        }

        // Progress.
        function showProgress(message) {
            // 'done' means the engine finished; the card stays at Saving until a video arrives.
            if (message.stage === 'done') message = { ...message, stage: 'saving' };
            const [label, fraction] = STAGES[message.stage] || [() => message.stage, null];
            if (message.stage === 'error') { running = false; settled = true; stopTick(); card.hidden = true; showFailure(message.message); update(); return; }
            running = true; failure.hidden = true; empty.hidden = true; card.hidden = false;
            player.dataset.dim = String(!player.hidden);
            last = { ...message, at: performance.now() };
            stageName.textContent = label();
            stageMeta.textContent = message.stage === 'loading' ? t('First run loads the model', '首次运行需加载模型') : (message.message || '');
            const steps = message.steps, step = message.step;
            stepStat.textContent = steps ? t(`Step ${step ?? 0}/${steps}`, `第 ${step ?? 0}/${steps} 步`) : '';
            let f = fraction;
            if (message.stage === 'denoising' && steps) f = .25 + .6 * Math.min(1, (step || 0) / steps);
            bar.dataset.indeterminate = String(f == null);
            fill.style.width = f == null ? '' : `${Math.round(f * 100)}%`;
            const order = ['queued', 'loading', 'conditioning', 'denoising', 'refining', 'decoding', 'saving'];
            const at = order.indexOf(message.stage);
            for (const li of timeline.children) {
                const i = order.indexOf(li.dataset.id);
                li.dataset.state = at < 0 ? 'idle' : i < at ? 'done' : i === at ? 'active' : 'idle';
            }
            startTick(); update();
        }
        function startTick() {
            if (tick) return;
            const render = () => { if (last) elapsedStat.textContent = seconds((last.elapsed_s || 0) + (performance.now() - last.at) / 1000); };
            render(); tick = setInterval(render, 250);
        }
        function stopTick() { clearInterval(tick); tick = null; }
        function showFailure(text) {
            failure.replaceChildren(icon('alert'), el('strong', null, t('Generation failed', '生成失败')), el('p', null, text || ''));
            failure.hidden = false; empty.hidden = true;
        }
        function showVideo(item) {
            const url = item.url || view(item.file, item.subfolder || '', item.type || 'output');
            if (item.subfolder?.startsWith('SolH3')) video.poster = thumbURL(`${item.subfolder}/${item.file}`);
            else video.removeAttribute('poster');
            if (item.poster) video.poster = item.poster;
            video.src = url; download.href = url; download.setAttribute('download', item.file);
            player.hidden = false; empty.hidden = true; failure.hidden = true; player.dataset.dim = 'false';
            resultRow.hidden = false;
            const bits = [];
            if (Number.isFinite(item.e2e_s)) bits.push([t('Generated in', '用时'), seconds(item.e2e_s)]);
            const p = profiles.find(q => q.name === item.profile);
            if (item.profile) bits.push([t('Pipeline', '流程'), p ? (PIPELINES[p.pipeline]?.[0]() || p.label) : item.profile]);
            if (item.seed != null) bits.push([t('Seed', '种子'), String(item.seed)]);
            resultMeta.replaceChildren(...bits.map(([k, v]) => { const s = el('span'); s.append(el('small', null, k), el('b', null, v)); return s; }));
            for (const tile of railList.children) tile.setAttribute('aria-current', String(tile.dataset.id === `${item.subfolder}/${item.file}`));
            video.play?.().catch(() => {});
        }
        let pendingTiming = null;
        const onProgress = e => {
            const m = e.detail; if (!m || (settled && m.prompt_id === promptId)) return;
            if (promptId ? m.prompt_id === promptId : (String(m.node) === String(node.id) && !['done', 'error'].includes(m.stage))) {
                if (!promptId && m.prompt_id) promptId = m.prompt_id;
                showProgress(m);
            }
        };
        const onExecuted = e => {
            const { node: id, output: out, prompt_id } = e.detail || {};
            if (!promptId || prompt_id !== promptId) return;
            if (String(id) === String(node.id) && out?.text?.[0]) {
                try { pendingTiming = JSON.parse(out.text[0]); } catch { pendingTiming = null; }
            }
            const files = [...(out?.images || []), ...(out?.videos || []), ...(out?.gifs || []), ...(out?.solh3_video || [])]
                .filter(f => isVideo(f.filename));
            if (files.length) {
                running = false; settled = true; stopTick(); card.hidden = true;
                const f = files[0];
                showVideo({ file: f.filename, subfolder: f.subfolder, type: f.type, e2e_s: pendingTiming?.e2e_s,
                    profile: pendingTiming?.profile, seed: pendingTiming?.seed ?? value(node, 'seed') });
                update(); refreshRecent();
            }
        };
        const onError = e => {
            if (!promptId || e.detail?.prompt_id !== promptId) return;
            running = false; settled = true; stopTick(); card.hidden = true;
            showFailure(e.detail?.exception_message || t('See the ComfyUI console.', '请查看 ComfyUI 控制台。')); update();
        };
        const onInterrupted = e => {
            if (!promptId || e.detail?.prompt_id !== promptId) return;
            running = false; settled = true; stopTick(); card.hidden = true; empty.hidden = !player.hidden; player.dataset.dim = 'false'; update();
        };
        const events = [['solh3_progress', onProgress], ['executed', onExecuted], ['execution_error', onError], ['execution_interrupted', onInterrupted]];
        for (const [name, f] of events) api.addEventListener(name, f);
        cleanup.push(() => { for (const [name, f] of events) api.removeEventListener(name, f); });
        const poll = async () => {
            try { for (const m of (await json('/solh3/progress')).progress || []) onProgress({ detail: m }); }
            catch { /* Live WebSocket updates continue across a failed poll. */ }
        };
        const pollTimer = setInterval(poll, 2000); cleanup.push(() => clearInterval(pollTimer), stopTick);

        async function submit() {
            if (generate.disabled) return;
            submitting = true; update();
            try {
                if (seedMode === 'random') { const n = randomSeed(); seedInput.value = String(n); set(node, 'seed', n); }
                // Studio owns the seed; ComfyUI's after-generate control would change it twice.
                const control = widget(node, 'control_after_generate') || widget(node, 'seed')?.linkedWidgets?.[0];
                if (control) control.value = 'fixed';
                set(node, 'prompt', prompt.value); set(node, 'mode', mode); set(node, 'image', mode === 'text' ? '' : images.join(','));
                set(node, 'backend', 'local'); set(node, 'resolution', resolution);
                renderTimeline();
                const snapshot = await app.graphToPrompt();
                const reply = await api.queuePrompt(0, snapshot);
                if (!reply?.prompt_id) throw new Error(t('ComfyUI did not accept the request.', 'ComfyUI 未接受该请求。'));
                promptId = reply.prompt_id; settled = false; pendingTiming = null;
                showProgress({ stage: 'queued', elapsed_s: 0, prompt_id: promptId, node: node.id });
            } catch (error) {
                showFailure(error?.response?.error?.message || error.message || String(error));
            } finally { submitting = false; update(); }
        }
        async function cancelRun() {
            cancel.disabled = true;
            try { await api.fetchApi('/solh3/cancel', { method: 'POST' }); } catch { /* Interrupt event or poll settles state. */ }
            finally { cancel.disabled = false; }
        }
        generate.onclick = submit;

        // Recent rail + library.
        function tile(item, cls) {
            const b = button('', cls); b.dataset.id = `${item.subfolder}/${item.file}`;
            const frame = el('div', 'sh-tile-frame');
            const img = el('img'); img.alt = ''; img.loading = 'lazy'; img.src = thumbURL(item.id || `${item.subfolder}/${item.file}`);
            img.onerror = () => {
                // No PyAV: let the browser decode a poster frame from the video itself.
                const v = el('video'); v.muted = true; v.preload = 'metadata';
                v.src = view(item.file, item.subfolder, item.type) + '#t=0.1'; img.replaceWith(v);
            };
            frame.append(img, el('span', 'sh-tile-play'));
            if (Number.isFinite(item.e2e_s)) frame.append(el('span', 'sh-tile-badge', seconds(item.e2e_s)));
            b.append(frame);
            b.title = item.prompt || item.file;
            b.onclick = () => { showTab('studio'); showVideo(item); closeLibrary(); };
            return b;
        }
        async function refreshRecent() {
            try {
                const items = (await json('/solh3/library?limit=8')).items || [];
                railList.replaceChildren(...items.map(item => tile(item, 'sh-tile')));
                rail.dataset.empty = String(!items.length);
                if (!items.length) railList.append(el('p', 'sh-rail-empty', t('Videos you generate will collect here.', '生成的视频会收集在这里。')));
                return items;
            } catch { rail.dataset.empty = 'true'; return []; }
        }
        let library = null;
        async function openLibrary() {
            if (library) return;
            library = el('div', 'sh-library'); library.setAttribute('role', 'region');
            library.setAttribute('aria-label', t('Library', '作品库'));
            const head = el('div', 'sh-library-head');
            const back = button('', 'sh-icon-button', closeLibrary); back.append(icon('close')); back.setAttribute('aria-label', t('Close library', '关闭作品库'));
            const title = el('div'); title.append(el('h2', null, t('Library', '作品库')), el('span', 'sh-muted', t('Saved under output/SolH3', '保存在 output/SolH3')));
            head.append(title, back);
            const grid = el('div', 'sh-library-grid');
            for (let i = 0; i < 8; i++) grid.append(el('div', 'sh-card sh-skeleton'));
            library.append(head, grid); body.append(library); back.focus({ focusVisible: false });
            try {
                const items = (await json('/solh3/library?limit=60')).items || [];
                grid.replaceChildren(...items.map(item => {
                    const c = el('article', 'sh-card'); const tl = tile(item, 'sh-card-media');
                    const meta = el('div', 'sh-card-meta');
                    meta.append(el('p', null, item.prompt || item.file));
                    const facts = el('div', 'sh-card-facts');
                    for (const f of [item.mode && item.mode !== 'text' ? MODE_NAME[item.mode]?.() : null, item.resolution && item.resolution !== '768p' ? item.resolution : null,
                        item.created ? new Date(item.created * 1000).toLocaleString(lang === 'zh' ? 'zh-CN' : 'en') : null].filter(Boolean)) facts.append(el('span', null, f));
                    meta.append(facts); c.append(tl, meta); return c;
                }));
                if (!items.length) grid.replaceChildren(el('p', 'sh-rail-empty', t('No saved videos yet.', '还没有保存的视频。')));
            } catch (error) { grid.replaceChildren(el('p', 'sh-rail-empty', t('Library unavailable: ', '作品库不可用：') + error.message)); }
        }
        function closeLibrary() { library?.remove(); library = null; }

        // ---- Gallery: every showcase clip plays at once on a staggered wall; hover for sound, click to remix.
        const gallery = el('div', 'sh-gallery'); gallery.hidden = true;
        gallery.setAttribute('role', 'region'); gallery.setAttribute('aria-label', t('Gallery', '作品墙'));
        const galleryHead = el('div', 'sh-gallery-head');
        const galleryTitle = el('div');
        galleryTitle.append(el('h2', null, t('Made with Sol-H3', 'Sol-H3 作品')),
            el('span', 'sh-muted', t('Hover to listen. Click a clip to reuse its prompt and first frame.', '悬停试听，点击即可复用它的提示词和首帧。')));
        galleryHead.append(galleryTitle);
        const wall = el('div', 'sh-wall');
        gallery.append(galleryHead, wall);
        body.append(gallery);
        let galleryItems = null;
        const sound = { video: null };
        function hush() { if (sound.video) { sound.video.muted = true; sound.video.closest('.sh-wall-item')?.removeAttribute('data-sound'); sound.video = null; } }
        async function loadGallery() {
            if (galleryItems) return;
            galleryItems = [];
            try { galleryItems = (await json('/solh3/gallery')).items || []; } catch { galleryItems = []; }
            if (!galleryItems.length) {
                wall.replaceChildren(el('p', 'sh-rail-empty', t('No showcase clips configured (config.json "gallery").', '未配置作品墙（config.json 中的 "gallery"）。')));
                return;
            }
            wall.replaceChildren(...galleryItems.map((item, i) => {
                const fig = button('', 'sh-wall-item');
                fig.style.setProperty('--i', String(i));
                const v = el('video'); v.muted = true; v.loop = true; v.playsInline = true; v.autoplay = true; v.preload = 'auto';
                v.src = galleryURL(item.video);
                if (item.image) v.poster = galleryURL(item.image);
                const caption = el('span', 'sh-wall-caption');
                caption.append(el('strong', null, (lang === 'zh' && item.title_zh) || item.title),
                    el('small', null, [MODE_NAME[item.mode]?.() || t('text', '文生视频'), item.pipeline === 'two_stage' ? t('two stage', '两阶段') : null].filter(Boolean).join(' · ')));
                const speaker = el('span', 'sh-wall-sound'); speaker.append(icon('sound'));
                fig.append(v, caption, speaker);
                fig.addEventListener('pointerenter', () => { hush(); v.muted = false; sound.video = v; fig.dataset.sound = 'true'; v.play?.().catch(() => {}); });
                fig.addEventListener('pointerleave', hush);
                fig.onclick = () => remix(item);
                fig.title = item.prompt;
                return fig;
            }));
            for (const v of wall.querySelectorAll('video')) v.play?.().catch(() => {});
        }
        async function remix(item) {
            hush();
            prompt.value = item.prompt; prompt.dispatchEvent(new Event('input'));
            const wanted = item.mode || 'text';
            // Pick the clip's pipeline when configured, else one that can run its mode.
            const fit = q => (wanted === 'text' || q.modes.includes(wanted)) && (!item.pipeline || q.pipeline === item.pipeline);
            if (!fit(profile())) { const q = profiles.find(fit) || profiles.find(q => wanted === 'text' || q.modes.includes(wanted)); if (q) set(node, 'profile', q.name); }
            if (item.image && wanted !== 'text' && profile().modes.includes(wanted)) {
                try {
                    const blob = await (await fetch(galleryURL(item.image))).blob();
                    images = await uploadFiles([new File([blob], item.image, { type: blob.type || 'image/png' })]);
                    mode = wanted;
                } catch { mode = 'text'; images = []; }
            } else { mode = 'text'; images = []; }
            set(node, 'mode', mode); set(node, 'image', images.join(','));
            renderPipelines(); renderMode(); renderFormat(); update();
            showTab('studio');
            showVideo({ url: galleryURL(item.video), file: item.video });
            resultMeta.replaceChildren(...[[t('Showcase', '示例'), (lang === 'zh' && item.title_zh) || item.title]].map(([k, v]) => { const s = el('span'); s.append(el('small', null, k), el('b', null, v)); return s; }));
            prompt.focus({ preventScroll: true });
        }

        let activeTab = 'studio';
        function showTab(id) {
            activeTab = id;
            closeLibrary();
            gallery.hidden = id !== 'gallery';
            studioTab.setAttribute('aria-pressed', String(id === 'studio'));
            galleryTab.setAttribute('aria-pressed', String(id === 'gallery'));
            nodesTab.setAttribute('aria-pressed', 'false');
            if (id === 'gallery') { video.pause?.(); loadGallery(); for (const v of wall.querySelectorAll('video')) v.play?.().catch(() => {}); }
            else { hush(); for (const v of wall.querySelectorAll('video')) v.pause?.(); }
        }

        // Capabilities: pipelines with labels/steps/modes/resolutions, the device, the gallery.
        json('/solh3/capabilities').then(caps => {
            if (caps.profiles?.length) profiles = caps.profiles.map(p => ({ resolutions: ['768p'], pipeline: 'one_stage', ...p }));
            if (caps.device?.label) device.lastChild.textContent = t(`Local · ${caps.device.label}`, `本地 · ${caps.device.label}`);
            galleryTab.hidden = caps.gallery === false;
            if (!profiles.some(p => p.name === value(node, 'profile'))) set(node, 'profile', (profiles.find(p => p.default) || profiles[0])?.name);
            if (!profile().modes.includes(mode)) { mode = 'text'; set(node, 'mode', 'text'); }
            renderPipelines(); renderMode(); renderFormat(); renderTimeline(); update();
        }).catch(() => {});

        const keys = e => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); submit(); }
            if (e.key === 'Escape' && (library || activeTab === 'gallery')) { e.preventDefault(); e.stopPropagation(); if (library) closeLibrary(); else showTab('studio'); }
        };
        dialog.addEventListener('keydown', keys);
        dialog.addEventListener('cancel', e => { if (library || activeTab === 'gallery') e.preventDefault(); });
        dialog.addEventListener('close', () => {
            for (const f of cleanup) f(); video.pause(); hush();
            for (const v of wall.querySelectorAll('video')) { v.pause?.(); v.removeAttribute('src'); }
            dialog.remove();
            if (current?.dialog === dialog) current = null;
        });
        renderPipelines(); renderMode(); renderFormat(); renderTimeline(); syncPrompt();
        refreshRecent(); poll();
        document.body.append(dialog); dialog.showModal();
        showTab(tab);
        requestAnimationFrame(() => (tab === 'studio' ? prompt : galleryTab).focus({ preventScroll: true, focusVisible: false }));
        Object.assign(current, { showProgress, showVideo, refreshRecent, showTab, remix });
        return dialog;
    }

    return { openStudio, get current() { return current; } };
}

// Inline 20px stroke icons; no icon font or network request.
const ICONS = {
    text: 'M4 6h16M4 12h10M4 18h13',
    first_frame: 'M4 5h16v14H4zM4 15l4.5-4.5 3.5 3.5 2.5-2.5L20 17M15 9.5h.01',
    reference: 'M3 7h11v11H3zM7 3h14v14M7 14l2.5-2.5 2 2',
    upload: 'M12 16V4m0 0-4.5 4.5M12 4l4.5 4.5M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3',
    close: 'M6 6l12 12M18 6 6 18',
    plus: 'M12 5v14M5 12h14',
    grid: 'M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z',
    dice: 'M5 4h14a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1zM8.5 8.5h.01M15.5 15.5h.01M12 12h.01M15.5 8.5h.01M8.5 15.5h.01',
    spark: 'M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8zM19 16l.7 1.8 1.8.7-1.8.7L19 21l-.7-1.8-1.8-.7 1.8-.7z',
    download: 'M12 4v11m0 0-4-4m4 4 4-4M5 19h14',
    alert: 'M12 8v5m0 3h.01M10.3 3.9 2.6 17.5A2 2 0 0 0 4.3 20.5h15.4a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z',
    sound: 'M4 9.5h3.5L12 5.5v13l-4.5-4H4zM15.5 9a4 4 0 0 1 0 6M18 6.5a7.5 7.5 0 0 1 0 11',
};
export function icon(name) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 24 24'); svg.setAttribute('aria-hidden', 'true'); svg.classList.add('sh-icon');
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', ICONS[name] || ''); svg.append(path);
    return svg;
}
