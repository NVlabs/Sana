// ComfyUI extension: Studio button on SolH3Generate, floating Studio/Nodes pill,
// and auto-open for workflows marked extra.solh3_studio. Hook pattern after FreeVideo web/freevideo.js (Apache-2.0).
import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';
import { createStudio, t } from './studio.js';
import { wordmark } from './wordmark.js';

const STUDIO_WIDGETS = ['mode', 'image', 'backend', 'resolution'];
const isGenerate = node => node?.comfyClass === 'SolH3Generate' || node?.type === 'SolH3Generate';
const generateNode = () => {
    const selected = Object.values(app.canvas?.selected_nodes || {}).find(isGenerate);
    return selected || (app.graph?._nodes || []).find(isGenerate);
};

let pill = null;
let restored = () => {};
const graphRestored = new Promise(r => { restored = r; });
// Studio is the default view; leaving it for the node graph is remembered per browser.
const VIEW_KEY = 'solh3.view';
const remember = view => { try { localStorage.setItem(VIEW_KEY, view); } catch {} };
const preferred = () => {
    if (new URLSearchParams(location.search).get('solh3') === 'nodes') return 'nodes';
    try { return localStorage.getItem(VIEW_KEY) || 'studio'; } catch { return 'studio'; }
};
function showPill(visible) { if (pill) pill.hidden = !visible; }
const studio = createStudio({ app, api, onNodes: () => { remember('nodes'); showPill(true); } });

// Load the bundled Studio workflow when the open graph has no Sol-H3 node (e.g. ComfyUI's default).
async function ensureStudioGraph() {
    if (generateNode()) return generateNode();
    const response = await api.fetchApi('/workflow_templates');
    const templates = response.ok ? await response.json() : {};
    const pack = Object.keys(templates).find(k => (templates[k] || []).includes('Sol-H3-Studio'));
    if (!pack) return null;
    const workflow = await (await fetch(`/api/workflow_templates/${encodeURIComponent(pack)}/Sol-H3-Studio.json`)).json();
    // Its own workflow tab, so whatever the user had open is left untouched.
    await app.loadGraphData(workflow, true, true, 'Sol-H3 Studio');
    return generateNode();
}

async function openStudio(node = generateNode()) {
    node = node || await ensureStudioGraph();
    if (!node) return null;
    remember('studio');
    const dialog = studio.openStudio(node);
    showPill(false);
    dialog.addEventListener('close', () => showPill(true), { once: true });
    return dialog;
}

function installPill() {
    pill = document.createElement('aside'); pill.className = 'sh-pill';
    pill.setAttribute('aria-label', t('Sol-H3 workspace', 'Sol-H3 工作区'));
    const views = document.createElement('div'); views.className = 'sh-switch'; views.setAttribute('role', 'group');
    for (const [label, pressed, action] of [[t('Studio', '创作'), false, () => openStudio()], [t('Nodes', '节点'), true, () => {}]]) {
        const b = document.createElement('button'); b.type = 'button'; b.className = 'sh-switch-item';
        b.textContent = label; b.setAttribute('aria-pressed', String(pressed)); b.onclick = action;
        views.append(b);
    }
    pill.append(wordmark(''), views);
    pill.hidden = true;
    document.body.append(pill);
}

app.registerExtension({
    name: 'SolH3.Studio',
    setup() {
        installPill();
        showPill(true);
        // Open the Studio once ComfyUI has restored its last workflow (signalled by the first
        // afterConfigureGraph), or after a short wait when nothing is restored.
        Promise.race([graphRestored, new Promise(r => setTimeout(r, 3000))]).then(async () => {
            if (preferred() === 'studio' && !studio.current) await openStudio();
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== 'SolH3Generate') return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            // Studio edits these; the node view stays the original four widgets.
            for (const name of STUDIO_WIDGETS) {
                const w = this.widgets?.find(x => x.name === name);
                if (w) { w.type = 'hidden'; w.hidden = true; w.computeSize = () => [0, -4]; w.draw = () => {}; }
            }
            this.addWidget('button', t('Open Sol-H3 Studio', '打开 Sol-H3 创作面板'), null, () => openStudio(this), { serialize: false });
            requestAnimationFrame(() => showPill(!studio.current));
            return result;
        };
    },
    afterConfigureGraph() {
        restored();
        showPill(!studio.current);
        if (app.graph?.extra?.solh3_studio === true) {
            const node = generateNode();
            if (node) requestAnimationFrame(() => openStudio(node));
        }
    },

});
