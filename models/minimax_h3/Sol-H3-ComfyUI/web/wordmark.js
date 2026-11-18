// Typographic Sol-H3 wordmark; all artwork is served by this custom node (no CDN/fonts).
const css = document.createElement('link');
css.rel = 'stylesheet'; css.href = new URL('./theme.css', import.meta.url).href;
document.head.append(css);

export function wordmark(subtitle = 'in ComfyUI') {
    const mark = document.createElement('span'); mark.className = 'sh-wordmark';
    const image = document.createElement('img');
    image.src = new URL('./assets/solh3.svg', import.meta.url).href;
    image.alt = ''; image.width = 26; image.height = 26; image.draggable = false;
    const name = document.createElement('b'); name.textContent = 'Sol-H3';
    mark.append(image, name);
    if (subtitle) { const small = document.createElement('small'); small.textContent = subtitle; mark.append(small); }
    return mark;
}
