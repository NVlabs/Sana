(() => {
  "use strict";

  const config = window.SANA_VIDEO_MEDIA || { intro: [], sections: [] };
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const saveData = Boolean(navigator.connection && navigator.connection.saveData);
  let hlsLoader;
  let syncDemoPlayback = null;
  let demoPlaybackSuspended = false;

  function resolveMediaUrl(path) {
    if (!path || /^(?:[a-z]+:)?\/\//i.test(path) || path.startsWith("data:")) return path || "";
    const encodedPath = path.split("/").map(encodeURIComponent).join("/");
    return config.assetBase ? `${config.assetBase.replace(/\/$/, "")}/${encodedPath}` : encodedPath;
  }

  function normalizeMedia(media) {
    return {
      ...media,
      poster: resolveMediaUrl(media.poster),
      hls: resolveMediaUrl(media.hls),
      mp4: resolveMediaUrl(media.mp4)
    };
  }

  async function loadCuratedSections() {
    const curated = config.curated;
    if (!curated || !curated.metadata) return;

    try {
      const response = await fetch(resolveMediaUrl(curated.metadata));
      if (!response.ok) throw new Error(`metadata request failed (${response.status})`);
      const payload = await response.json();
      const records = Array.isArray(payload.records) ? payload.records : [];
      if (!records.length) return;
      const root = curated.root.replace(/\/$/, "");

      const itemsFor = mode => records.map((record, recordIndex) => {
        const index = Number.isInteger(record.selection_index) ? record.selection_index : recordIndex;
        const prefix = String(index).padStart(2, "0");
        const stem = `${prefix}_${record.source_sample_id}`;
        const poster = mode === "ti2v"
          ? `${root}/input-frames/${stem}.png`
          : `${root}/posters/t2v/${stem}.jpg`;
        return {
          selectionIndex: index,
          title: curated.titles?.[index] || `Sample ${index + 1}`,
          prompt: record.prompt,
          poster,
          duration: "8 s",
          resolution: "1280×736",
          width: 1280,
          height: 736,
          hls: "",
          mp4: `${root}/${mode}/${stem}.mp4`
        };
      });

      const t2vItems = itemsFor("t2v");
      const cinematicSamples = new Set(curated.cinematicSamples || []);
      const motionSamples = new Set(curated.motionSamples || []);
      const cinematicSection = config.sections.find(section => section.id === "cinematic");
      const motionSection = config.sections.find(section => section.id === "motion");
      if (cinematicSection) {
        cinematicSection.batches.push(t2vItems.filter(item => cinematicSamples.has(item.selectionIndex)));
      }
      if (motionSection) {
        motionSection.batches.push(t2vItems.filter(item => motionSamples.has(item.selectionIndex)));
      }

      const sections = [{
        id: "curated-ti2v",
        kicker: "03 · Image-conditioned generation",
        title: "Text + Image-to-Video",
        description: "The same ten prompts, conditioned on their corresponding first frames.",
        gridColumns: 2,
        items: itemsFor("ti2v")
      }];
      const insertionIndex = Math.max(0, config.sections.findIndex(section => section.id === "physical-ai"));
      config.sections.splice(insertionIndex, 0, ...sections);
    } catch (error) {
      console.warn("Curated SANA-Video samples could not be loaded.", error);
    }
  }

  function loadHlsLibrary() {
    if (window.Hls) return Promise.resolve(window.Hls);
    if (hlsLoader) return hlsLoader;
    hlsLoader = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "https://cdn.jsdelivr.net/npm/hls.js@1/dist/hls.min.js";
      script.async = true;
      script.onload = () => resolve(window.Hls);
      script.onerror = () => reject(new Error("Unable to load HLS player"));
      document.head.appendChild(script);
    });
    return hlsLoader;
  }

  async function attachStream(video, media, host, options = {}) {
    if (!video) return;
    video.dataset.shouldPlay = "true";
    if (video.dataset.attached === "true") {
      if (video._hls) video._hls.startLoad();
      video.play().catch(() => {});
      return;
    }

    const hls = media.hls || "";
    const mp4 = media.mp4 || "";
    const preferMp4 = Boolean(options.preferMp4 && mp4);
    if (!hls && !mp4) return;
    video.dataset.attached = "true";

    try {
      if (preferMp4) {
        video.src = mp4;
      } else if (hls && video.canPlayType("application/vnd.apple.mpegurl")) {
        video.src = hls;
      } else if (hls) {
        const Hls = await loadHlsLibrary();
        if (!Hls || !Hls.isSupported()) throw new Error("HLS unsupported");
        if (video.dataset.shouldPlay !== "true") return;
        const player = new Hls({
          startLevel: 0,
          capLevelToPlayerSize: true,
          maxBufferLength: options.hlsBufferLength || 4,
          backBufferLength: 0
        });
        player.loadSource(hls);
        player.attachMedia(video);
        video._hls = player;
      } else {
        video.src = mp4;
      }

      if (video.dataset.shouldPlay !== "true") return;
      video.addEventListener("playing", () => host && host.classList.add("is-playing"), { once: true });
      await video.play();
    } catch (_) {
      if (video.dataset.shouldPlay !== "true") return;
      if (preferMp4 && hls) {
        if (video._hls) video._hls.destroy();
        video._hls = null;
        video.removeAttribute("src");
        video.dataset.attached = "false";
        video.load();
        attachStream(video, { ...media, mp4: "" }, host, options);
      } else if (mp4) {
        video.src = mp4;
        video.play().catch(() => {});
      } else {
        video.dataset.attached = "false";
      }
    }
  }

  function pauseStream(video) {
    if (!video) return;
    video.dataset.shouldPlay = "false";
    if (video.dataset.attached !== "true") return;
    video.pause();
    if (video._hls) video._hls.stopLoad();
  }

  const dialog = document.querySelector("[data-lightbox]");
  const dialogVideo = dialog && dialog.querySelector("[data-lightbox-video]");
  const dialogCaption = dialog && dialog.querySelector("[data-lightbox-caption]");
  const dialogPlaceholder = dialog && dialog.querySelector("[data-lightbox-placeholder]");
  let lightboxSourceWidth = 1280;
  let lightboxSourceHeight = 720;

  function fitLightbox(width = lightboxSourceWidth, height = lightboxSourceHeight) {
    if (!dialog || !width || !height) return;
    lightboxSourceWidth = width;
    lightboxSourceHeight = height;
    const compact = window.matchMedia("(max-width: 600px)").matches;
    const horizontalMargin = compact ? 24 : 48;
    const verticalMargin = compact ? 40 : 72;
    const captionHeight = dialogCaption && !dialogCaption.hidden ? 50 : 0;
    const maxWidth = Math.max(1, window.innerWidth - horizontalMargin);
    const maxHeight = Math.max(1, window.innerHeight - verticalMargin - captionHeight);
    const scale = Math.min(1, maxWidth / width, maxHeight / height);
    dialog.style.width = `${Math.max(1, Math.floor(width * scale))}px`;
    dialog.style.setProperty("--media-height", `${Math.max(1, Math.floor(height * scale))}px`);
  }

  function closeLightbox() {
    if (!dialog || !dialogVideo) return;
    dialog.classList.remove("is-playing");
    dialogVideo.pause();
    if (dialogVideo._hls) dialogVideo._hls.destroy();
    dialogVideo._hls = null;
    dialogVideo.removeAttribute("src");
    dialogVideo.dataset.attached = "false";
    dialogVideo.dataset.shouldPlay = "false";
    dialogVideo.load();
    if (dialog.open) dialog.close();
    demoPlaybackSuspended = false;
    if (syncDemoPlayback) syncDemoPlayback();
  }

  function openLightbox(media) {
    if (!dialog || !dialogVideo) return;
    const hasMedia = Boolean(media.hls || media.mp4);
    const caption = media.prompt || "";
    dialogCaption.textContent = caption;
    dialogCaption.hidden = !caption;
    dialogVideo.hidden = !hasMedia;
    if (dialogPlaceholder) {
      dialogPlaceholder.hidden = false;
      dialogPlaceholder.textContent = hasMedia ? "LOADING VIDEO" : "VIDEO SOURCE COMING SOON";
      dialogPlaceholder.style.backgroundImage = media.poster
        ? `linear-gradient(rgba(0,0,0,.14), rgba(0,0,0,.4)), url("${media.poster}")`
        : "";
    }
    fitLightbox(media.width || 1280, media.height || 720);
    demoPlaybackSuspended = true;
    if (syncDemoPlayback) syncDemoPlayback();
    dialog.showModal();
    if (hasMedia) attachStream(dialogVideo, media, dialog, { preferMp4: true });
  }

  if (dialog && dialogVideo) {
    dialog.querySelector("[data-lightbox-close]").addEventListener("click", closeLightbox);
    dialog.addEventListener("click", event => {
      if (event.target === dialog) closeLightbox();
    });
    dialog.addEventListener("cancel", event => {
      event.preventDefault();
      closeLightbox();
    });
    dialogVideo.addEventListener("loadedmetadata", () => {
      if (dialogVideo.videoWidth && dialogVideo.videoHeight) {
        fitLightbox(dialogVideo.videoWidth, dialogVideo.videoHeight);
      }
    });
    window.addEventListener("resize", () => {
      if (dialog.open) fitLightbox();
    }, { passive: true });
  }

  function wireMediaInteraction(entry) {
    const open = () => openLightbox(entry.media);
    entry.card.addEventListener("click", open);
    entry.card.addEventListener("keydown", event => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      open();
    });
  }

  function createIntroCard(media, index) {
    const card = document.createElement("article");
    card.className = `intro-card${media.hls || media.mp4 ? "" : " is-poster-only"}`;
    card.setAttribute("aria-hidden", "true");
    card.innerHTML = `
      <img src="${media.poster}" alt="" ${index < 2 ? "fetchpriority=\"high\"" : "loading=\"lazy\""} decoding="async" />
      <video muted loop playsinline preload="none"></video>`;
    return { card, video: card.querySelector("video"), media };
  }

  function createDemoCard(media, index) {
    const card = document.createElement("article");
    card.className = `demo-card${media.hls || media.mp4 ? "" : " is-poster-only"}`;
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute("aria-label", `Open ${media.title || `video ${index + 1}`}`);
    const prompt = media.prompt ? `<p>${media.prompt}</p>` : "";
    card.innerHTML = `
      <img src="${media.poster}" alt="" loading="lazy" decoding="async" />
      <video muted loop playsinline preload="none" aria-label="${media.title || `Video sample ${index + 1}`}"></video>
      <div class="demo-meta" aria-hidden="true"><span>${media.resolution || "720p"}</span><span>${media.duration || "8 s"}</span></div>
      <div class="demo-copy"><h3>${media.title || `Sample ${index + 1}`}</h3>${prompt}</div>`;
    const entry = { card, video: card.querySelector("video"), media };
    wireMediaInteraction(entry);
    return entry;
  }

  function buildPage() {
    const introReel = document.querySelector("[data-intro-reel]");
    const demoHost = document.querySelector("[data-demo-scenes]");
    const introEntries = [];
    const demoEntries = [];

    [0, 1].forEach(repeatIndex => {
      const set = document.createElement("div");
      set.className = "intro-reel-set";
      (config.intro || []).map(normalizeMedia).forEach((media, mediaIndex) => {
        const entry = createIntroCard(media, repeatIndex * config.intro.length + mediaIndex);
        introEntries.push(entry);
        set.appendChild(entry.card);
      });
      introReel.appendChild(set);
    });

    (config.sections || []).forEach((section, sectionIndex) => {
      const element = document.createElement("section");
      element.className = "demo-section";
      element.id = section.id;
      element.style.setProperty("--grid-columns", String(section.gridColumns || 3));
      element.setAttribute("aria-labelledby", `${section.id}-title`);
      element.innerHTML = `
        <header class="demo-section-header">
          <div>
            <p class="eyebrow">${section.kicker || `0${sectionIndex + 1} · Generated results`}</p>
            <h2 id="${section.id}-title">${section.title}</h2>
          </div>
          <p>${section.description || ""}</p>
        </header>
        <div class="demo-grid" aria-label="${section.title} videos"></div>`;
      const grid = element.querySelector(".demo-grid");
      const items = section.items || (section.batches || []).flat();
      items.map(normalizeMedia).map(createDemoCard).forEach(entry => {
        demoEntries.push(entry);
        grid.appendChild(entry.card);
      });
      demoHost.appendChild(element);
    });
    return { introEntries, demoEntries };
  }

  function setupIntroPlayback(entries) {
    if (!entries.length || saveData || reduceMotion) return;
    const visible = new Set();
    const sync = () => entries.forEach(entry => {
      if (!document.hidden && visible.has(entry)) attachStream(entry.video, entry.media, entry.card);
      else pauseStream(entry.video);
    });
    const observer = new IntersectionObserver(changes => {
      changes.forEach(change => {
        const entry = change.target._mediaEntry;
        if (change.isIntersecting) visible.add(entry);
        else visible.delete(entry);
      });
      sync();
    }, { rootMargin: "100px", threshold: 0.01 });
    entries.forEach(entry => {
      entry.card._mediaEntry = entry;
      observer.observe(entry.card);
    });
    document.addEventListener("visibilitychange", sync);
  }

  function setupDemoPlayback(entries) {
    const visible = new Set();
    const sync = () => entries.forEach(entry => {
      const canPlay = !saveData && !reduceMotion && !document.hidden && !demoPlaybackSuspended && visible.has(entry);
      if (canPlay) attachStream(entry.video, entry.media, entry.card);
      else pauseStream(entry.video);
    });
    syncDemoPlayback = sync;

    if (!("IntersectionObserver" in window)) {
      entries.forEach(entry => visible.add(entry));
      sync();
      return;
    }
    const observer = new IntersectionObserver(changes => {
      changes.forEach(change => {
        const entry = change.target._mediaEntry;
        if (change.isIntersecting) visible.add(entry);
        else visible.delete(entry);
      });
      sync();
    }, { rootMargin: "120px 0px", threshold: 0.08 });
    entries.forEach(entry => {
      entry.card._mediaEntry = entry;
      observer.observe(entry.card);
    });
    document.addEventListener("visibilitychange", sync);
  }

  function setupCitation() {
    const button = document.querySelector("[data-copy-citation]");
    const citation = document.querySelector("[data-citation]");
    if (button && citation) {
      button.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(citation.textContent.trim());
          button.textContent = "Copied";
          window.setTimeout(() => { button.textContent = "Copy"; }, 1600);
        } catch (_) {
          button.textContent = "Select and copy";
        }
      });
    }
    document.querySelectorAll("[data-jump]").forEach(buttonElement => {
      buttonElement.addEventListener("click", () => {
        document.querySelector("#cinematic")?.scrollIntoView();
      });
    });
  }

  async function initialize() {
    await loadCuratedSections();
    const { introEntries, demoEntries } = buildPage();
    setupIntroPlayback(introEntries);
    setupDemoPlayback(demoEntries);
    setupCitation();
  }

  initialize();
})();
