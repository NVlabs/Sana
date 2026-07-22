(() => {
  "use strict";

  const config = window.SANA_VIDEO_MEDIA || { intro: [], sections: [] };
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const saveData = Boolean(navigator.connection && navigator.connection.saveData);
  const coarsePointer = window.matchMedia("(pointer: coarse)").matches;
  const clamp = (value, min = 0, max = 1) => Math.max(min, Math.min(max, value));
  const delay = duration => new Promise(resolve => window.setTimeout(resolve, duration));
  let hlsLoader;

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

  async function attachStream(video, media, host) {
    if (!video) return;
    if (video.dataset.attached === "true") {
      if (video._hls) video._hls.startLoad();
      video.play().catch(() => {});
      return;
    }

    const hls = media.hls || "";
    const mp4 = media.mp4 || "";
    if (!hls && !mp4) return;
    video.dataset.attached = "true";

    try {
      if (hls && video.canPlayType("application/vnd.apple.mpegurl")) {
        video.src = hls;
      } else if (hls) {
        const Hls = await loadHlsLibrary();
        if (!Hls || !Hls.isSupported()) throw new Error("HLS unsupported");
        const player = new Hls({
          startLevel: 0,
          capLevelToPlayerSize: true,
          maxBufferLength: 12,
          backBufferLength: 0
        });
        player.loadSource(hls);
        player.attachMedia(video);
        video._hls = player;
      } else {
        video.src = mp4;
      }

      video.addEventListener("playing", () => host && host.classList.add("is-playing"), { once: true });
      await video.play();
    } catch (_) {
      if (mp4 && !video.src.endsWith(mp4)) {
        video.src = mp4;
        video.play().catch(() => {});
      } else {
        video.dataset.attached = "false";
      }
    }
  }

  function pauseStream(video) {
    if (!video || video.dataset.attached !== "true") return;
    video.pause();
    if (video._hls) video._hls.stopLoad();
  }

  const dialog = document.querySelector("[data-lightbox]");
  const dialogVideo = dialog && dialog.querySelector("[data-lightbox-video]");
  const dialogCaption = dialog && dialog.querySelector("[data-lightbox-caption]");
  const dialogPlaceholder = dialog && dialog.querySelector("[data-lightbox-placeholder]");

  function closeLightbox() {
    if (!dialog || !dialogVideo) return;
    dialog.classList.remove("is-playing");
    dialogVideo.pause();
    if (dialogVideo._hls) {
      dialogVideo._hls.destroy();
      dialogVideo._hls = null;
    }
    dialogVideo.removeAttribute("src");
    dialogVideo.dataset.attached = "false";
    dialogVideo.load();
    if (dialog.open) dialog.close();
  }

  function openLightbox(media) {
    if (!dialog || !dialogVideo) return;
    const hasMedia = Boolean(media.hls || media.mp4);
    dialogCaption.textContent = media.prompt || media.title || "";
    dialogVideo.hidden = !hasMedia;
    if (dialogPlaceholder) {
      dialogPlaceholder.hidden = false;
      dialogPlaceholder.textContent = hasMedia ? "LOADING VIDEO" : "VIDEO SOURCE COMING SOON";
      dialogPlaceholder.style.backgroundImage = media.poster
        ? `linear-gradient(rgba(0,0,0,.14), rgba(0,0,0,.4)), url("${media.poster}")`
        : "";
    }
    dialog.showModal();
    if (hasMedia) attachStream(dialogVideo, media, dialog);
  }

  if (dialog) {
    dialog.querySelector("[data-lightbox-close]").addEventListener("click", closeLightbox);
    dialog.addEventListener("click", event => {
      if (event.target === dialog) closeLightbox();
    });
    dialog.addEventListener("cancel", event => {
      event.preventDefault();
      closeLightbox();
    });
  }

  function wireMediaInteraction(entry) {
    const preview = () => {
      if (!saveData && !reduceMotion) attachStream(entry.video, entry.media, entry.card);
    };
    const open = () => openLightbox(entry.media);

    if (!coarsePointer) {
      entry.card.addEventListener("pointerenter", preview);
      entry.card.addEventListener("pointerleave", () => pauseStream(entry.video));
    }
    entry.card.addEventListener("focus", preview);
    entry.card.addEventListener("blur", () => pauseStream(entry.video));
    entry.card.addEventListener("click", open);
    entry.card.addEventListener("keydown", event => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      open();
    });
  }

  function createIntroCard(media, index) {
    const card = document.createElement("article");
    const hasMedia = Boolean(media.hls || media.mp4);
    card.className = `intro-card${hasMedia ? "" : " is-poster-only"}`;
    card.setAttribute("aria-hidden", "true");
    card.innerHTML = `
      <img src="${media.poster}" alt="" ${index < 2 ? "fetchpriority=\"high\"" : "loading=\"lazy\""} decoding="async" />
      <video muted loop playsinline preload="none"></video>`;
    return { card, video: card.querySelector("video"), media };
  }

  function createDemoCard(media, index) {
    const card = document.createElement("article");
    const hasMedia = Boolean(media.hls || media.mp4);
    card.className = `demo-card${hasMedia ? "" : " is-poster-only"}`;
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute("aria-label", `Open ${media.title || `video ${index + 1}`}`);
    card.innerHTML = `
      <img src="${media.poster}" alt="" loading="lazy" decoding="async" />
      <video muted loop playsinline preload="none" aria-label="${media.title || `Video sample ${index + 1}`}"></video>
      <div class="demo-meta" aria-hidden="true"><span>${media.resolution || "720p"}</span><span>${media.duration || "8 s"}</span></div>
      <div class="demo-copy"><h3>${media.title || `Sample ${index + 1}`}</h3><p>${media.prompt || ""}</p></div>`;
    const entry = { card, video: card.querySelector("video"), media };
    wireMediaInteraction(entry);
    return entry;
  }

  function buildScenes() {
    const introReel = document.querySelector("[data-intro-reel]");
    const demoHost = document.querySelector("[data-demo-scenes]");
    const models = [];

    [0, 1].forEach(repeatIndex => {
      const set = document.createElement("div");
      set.className = "intro-reel-set";
      (config.intro || []).forEach((media, mediaIndex) => {
        const index = repeatIndex * (config.intro || []).length + mediaIndex;
        const entry = createIntroCard(media, index);
        set.appendChild(entry.card);
      });
      introReel.appendChild(set);
    });

    (config.sections || []).forEach((section, sectionIndex) => {
      const sceneIndex = sectionIndex;
      const scene = document.createElement("section");
      scene.className = "scene demo-scene";
      scene.dataset.scene = String(sceneIndex);
      scene.id = section.id;
      scene.setAttribute("aria-label", section.title);
      scene.innerHTML = `
        <header class="scene-header">
          <div class="scene-heading-panel">
            <p class="eyebrow">${section.kicker || `0${sceneIndex + 1} · Generated results`}</p>
            <h2>${section.title}</h2>
            <span class="batch-count" data-batch-count></span>
          </div>
          <p class="scene-description-panel">${section.description || ""}</p>
        </header>
        <div class="batch-viewport" data-batch-host></div>`;
      demoHost.appendChild(scene);

      const viewport = scene.querySelector("[data-batch-host]");
      const items = section.items || (section.batches || []).flat();
      const scrollPanel = document.createElement("div");
      scrollPanel.className = "grid-scroll-panel";
      const gridPanel = document.createElement("div");
      gridPanel.className = "grid-section-panel";
      const grid = document.createElement("div");
      grid.className = "demo-grid-track";
      grid.setAttribute("aria-label", `${section.title} videos`);
      const entries = items.map(createDemoCard);
      entries.forEach(entry => grid.appendChild(entry.card));
      gridPanel.appendChild(grid);
      scrollPanel.appendChild(gridPanel);
      viewport.appendChild(scrollPanel);
      const rowCount = Math.max(1, Math.ceil(entries.length / 3));

      models.push({
        id: section.id,
        label: section.title,
        element: scene,
        viewport,
        scrollPanel,
        gridPanel,
        grid,
        entries,
        rowCount,
        batchCount: scene.querySelector("[data-batch-count]"),
        headingPanel: scene.querySelector(".scene-heading-panel"),
        descriptionPanel: scene.querySelector(".scene-description-panel"),
      });
    });

    return models;
  }

  function setupStory(models) {
    const story = document.querySelector("[data-story]");
    const quickNav = document.querySelector("[data-quick-nav]");
    const progressBar = document.querySelector("[data-story-progress]");
    const storyLabel = document.querySelector("[data-story-label]");
    const storyCount = document.querySelector("[data-story-count]");
    const citation = document.querySelector("#citation");
    const totalSections = Math.max(1, models.length);
    let timeline = [];
    let totalTravel = 1;
    let renderFrame = 0;
    let resizeFrame = 0;
    let activeModelIndex = -1;
    let lastMediaWindow = "";
    let transitionRunning = false;
    let pendingModelIndex = null;
    const transitionDuration = reduceMotion ? 0 : 420;

    models.forEach((model, modelIndex) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = model.label;
      button.dataset.stepJump = String(modelIndex);
      button.setAttribute("aria-label", `Jump to ${model.label}`);
      quickNav.appendChild(button);
    });

    const citationButton = document.createElement("button");
    citationButton.type = "button";
    citationButton.textContent = "BibTeX";
    citationButton.dataset.citationJump = "true";
    quickNav.appendChild(citationButton);
    const navButtons = Array.from(quickNav.querySelectorAll("[data-step-jump]"));

    function layoutGrid(model) {
      const gap = Number.parseFloat(window.getComputedStyle(model.grid).rowGap) || 0;
      const viewportHeight = model.scrollPanel.clientHeight;
      const compact = window.matchMedia("(max-width: 600px)").matches;
      const columnCount = compact ? 2 : 3;
      const topBuffer = window.innerHeight * (compact ? 0.29 : 0.31);
      const baseBottomBuffer = compact ? 84 : 68;
      const focusHeight = Math.max(2, viewportHeight - topBuffer - baseBottomBuffer);
      const cardWidth = Math.max(1, (model.viewport.clientWidth - gap * (columnCount - 1)) / columnCount);
      const rowHeight = Math.max(1, Math.min((focusHeight - gap) / 2, cardWidth * 9 / 16));
      const bottomBuffer = Math.max(baseBottomBuffer, viewportHeight - topBuffer - (rowHeight * 2 + gap));
      model.columnCount = columnCount;
      model.rowCount = Math.max(1, Math.ceil(model.entries.length / columnCount));
      model.rowDistance = rowHeight + gap;
      model.gridTravel = Math.max(0, (model.rowCount - 2) * model.rowDistance);
      const rowsHeight = model.rowCount * rowHeight + Math.max(0, model.rowCount - 1) * gap;
      model.grid.style.gridTemplateColumns = `repeat(${columnCount}, minmax(0, 1fr))`;
      model.grid.style.gridAutoRows = `${rowHeight}px`;
      model.grid.style.top = `${topBuffer}px`;
      model.grid.style.bottom = "auto";
      model.grid.style.height = `${rowsHeight}px`;
      model.gridPanel.style.height = `${topBuffer + rowsHeight + bottomBuffer}px`;
    }

    function buildTimeline() {
      const viewportHeight = window.innerHeight;
      const compact = window.matchMedia("(max-width: 600px)").matches;
      let cursor = 0;
      timeline = models.map(model => {
        layoutGrid(model);
        const edgeHold = Math.max(180, viewportHeight * (compact ? 0.28 : 0.36));
        const movementSpan = Math.max(
          viewportHeight * (compact ? 0.72 : 0.78),
          model.gridTravel * 1.65
        );
        const frame = {
          start: cursor,
          moveStart: cursor + edgeHold,
          end: cursor + edgeHold + movementSpan + edgeHold,
          movementSpan
        };
        cursor = frame.end;
        return frame;
      });
      totalTravel = Math.max(1, cursor);
      story.style.height = `${Math.ceil(totalTravel + viewportHeight)}px`;
    }

    function pauseOutsideViewport(modelIndex, firstRow) {
      const mediaWindow = `${modelIndex}:${firstRow}`;
      if (mediaWindow === lastMediaWindow) return;
      lastMediaWindow = mediaWindow;
      models.forEach((model, currentModelIndex) => {
        model.entries.forEach((entry, entryIndex) => {
          const row = Math.floor(entryIndex / (model.columnCount || 3));
          const visible = currentModelIndex === modelIndex && row >= firstRow - 1 && row <= firstRow + 2;
          if (!visible) pauseStream(entry.video);
        });
      });
    }

    function setActiveModel(modelIndex) {
      if (modelIndex === activeModelIndex) return;
      activeModelIndex = modelIndex;
      models.forEach((model, index) => {
        const active = index === modelIndex;
        model.element.classList.toggle("is-active", active);
        model.element.inert = !active;
        model.element.setAttribute("aria-hidden", active ? "false" : "true");
      });
      navButtons.forEach((button, index) => button.classList.toggle("is-active", index === modelIndex));
      storyLabel.textContent = models[modelIndex].label;
      storyCount.textContent = `${String(modelIndex + 1).padStart(2, "0")} / ${String(totalSections).padStart(2, "0")}`;
    }

    function sceneLayers(model) {
      return [model.headingPanel, model.descriptionPanel, model.scrollPanel].filter(Boolean);
    }

    function clearTransitionClasses(model) {
      sceneLayers(model).forEach(layer => {
        layer.classList.remove(
          "section-exit-forward",
          "section-exit-backward",
          "section-enter-forward",
          "section-enter-backward"
        );
      });
    }

    function showOnly(modelIndex) {
      models.forEach((model, index) => {
        clearTransitionClasses(model);
        model.element.classList.toggle("is-visible", index === modelIndex);
      });
      setActiveModel(modelIndex);
    }

    async function transitionTo(modelIndex) {
      const nextIndex = clamp(modelIndex, 0, models.length - 1);
      if (nextIndex === activeModelIndex && !transitionRunning) return;
      if (transitionRunning) {
        pendingModelIndex = nextIndex;
        return;
      }

      const previousIndex = activeModelIndex;
      if (previousIndex < 0 || transitionDuration === 0) {
        showOnly(nextIndex);
        return;
      }

      transitionRunning = true;
      pendingModelIndex = null;
      const direction = nextIndex > previousIndex ? "forward" : "backward";
      const outgoing = models[previousIndex];
      const incoming = models[nextIndex];
      outgoing.element.classList.add("is-visible");
      incoming.element.classList.add("is-visible");
      clearTransitionClasses(outgoing);
      clearTransitionClasses(incoming);
      void incoming.element.offsetHeight;
      sceneLayers(outgoing).forEach(layer => layer.classList.add(`section-exit-${direction}`));
      sceneLayers(incoming).forEach(layer => layer.classList.add(`section-enter-${direction}`));

      await delay(transitionDuration);
      showOnly(nextIndex);
      transitionRunning = false;

      const queuedIndex = pendingModelIndex;
      pendingModelIndex = null;
      if (queuedIndex !== null && queuedIndex !== activeModelIndex) transitionTo(queuedIndex);
    }

    function updateGrid(modelIndex, progress) {
      const model = models[modelIndex];
      const gridOffset = model.gridTravel * clamp(progress);
      model.grid.style.transform = `translate3d(0, ${-gridOffset}px, 0)`;
      const firstRow = clamp(Math.floor((gridOffset + 1) / Math.max(1, model.rowDistance)), 0, model.rowCount - 1);
      const lastRow = Math.min(model.rowCount, firstRow + 2);
      if (model.batchCount) {
        model.batchCount.textContent = `Rows ${String(firstRow + 1).padStart(2, "0")}–${String(lastRow).padStart(2, "0")} / ${String(model.rowCount).padStart(2, "0")}`;
      }
      return firstRow;
    }

    function render() {
      renderFrame = 0;
      const travel = clamp(window.scrollY - story.offsetTop, 0, totalTravel);
      let sectionIndex = timeline.findIndex(frame => travel < frame.end - 0.5);
      if (sectionIndex < 0) sectionIndex = models.length - 1;
      const frame = timeline[sectionIndex];
      const contentProgress = clamp((travel - frame.moveStart) / Math.max(1, frame.movementSpan));
      const firstRow = updateGrid(sectionIndex, contentProgress);

      if (activeModelIndex < 0) showOnly(sectionIndex);
      else if (transitionRunning) pendingModelIndex = sectionIndex;
      else if (sectionIndex !== activeModelIndex) transitionTo(sectionIndex);
      pauseOutsideViewport(sectionIndex, firstRow);

      if (progressBar) progressBar.style.transform = `scaleX(${travel / totalTravel})`;
    }

    function requestRender() {
      if (!renderFrame) renderFrame = window.requestAnimationFrame(render);
    }

    function scrollToSection(sectionIndex) {
      const next = clamp(sectionIndex, 0, models.length - 1);
      window.scrollTo({
        top: story.offsetTop + timeline[next].start,
        behavior: reduceMotion ? "auto" : "smooth"
      });
    }

    quickNav.addEventListener("click", event => {
      const sectionButton = event.target.closest("[data-step-jump]");
      if (sectionButton) {
        scrollToSection(Number(sectionButton.dataset.stepJump));
        return;
      }
      if (event.target.closest("[data-citation-jump]")) {
        citation.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth" });
      }
    });

    document.querySelectorAll("[data-jump]").forEach(control => {
      control.addEventListener("click", event => {
        event.preventDefault();
        scrollToSection(Number(control.dataset.jump));
      });
    });

    buildTimeline();
    render();
    window.addEventListener("scroll", requestRender, { passive: true });
    window.addEventListener("resize", () => {
      if (resizeFrame) window.cancelAnimationFrame(resizeFrame);
      resizeFrame = window.requestAnimationFrame(() => {
        resizeFrame = 0;
        buildTimeline();
        render();
      });
    }, { passive: true });
  }

  function setupCitation() {
    const button = document.querySelector("[data-copy-citation]");
    const citation = document.querySelector("[data-citation]");
    if (!button || !citation) return;
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

  const models = buildScenes();
  setupStory(models);
  setupCitation();
})();
