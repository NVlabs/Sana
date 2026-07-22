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
    const totalSteps = Math.max(1, models.length);
    let displayedStep = 0;
    let thresholdStep = 0;
    let pendingRequest = null;
    let transitioning = false;
    let ignoreScrollUntil = 0;
    let storyWheelArmed = story.getBoundingClientRect().top <= 1;
    let storyArmTimer = 0;
    let wheelActive = false;
    let lastWheelTime = 0;
    let lastWheelMagnitude = 0;
    let wheelEndTimer = 0;
    let releasingStory = false;
    let lastDocumentY = window.scrollY;

    story.style.height = `${Math.max(300, totalSteps * 100)}svh`;

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

    function stepScrollTop(stepIndex) {
      const range = Math.max(1, story.offsetHeight - window.innerHeight);
      return story.offsetTop + (stepIndex / Math.max(1, totalSteps - 1)) * range;
    }

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
      const rowsHeight = model.rowCount * rowHeight + Math.max(0, model.rowCount - 1) * gap;
      model.rowDistance = rowHeight + gap;
      model.grid.style.gridTemplateColumns = `repeat(${columnCount}, minmax(0, 1fr))`;
      model.grid.style.gridAutoRows = `${rowHeight}px`;
      model.grid.style.top = `${topBuffer}px`;
      model.grid.style.bottom = "auto";
      model.grid.style.height = `${rowsHeight}px`;
      model.gridPanel.style.height = `${topBuffer + rowsHeight + bottomBuffer}px`;
    }

    function maxScroll(model) {
      return Math.max(0, model.scrollPanel.scrollHeight - model.scrollPanel.clientHeight);
    }

    function pauseOutsideViewport(activeModelIndex, firstRow) {
      models.forEach((model, modelIndex) => {
        model.entries.forEach((entry, entryIndex) => {
          const row = Math.floor(entryIndex / (model.columnCount || 3));
          const visible = modelIndex === activeModelIndex && row >= firstRow - 1 && row <= firstRow + 2;
          if (!visible) pauseStream(entry.video);
        });
      });
    }

    function updateModelState(modelIndex = displayedStep) {
      const model = models[modelIndex];
      if (!model) return;
      const distance = model.rowDistance || 1;
      const firstRow = clamp(Math.floor(model.scrollPanel.scrollTop / distance), 0, model.rowCount - 1);
      const lastRow = Math.min(model.rowCount, firstRow + 3);
      const innerProgress = maxScroll(model) ? model.scrollPanel.scrollTop / maxScroll(model) : 0;
      if (model.batchCount) {
        model.batchCount.textContent = `Rows ${String(firstRow + 1).padStart(2, "0")}–${String(lastRow).padStart(2, "0")} / ${String(model.rowCount).padStart(2, "0")}`;
      }
      storyLabel.textContent = model.label;
      storyCount.textContent = `${String(modelIndex + 1).padStart(2, "0")} / ${String(totalSteps).padStart(2, "0")}`;
      if (progressBar) {
        progressBar.style.transform = `scaleX(${(modelIndex + innerProgress) / totalSteps})`;
      }
      pauseOutsideViewport(modelIndex, firstRow);
    }

    function applyStep(stepIndex) {
      displayedStep = clamp(stepIndex, 0, totalSteps - 1);
      models.forEach((model, modelIndex) => {
        const active = modelIndex === displayedStep;
        model.element.classList.toggle("is-active", active);
        model.element.inert = !active;
        model.element.setAttribute("aria-hidden", active ? "false" : "true");
      });
      navButtons.forEach((button, modelIndex) => button.classList.toggle("is-active", modelIndex === displayedStep));
      updateModelState();
    }

    function layers(model) {
      return [model.headingPanel, model.descriptionPanel, model.scrollPanel].filter(Boolean);
    }

    async function transitionTo(stepIndex, entryEdge = "auto") {
      const nextStep = clamp(stepIndex, 0, totalSteps - 1);
      if (nextStep === displayedStep) return;
      if (transitioning) {
        pendingRequest = { step: nextStep, edge: entryEdge };
        return;
      }

      transitioning = true;
      const direction = nextStep > displayedStep ? "forward" : "backward";
      const outgoing = models[displayedStep];
      const incoming = models[nextStep];
      const enterAtBottom = entryEdge === "bottom" || (entryEdge === "auto" && direction === "backward");
      incoming.scrollPanel.scrollTop = enterAtBottom ? maxScroll(incoming) : 0;
      incoming.element.classList.add("is-transitioning");
      updateModelState(nextStep);

      if (!reduceMotion) {
        layers(outgoing).forEach(layer => layer.classList.add(`scroll-exit-${direction}`));
        layers(incoming).forEach(layer => layer.classList.add(`scroll-enter-${direction}`));
        await delay(440);
      }

      layers(outgoing).forEach(layer => layer.classList.remove("scroll-exit-forward", "scroll-exit-backward"));
      layers(incoming).forEach(layer => layer.classList.remove("scroll-enter-forward", "scroll-enter-backward"));
      applyStep(nextStep);
      incoming.element.classList.remove("is-transitioning");
      outgoing.element.classList.remove("is-transitioning");
      transitioning = false;

      const queued = pendingRequest;
      pendingRequest = null;
      if (queued && queued.step !== displayedStep) transitionTo(queued.step, queued.edge);
    }

    function requestStep(stepIndex, entryEdge = "auto") {
      const next = clamp(stepIndex, 0, totalSteps - 1);
      thresholdStep = next;
      if (transitioning) pendingRequest = { step: next, edge: entryEdge };
      else transitionTo(next, entryEdge);
    }

    function jumpToStep(stepIndex, entryEdge = "top") {
      const next = clamp(stepIndex, 0, totalSteps - 1);
      thresholdStep = next;
      storyWheelArmed = true;
      ignoreScrollUntil = performance.now() + 160;
      const root = document.documentElement;
      const previous = root.style.scrollBehavior;
      root.style.scrollBehavior = "auto";
      window.scrollTo(0, stepScrollTop(next));
      root.style.scrollBehavior = previous;
      requestStep(next, entryEdge);
    }

    function resetWheelAfterIdle() {
      window.clearTimeout(wheelEndTimer);
      wheelEndTimer = window.setTimeout(() => {
        wheelActive = false;
        lastWheelMagnitude = 0;
        releasingStory = false;
      }, 120);
    }

    function scrollDocumentBy(deltaY) {
      const root = document.documentElement;
      const previous = root.style.scrollBehavior;
      root.style.scrollBehavior = "auto";
      window.scrollBy(0, deltaY);
      root.style.scrollBehavior = previous;
    }

    function scrollDocumentTo(top) {
      const root = document.documentElement;
      const previous = root.style.scrollBehavior;
      root.style.scrollBehavior = "auto";
      window.scrollTo(0, top);
      root.style.scrollBehavior = previous;
    }

    function handleOuterBoundary(event) {
      if ((dialog && dialog.open) || Math.abs(event.deltaY) <= Math.abs(event.deltaX)) return;
      const rect = story.getBoundingClientRect();
      const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? window.innerHeight : 1;
      const deltaY = event.deltaY * unit;
      const crossingFromTitle = deltaY > 0 && rect.top > 0 && deltaY >= rect.top;
      const crossingFromCitation = deltaY < 0 && rect.bottom < window.innerHeight && -deltaY >= window.innerHeight - rect.bottom;
      if (!crossingFromTitle && !crossingFromCitation) return;

      event.preventDefault();
      event.stopPropagation();
      const boundary = crossingFromTitle
        ? story.offsetTop
        : story.offsetTop + story.offsetHeight - window.innerHeight;
      ignoreScrollUntil = performance.now() + 180;
      thresholdStep = crossingFromTitle ? 0 : totalSteps - 1;
      storyWheelArmed = false;
      wheelActive = true;
      scrollDocumentTo(boundary);
      lastDocumentY = boundary;
    }

    function handleWheel(event) {
      const rect = story.getBoundingClientRect();
      const pinned = rect.top <= 1 && rect.bottom >= window.innerHeight - 1;
      if ((dialog && dialog.open) || Math.abs(event.deltaY) <= Math.abs(event.deltaX)) return;

      if (!pinned) {
        event.preventDefault();
        storyWheelArmed = false;
        releasingStory = false;
        scrollDocumentBy(event.deltaY);
        return;
      }

      if (releasingStory) {
        event.preventDefault();
        scrollDocumentBy(event.deltaY);
        resetWheelAfterIdle();
        return;
      }

      const model = models[displayedStep];
      const direction = event.deltaY > 0 ? 1 : -1;
      const atTop = model.scrollPanel.scrollTop <= 0.5;
      const atBottom = model.scrollPanel.scrollTop >= maxScroll(model) - 0.5;
      const magnitude = Math.abs(event.deltaY);
      const now = performance.now();
      const freshImpulse = !wheelActive || now - lastWheelTime > 140;

      lastWheelTime = now;
      lastWheelMagnitude = magnitude;
      resetWheelAfterIdle();

      const leavingStory = (displayedStep === 0 && atTop && direction < 0) ||
        (displayedStep === totalSteps - 1 && atBottom && direction > 0);
      if (leavingStory) {
        if (!freshImpulse) event.preventDefault();
        else {
          event.preventDefault();
          storyWheelArmed = false;
          releasingStory = true;
          scrollDocumentBy(event.deltaY);
        }
        wheelActive = true;
        return;
      }

      if (!storyWheelArmed) {
        event.preventDefault();
        window.clearTimeout(storyArmTimer);
        storyArmTimer = window.setTimeout(() => {
          storyWheelArmed = true;
          wheelActive = false;
        }, 130);
        return;
      }

      const changingSection = (atBottom && direction > 0) || (atTop && direction < 0);
      if (changingSection) {
        event.preventDefault();
        if (freshImpulse && !transitioning) {
          wheelActive = true;
          jumpToStep(displayedStep + direction, direction < 0 ? "bottom" : "top");
        }
        return;
      }

      event.preventDefault();
      wheelActive = true;
      model.scrollPanel.scrollTop += event.deltaY;
      updateModelState();
    }

    quickNav.addEventListener("click", event => {
      const button = event.target.closest("[data-step-jump]");
      if (button) {
        const target = Number(button.dataset.stepJump);
        if (target !== displayedStep) jumpToStep(target);
        return;
      }
      if (event.target.closest("[data-citation-jump]")) {
        citation.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth" });
      }
    });

    document.querySelectorAll("[data-jump]").forEach(control => {
      control.addEventListener("click", event => {
        event.preventDefault();
        jumpToStep(Number(control.dataset.jump));
      });
    });

    function readScroll() {
      const rect = story.getBoundingClientRect();
      const range = Math.max(1, story.offsetHeight - window.innerHeight);
      const currentY = window.scrollY;
      const documentDirection = Math.sign(currentY - lastDocumentY);
      lastDocumentY = currentY;

      if (performance.now() >= ignoreScrollUntil && !storyWheelArmed) {
        const enteredFromTitle = documentDirection > 0 && rect.top < 0 && displayedStep === 0;
        const enteredFromCitation = documentDirection < 0 && rect.bottom > window.innerHeight && displayedStep === totalSteps - 1;
        if (enteredFromTitle || enteredFromCitation) {
          ignoreScrollUntil = performance.now() + 160;
          const boundary = enteredFromTitle
            ? story.offsetTop
            : story.offsetTop + story.offsetHeight - window.innerHeight;
          scrollDocumentTo(boundary);
          thresholdStep = enteredFromTitle ? 0 : totalSteps - 1;
          return;
        }
      }

      const progress = clamp(-rect.top / range);
      if (performance.now() < ignoreScrollUntil) return;
      const next = clamp(Math.round(progress * Math.max(1, totalSteps - 1)), 0, totalSteps - 1);
      if (next !== thresholdStep) requestStep(next, next < displayedStep ? "bottom" : "top");
    }

    models.forEach(model => {
      layoutGrid(model);
      model.scrollPanel.addEventListener("scroll", () => {
        if (model === models[displayedStep]) updateModelState();
      }, { passive: true });
    });
    applyStep(0);

    const initialRect = story.getBoundingClientRect();
    const initialRange = Math.max(1, story.offsetHeight - window.innerHeight);
    thresholdStep = clamp(Math.round(clamp(-initialRect.top / initialRange) * Math.max(1, totalSteps - 1)), 0, totalSteps - 1);
    if (thresholdStep) applyStep(thresholdStep);

    window.addEventListener("scroll", readScroll, { passive: true });
    window.addEventListener("wheel", handleOuterBoundary, { passive: false, capture: true });
    window.addEventListener("resize", () => {
      models.forEach(layoutGrid);
      updateModelState();
      readScroll();
    });
    story.addEventListener("wheel", handleWheel, { passive: false, capture: true });
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
