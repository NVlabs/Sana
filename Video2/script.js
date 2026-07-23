(() => {
  "use strict";

  const config = window.SANA_VIDEO_MEDIA || { intro: [], sections: [] };
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const saveData = Boolean(navigator.connection && navigator.connection.saveData);
  const coarsePointer = window.matchMedia("(pointer: coarse)").matches;
  const clamp = (value, min = 0, max = 1) => Math.max(min, Math.min(max, value));
  const smoothRange = (value, start, end) => {
    const progress = clamp((value - start) / Math.max(0.0001, end - start));
    return progress * progress * (3 - 2 * progress);
  };
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

  async function attachStream(video, media, host, options = {}) {
    if (!video) return;
    if (video.dataset.attached === "true") {
      if (video._hls) video._hls.startLoad();
      video.play().catch(() => {});
      return;
    }

    const hls = media.hls || "";
    const mp4 = media.mp4 || "";
    const preferMp4 = Boolean(options.preferMp4 && mp4);
    const hlsBufferLength = options.hlsBufferLength || 12;
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
        const player = new Hls({
          startLevel: 0,
          capLevelToPlayerSize: true,
          maxBufferLength: hlsBufferLength,
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
      if (preferMp4 && hls) {
        if (video._hls) {
          video._hls.destroy();
          video._hls = null;
        }
        video.removeAttribute("src");
        video.dataset.attached = "false";
        video.load();
        return attachStream(video, { ...media, mp4: "" }, host, { hlsBufferLength });
      }
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
    const fittedWidth = Math.max(1, Math.floor(width * scale));
    const fittedHeight = Math.max(1, Math.floor(height * scale));
    dialog.style.width = `${fittedWidth}px`;
    dialog.style.setProperty("--media-height", `${fittedHeight}px`);
  }

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
    dialog.showModal();
    if (hasMedia) attachStream(dialogVideo, media, dialog, { preferMp4: true });
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
    const preview = () => {
      if (!saveData && !reduceMotion) attachStream(entry.video, entry.media, entry.card);
    };
    const open = () => {
      pauseStream(entry.video);
      openLightbox(entry.media);
    };

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

  function setupIntroPlayback(entries) {
    if (!entries.length || saveData || reduceMotion) return;
    const visibleEntries = new Set();
    const play = entry => attachStream(entry.video, entry.media, entry.card, { hlsBufferLength: 4 });
    const pause = entry => pauseStream(entry.video);

    if (!("IntersectionObserver" in window)) {
      entries.slice(0, 6).forEach(play);
      return;
    }

    const observer = new IntersectionObserver(changes => {
      changes.forEach(change => {
        const entry = change.target._introEntry;
        if (!entry) return;
        if (change.isIntersecting) {
          visibleEntries.add(entry);
          if (!document.hidden) play(entry);
        } else {
          visibleEntries.delete(entry);
          pause(entry);
        }
      });
    }, { rootMargin: "80px", threshold: 0.01 });

    entries.forEach(entry => {
      entry.card._introEntry = entry;
      observer.observe(entry.card);
    });

    document.addEventListener("visibilitychange", () => {
      if (document.hidden) entries.forEach(pause);
      else visibleEntries.forEach(play);
    });
  }

  function createDemoCard(media, index) {
    const card = document.createElement("article");
    const hasMedia = Boolean(media.hls || media.mp4);
    card.className = `demo-card${hasMedia ? "" : " is-poster-only"}`;
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

  function buildScenes() {
    const introReel = document.querySelector("[data-intro-reel]");
    const demoHost = document.querySelector("[data-demo-scenes]");
    const models = [];
    const introEntries = [];

    [0, 1].forEach(repeatIndex => {
      const set = document.createElement("div");
      set.className = "intro-reel-set";
      (config.intro || []).forEach((media, mediaIndex) => {
        const index = repeatIndex * (config.intro || []).length + mediaIndex;
        const entry = createIntroCard(media, index);
        introEntries.push(entry);
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

    return { models, introEntries };
  }

  function setupHeroTimeline() {
    const hero = document.querySelector("[data-hero-timeline]");
    const headline = hero && hero.querySelector("[data-hero-headline]");
    const paper = hero && hero.querySelector("[data-hero-paper]");
    const paperTitle = hero && hero.querySelector("[data-hero-paper-title]");
    const byline = hero && hero.querySelector("[data-hero-byline]");
    const abstract = hero && hero.querySelector("[data-hero-abstract]");
    const results = hero ? Array.from(hero.querySelectorAll("[data-hero-results]")) : [];
    if (!hero || !headline || !paper || !paperTitle || !byline || !abstract) return;

    let travel = 1;
    let paperTitleScaleGain = 0.16;
    let renderFrame = 0;
    let resizeFrame = 0;

    function layout() {
      const compact = window.matchMedia("(max-width: 600px)").matches;
      travel = Math.max(760, window.innerHeight * (compact ? 1.65 : 1.42));
      paperTitleScaleGain = compact ? 0.08 : 0.16;
      hero.style.height = `${Math.ceil(window.innerHeight + travel)}px`;
    }

    function render() {
      renderFrame = 0;
      const progress = clamp((window.scrollY - hero.offsetTop) / travel);
      const headlineExit = smoothRange(progress, 0.04, 0.34);
      const bylineExit = smoothRange(progress, 0.18, 0.46);
      const paperSettle = smoothRange(progress, 0.16, 0.68);
      const paperTitleEmphasis = smoothRange(progress, 0.3, 0.78);
      const abstractReveal = smoothRange(progress, 0.52, 0.84);
      const resultsExit = smoothRange(progress, 0.41, 0.53);
      const motion = reduceMotion ? 0 : 1;
      const headlineShift = Math.min(180, window.innerHeight * 0.2) * headlineExit * motion;
      const paperShift = Math.min(155, window.innerHeight * 0.18) * paperSettle * motion;
      const resultsShift = Math.min(180, window.innerHeight * 0.2) * resultsExit * motion;

      headline.style.transform = `translate3d(0, ${-headlineShift}px, 0)`;
      headline.style.opacity = String(1 - headlineExit);
      paper.style.transform = `translate3d(0, ${-paperShift}px, 0)`;
      paperTitle.style.transform = `scale(${1 + paperTitleScaleGain * paperTitleEmphasis * motion})`;
      byline.style.transform = `translate3d(0, ${-26 * bylineExit * motion}px, 0)`;
      byline.style.opacity = String(1 - bylineExit);
      abstract.style.transform = `translate3d(0, ${(1 - abstractReveal) * 44 * motion}px, 0)`;
      abstract.style.opacity = String(abstractReveal);
      results.forEach(result => {
        result.style.transform = `translate3d(0, ${-resultsShift}px, 0)`;
        result.style.opacity = String(1 - resultsExit);
      });
    }

    function requestRender() {
      if (!renderFrame) renderFrame = window.requestAnimationFrame(render);
    }

    layout();
    render();
    window.addEventListener("scroll", requestRender, { passive: true });
    window.addEventListener("resize", () => {
      if (resizeFrame) window.cancelAnimationFrame(resizeFrame);
      resizeFrame = window.requestAnimationFrame(() => {
        resizeFrame = 0;
        layout();
        render();
      });
    }, { passive: true });
  }

  function setupHeroCarousels() {
    const hero = document.querySelector("[data-hero-timeline]");
    const metricSlides = hero ? Array.from(hero.querySelectorAll("[data-metric-slide]")) : [];
    const metricDots = hero ? Array.from(hero.querySelectorAll("[data-metric-dot]")) : [];
    const chartSlides = hero ? Array.from(hero.querySelectorAll("[data-chart-slide]")) : [];
    const chartDots = hero ? Array.from(hero.querySelectorAll("[data-chart-dot]")) : [];
    if (!hero || !metricSlides.length || !chartSlides.length) return;

    let metricIndex = 0;
    let chartIndex = 0;
    let metricTimer = 0;
    let chartTimer = 0;
    let visible = false;
    const switchDuration = reduceMotion ? 0 : 480;

    function replayChart(index) {
      const chart = chartSlides[index];
      chart.classList.remove("is-drawing");
      void chart.offsetWidth;
      chart.classList.add("is-drawing");
    }

    function setDots(dots, activeIndex) {
      dots.forEach((dot, index) => {
        const active = index === activeIndex;
        dot.classList.toggle("is-active", active);
        dot.setAttribute("aria-pressed", active ? "true" : "false");
      });
    }

    function activate(slides, dots, currentIndex, nextIndex, drawChart = false) {
      const next = clamp(nextIndex, 0, slides.length - 1);
      if (next === currentIndex) {
        if (drawChart) replayChart(next);
        return currentIndex;
      }

      const outgoing = slides[currentIndex];
      const incoming = slides[next];
      slides.forEach(slide => slide.classList.remove("is-entering", "is-leaving"));
      outgoing.classList.remove("is-active", "is-drawing");
      incoming.classList.add("is-active");
      outgoing.setAttribute("aria-hidden", "true");
      incoming.setAttribute("aria-hidden", "false");

      if (switchDuration) {
        void incoming.offsetWidth;
        outgoing.classList.add("is-leaving");
        incoming.classList.add("is-entering");
      }
      setDots(dots, next);
      if (drawChart) replayChart(next);
      return next;
    }

    function restartMetricTimer() {
      window.clearInterval(metricTimer);
      metricTimer = 0;
      if (!visible || document.hidden || reduceMotion) return;
      metricTimer = window.setInterval(() => {
        metricIndex = activate(metricSlides, metricDots, metricIndex, (metricIndex + 1) % metricSlides.length);
      }, 2600);
    }

    function restartChartTimer() {
      window.clearInterval(chartTimer);
      chartTimer = 0;
      if (!visible || document.hidden || reduceMotion) return;
      chartTimer = window.setInterval(() => {
        chartIndex = activate(chartSlides, chartDots, chartIndex, (chartIndex + 1) % chartSlides.length, true);
      }, 6800);
    }

    metricSlides.forEach((slide, index) => slide.setAttribute("aria-hidden", index ? "true" : "false"));
    chartSlides.forEach((slide, index) => slide.setAttribute("aria-hidden", index ? "true" : "false"));
    setDots(metricDots, metricIndex);
    setDots(chartDots, chartIndex);
    replayChart(chartIndex);

    metricDots.forEach((dot, index) => dot.addEventListener("click", () => {
      metricIndex = activate(metricSlides, metricDots, metricIndex, index);
      restartMetricTimer();
    }));
    chartDots.forEach((dot, index) => dot.addEventListener("click", () => {
      chartIndex = activate(chartSlides, chartDots, chartIndex, index, true);
      restartChartTimer();
    }));

    const observer = new IntersectionObserver(entries => {
      visible = Boolean(entries[0] && entries[0].isIntersecting);
      restartMetricTimer();
      restartChartTimer();
    }, { threshold: 0.01 });
    observer.observe(hero);
    document.addEventListener("visibilitychange", () => {
      restartMetricTimer();
      restartChartTimer();
    });
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
          moveEnd: cursor + edgeHold + movementSpan,
          end: cursor + edgeHold + movementSpan + edgeHold,
          movementSpan,
          edgeHold
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

    function updateGrid(modelIndex, requestedOffset) {
      const model = models[modelIndex];
      const gridOffset = clamp(
        requestedOffset,
        -model.rowDistance,
        model.gridTravel + model.rowDistance
      );
      model.grid.style.transform = `translate3d(0, ${-gridOffset}px, 0)`;
      const firstRow = clamp(Math.floor((gridOffset + 1) / Math.max(1, model.rowDistance)), 0, model.rowCount - 1);
      const lastRow = Math.min(model.rowCount, firstRow + 2);
      if (model.batchCount) {
        model.batchCount.textContent = `Rows ${String(firstRow + 1).padStart(2, "0")}–${String(lastRow).padStart(2, "0")} / ${String(model.rowCount).padStart(2, "0")}`;
      }
      return firstRow;
    }

    function gridOffsetForTravel(model, frame, travel) {
      if (travel < frame.moveStart) {
        const edgeProgress = clamp((travel - frame.start) / Math.max(1, frame.edgeHold));
        return -model.rowDistance * (1 - edgeProgress);
      }
      if (travel <= frame.moveEnd) {
        const movementProgress = clamp((travel - frame.moveStart) / Math.max(1, frame.movementSpan));
        return model.gridTravel * movementProgress;
      }
      const edgeProgress = clamp((travel - frame.moveEnd) / Math.max(1, frame.edgeHold));
      return model.gridTravel + model.rowDistance * edgeProgress;
    }

    function render() {
      renderFrame = 0;
      const travel = clamp(window.scrollY - story.offsetTop, 0, totalTravel);
      let sectionIndex = timeline.findIndex(frame => travel < frame.end - 0.5);
      if (sectionIndex < 0) sectionIndex = models.length - 1;
      const frame = timeline[sectionIndex];
      const gridOffset = gridOffsetForTravel(models[sectionIndex], frame, travel);
      const firstRow = updateGrid(sectionIndex, gridOffset);

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
        top: story.offsetTop + timeline[next].moveStart,
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

  const { models, introEntries } = buildScenes();
  setupIntroPlayback(introEntries);
  setupHeroTimeline();
  setupHeroCarousels();
  setupStory(models);
  setupCitation();
})();
