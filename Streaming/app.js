const root = document.getElementById("app");

const base = "https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets/resolve/main/Streaming/assets/media/";

const authors = [
  { name: "Yuyang Zhao", href: "https://yuyangzhao.com/", equal: true },
  { name: "Yicheng Pan", href: "https://ryanpyc27.github.io/", equal: true },
  { name: "Qiyuan He", href: "https://qy-h00.github.io/", equal: true },
  { name: "Jincheng Yu", href: "https://github.com/yujincheng08", equal: true },
  { name: "Junsong Chen", href: "https://lawrence-cj.github.io/", equal: true, breakAfter: true },
  { name: "Tian Ye", href: "https://owen718.github.io/" },
  { name: "Haozhe Liu", href: "https://haozheliu-st.github.io/" },
  { name: "Enze Xie", href: "https://xieenze.github.io/" },
  { name: "Song Han", href: "https://hanlab.mit.edu/songhan/" }
];

const features = [
  {
    eyebrow: "Streaming",
    title: "Minute-Length Editing",
    body: "Causal video-to-video editing over one-minute streams while preserving source motion and non-edited content."
  },
  {
    eyebrow: "Real Time",
    title: "24 FPS End-to-End",
    body: "Runs at real-time speed on a single RTX 5090, with the DiT core reaching 58 FPS after system co-design."
  },
  {
    eyebrow: "Resolution",
    title: "1280 x 704 Output",
    body: "High-resolution editing with a causal VAE decoder for streaming generation and decoding."
  },
  {
    eyebrow: "Architecture",
    title: "Hybrid Diffusion Transformer",
    body: "Interleaves GDN blocks for compact global memory with softmax-attention blocks for local source alignment."
  }
];

const minuteLengthGroups = [
  {
    title: "Local Editing",
    items: [
      {
        src: `ready-to-use/long-select/000026. Remove the thick, textured gold hoop earrings from8b7ce0e68f.mp4`,
        prompt: `Remove the thick, textured gold hoop earrings from the woman's ears. Carefully reconstruct the exposed earlobes to match her natural skin tone and texture. Ensure the lighting and soft shadows on the newly bare ears blend seamlessly with the rest of her face, leaving no trace or reflection of the metallic jewelry behind.`
      },
      {
        src: `ready-to-use/long-select/005356. Replace the subject's white button-up shirt with a24dfe94d60.mp4`,
        prompt: `Replace the subject's white button-up shirt with a luxurious, dark navy blue silk blouse. The new garment should feature a soft, elegantly draped ruffled collar and a line of small, iridescent pearl buttons. Ensure the smooth silk material has a subtle sheen that realistically catches and reflects the warm, golden light coming from the background lamp throughout the sequence.`
      },
      {
        src: `ready-to-use/long-select/005571. Replace the subject's white and red track jacket w4665d224b6.mp4`,
        prompt: `Replace the subject's white and red track jacket with a vintage dark brown leather aviator jacket. The new garment should feature a thick, cream-colored shearling collar around the neck, rugged and slightly distressed leather texturing across the shoulders, and a heavy antique brass zipper detail along the front edge, all interacting naturally with the warm, golden ambient lighting.`
      },
      {
        src: `ready-to-use/long-select/005632. Give the subject a pair of delicate, round gold-wi7b739cce30.mp4`,
        prompt: `Give the subject a pair of delicate, round gold-wire eyeglasses. Ensure the thin metallic frames rest naturally on the bridge of his nose, with the clear lenses catching soft, realistic reflections from the ambient cinematic lighting, perfectly complementing his retro, preppy aesthetic without obscuring his tearful expression.`
      }
    ]
  },
  {
    title: "Background Editing",
    items: [
      {
        src: `ready-to-use/long-select/000049. Replace the solid black background with a sleek, c7285bb63b2.mp4`,
        prompt: `Replace the solid black studio background with a clean, minimalist white-and-gray showroom interior. Add smooth light-gray paneled walls and a large rectangular overhead softbox that casts bright, diffused light across the space. Preserve the subject's pose, hand motion, white textured top, jewelry, skin tone, and shallow-depth-of-field cinematography while making the new geometric interior feel naturally integrated behind her.`
      },
      {
        src: `ready-to-use/long-select/005810. Replace the background with a cinematic, rain-stre2eccb1022c.mp4`,
        prompt: `Replace the background with a cinematic, rain-streaked windowpane at dusk. Feature softly out-of-focus city lights in moody cool teal and muted amber glowing through the wet glass. Add delicate condensation and trickling raindrops to the window surface, maintaining a shallow depth of field to enhance the deeply emotional, melancholic atmosphere without altering the subject's lighting or appearance.`
      },
      {
        src: `ready-to-use/long-select/006386. Replace the background with a quiet, softly blurre40fabb12db.mp4`,
        prompt: `Replace the background with a quiet, softly blurred European city street under an overcast sky. Include out-of-focus historic stone buildings in muted earth tones, a distant wrought-iron street lamp, and wet cobblestones subtly reflecting the ambient light. Ensure the new environment perfectly matches the soft, evenly diffused daylight on the subject.`
      }
    ]
  },
  {
    title: "Style Transfer",
    items: [
      {
        src: `ready-to-use/long-select/001657. Re-imagine the entire scene as an ancient fresco p0f677feae6.mp4`,
        prompt: `Re-imagine the entire office scene as a warm antique wall fresco painted on aged plaster. Convert the man, desk, laptop, notebook, shelves, plants, and lamp into hand-painted forms with soft ochre tones, faded blues, simplified outlines, and visible brush texture. Add a worn gilded border, raised plaster grain, and subtle cracks across the image while preserving the original composition, gestures, object layout, and temporal motion.`
      },
      {
        src: `ready-to-use/long-select/005858. Transform the entire scene into a vibrant Fauvist 0147d40c81.mp4`,
        prompt: `Transform the entire scene into a vibrant Fauvist painting. Re-render the woman, her phone, and the background using wild, non-naturalistic colors like electric blues, vivid greens, and intense oranges. Replace all realistic textures with energetic, thick, painterly brushstrokes and bold, contrasting outlines. Simplify her face, clothing, and the glowing lamp into flat, expressive blocks of highly saturated color, abandoning realistic lighting to create a bold, emotionally charged artwork.`
      },
      {
        src: `ready-to-use/long-select/006633. Transform the entire scene into a breathtaking Sci2e7ec2e016.mp4`,
        prompt: `Transform the entire scene into a breathtaking Sci-Fi Art digital painting. Re-render the background as an out-of-focus futuristic cityscape with glowing holographic bokeh and sleek technological structures. Re-imagine the subject in a highly detailed, futuristic illustration style, giving her skin a flawless, subtly luminescent quality. Keep her exact features, pose, and emotional expression intact, while rendering her hair, clothing, and phone with advanced, sleek synthetic textures. Bathe the composition in atmospheric neon blues, cool cyans, and deep purples to reflect a highly advanced civilization.`
      }
    ]
  },
  {
    title: "Object Removal",
    items: [
      {
        src: `ready-to-use/long-select/006131. Remove the white "GagaOOLala" watermark logo locatecd4215c98.mp4`,
        prompt: `Remove the white "GagaOOLala" watermark logo located in the top-left corner of the frame. Seamlessly blend the removed area with the surrounding background, maintaining the natural appearance of the sky, foliage, and building edges.`
      }
    ]
  }
];

const liveDemoVideos = [
  {
    title: "Sunglasses Live Demo",
    src: `live-demo/sunglasses-live-demo.mp4`
  },
  {
    title: "Van Gogh Live Demo",
    src: `live-demo/vangogh-live-demo.mp4`
  }
];

const oneSourceMultipleEdits = [
  {
    src: `ready-to-use/long-one-sample-more-edits/005500. Replace the background with a dimly lit, vintage s593892226e.mp4`,
    prompt: `Replace the background with a dimly lit, vintage speakeasy lounge, leaving the subject and foreground elements entirely unchanged. The new environment should feature out-of-focus dark mahogany wood paneling, antique glass bottles, and softly glowing amber wall sconces. Maintain a shallow depth of field with rich, warm-toned bokeh that seamlessly complements the soft, directional lighting and classic tweed attire of the subject.`
  },
  {
    src: `ready-to-use/long-one-sample-more-edits/005503. Replace the subject's textured grey blazer with a b8ea3a72ab.mp4`,
    prompt: `Replace the subject's textured grey blazer with a plush, deep burgundy velvet smoking jacket. The new garment should feature smooth, black silk peak lapels that softly reflect the warm ambient light of the corridor. Ensure the rich velvet fabric maintains a tailored fit, draping naturally over his shoulders and back to provide a seamless, luxurious silhouette as he speaks, turns, and walks away.`
  },
  {
    src: `ready-to-use/long-one-sample-more-edits/005792. Transform the background into a luxurious high-ris95a3e689b3.mp4`,
    prompt: `Transform the background into a luxurious high-rise executive office. Feature sweeping floor-to-ceiling windows that reveal a gleaming, modern metropolis under a crisp blue daytime sky. Flank the windows with rich, dark walnut wood paneling to provide elegant contrast. Enhance the spatial depth by including a subtly out-of-focus, minimalist bookshelf adorned with abstract metallic sculptures, bathed in soft, diffused natural daylight that harmonizes seamlessly with the scene.`
  },
  {
    src: `ready-to-use/long-one-sample-more-edits/005794. Remove the white flower arrangement and its green 28ff69cff3.mp4`,
    prompt: `Remove the white flower arrangement and its green leaves from the bottom right corner of the scene, leaving a clean, empty desk surface in its place.`
  }
];

const shortVideoGroups = [
  {
    title: "Local Editing",
    items: [
      {
        source: `ready-to-use/selected_short_videos/short-local_0111_local_change_Replace_the_green_mu__original.mp4`,
        edit: `ready-to-use/selected_short_videos/short-local_0111_local_change_Replace_the_green_mu__edited.mp4`,
        prompt: `Replace the green muscle car with a sleek metallic red muscle car, ensuring it maintains the same position and pose within the video scene.`
      },
      {
        source: `ready-to-use/selected_short_videos/short-local_0072_local_change_Replace_the_middle_a__original.mp4`,
        edit: `ready-to-use/selected_short_videos/short-local_0072_local_change_Replace_the_middle_a__edited.mp4`,
        prompt: `Replace the middle-aged man with an elderly gentleman with silver hair and wrinkles, maintaining the same position and pose within the scene.`
      },
      {
        source: `ready-to-use/selected_short_videos/short-local_0228_local_remove_Remove_the_woman_wit__original.mp4`,
        edit: `ready-to-use/selected_short_videos/short-local_0228_local_remove_Remove_the_woman_wit__edited.mp4`,
        prompt: `Remove the woman with shoulder-length blonde hair wearing a black blazer over a black top from the entire video sequence. Ensure temporal consistent background inpainting, and leave all other video content unchanged.`
      },
      {
        source: `ready-to-use/selected_short_videos/short-local_0253_local_add_Overlay_an_animated__original.mp4`,
        edit: `ready-to-use/selected_short_videos/short-local_0253_local_add_Overlay_an_animated__edited.mp4`,
        prompt: `Overlay an animated colorful kite in the upper left sky area of the video. The kite should flutter and sway gently as it flies, with its tail moving naturally in the wind. The kite must be tracked to the sky background as the camera moves, with lighting and shadows adjusting dynamically. All other parts of the video must remain unchanged.`
      }
    ]
  },
  {
    title: "Background Editing",
    items: [
      {
        source: `ready-to-use/selected_short_videos/short-bg_0131_background_change_Transform_the_backgr__original.mp4`,
        edit: `ready-to-use/selected_short_videos/short-bg_0131_background_change_Transform_the_backgr__edited.mp4`,
        prompt: `Transform the background into a modern art gallery. The lighting should subtly shift to highlight different paintings, with occasional soft footsteps and distant murmurs implied. The man in the foreground should remain perfectly still.`
      },
      {
        source: `ready-to-use/selected_short_videos/short-bg_0142_background_change_Replace_the_backgrou__original.mp4`,
        edit: `ready-to-use/selected_short_videos/short-bg_0142_background_change_Replace_the_backgrou__edited.mp4`,
        prompt: `Replace the background with a dynamic ancient Roman forum. Include subtle movement of fluttering banners, distant crowds milling about, birds flying overhead, and sunlight casting moving shadows across the stone surfaces. The subject should remain perfectly still.`
      },
      {
        source: `ready-to-use/selected_short_videos/short-bg_0161_background_change_Create_a_dynamic_cel__original.mp4`,
        edit: `ready-to-use/selected_short_videos/short-bg_0161_background_change_Create_a_dynamic_cel__edited.mp4`,
        prompt: `Create a dynamic celestial night sky background with twinkling stars, slowly drifting nebula clouds, occasional shooting stars streaking across the sky, and a softly glowing moon casting gentle light. The blue sunflowers in the foreground remain perfectly still.`
      },
      {
        source: `ready-to-use/selected_short_videos/short-bg_0179_background_change_Replace_the_backgrou__original.mp4`,
        edit: `ready-to-use/selected_short_videos/short-bg_0179_background_change_Replace_the_backgrou__edited.mp4`,
        prompt: `Replace the background with a dynamic tropical beach at sunset. The scene should include softly rolling ocean waves, palm fronds swaying in a gentle breeze, and warm, shifting colors in the sky as the sun sets. The black Hyundai SUV and the person inside should remain perfectly still.`
      }
    ]
  },
  {
    title: "Style Transfer",
    items: [
      {
        source: `ready-to-use/selected_short_videos/short-style_0005_global_style_Apply_the_Aesthetic__original.mp4`,
        edit: `ready-to-use/selected_short_videos/short-style_0005_global_style_Apply_the_Aesthetic__edited.mp4`,
        prompt: `Apply the Aesthetic Ancient-style to this video, ensuring seamless temporal consistency across all frames. The final output should emulate the aesthetic of ancient hand-painted scrolls or temple murals, with fluid transitions between frames and soft, diffused lighting. All original motion-including character movements, camera panning, and environmental dynamics-must be precisely maintained without disruption.`
      },
      {
        source: `ready-to-use/selected_short_videos/short-style_0025_global_style_Apply_the_dawn_aesth__original.mp4`,
        edit: `ready-to-use/selected_short_videos/short-style_0025_global_style_Apply_the_dawn_aesth__edited.mp4`,
        prompt: `Apply the dawn aesthetic to this video, ensuring seamless temporal consistency. The final output should exude the soft, warm ambiance of early morning, with gradual light transitions and pastel sky tones, all while maintaining the original motion, character actions, and camera movements without distortion.`
      },
      {
        source: `ready-to-use/selected_short_videos/short-style_0047_global_style_Apply_the_Watercolor__original.mp4`,
        edit: `ready-to-use/selected_short_videos/short-style_0047_global_style_Apply_the_Watercolor__edited.mp4`,
        prompt: `Apply the Watercolor animation style to this video, ensuring seamless temporal consistency across all frames. The final result should mirror the soft, painterly aesthetic of watercolor art, with blended colors, fluid transitions, and gentle motion blur that aligns with the medium's organic flow. Preserve the original motion, character actions, camera movements, and narrative flow, ensuring no frame exhibits jarring changes or inconsistencies in color or texture.`
      },
      {
        source: `ready-to-use/selected_short_videos/short-style_0048_global_style_Apply_the_Chinese_In__original.mp4`,
        edit: `ready-to-use/selected_short_videos/short-style_0048_global_style_Apply_the_Chinese_In__edited.mp4`,
        prompt: `Apply the Chinese Ink Wash Painting style to this video, ensuring seamless temporal consistency across all frames. The output should mimic traditional ink wash animation, with fluid ink flow and consistent brushstroke patterns. Preserve all original motion, character actions, and camera movements to maintain narrative coherence.`
      }
    ]
  }
];

const physicalAiGroups = [
  {
    title: "Autonomous Driving",
    items: [
      {
        src: `ready-to-use/autodrive/000009. Transform this front-facing autonomous-driving vidd2f25f0656.mp4`,
        prompt: `Transform this front-facing autonomous-driving video into a light snowfall scene at early morning. Replace rain and mist with gently falling snow, pale blue-gray dawn light, thin snow accumulation along road edges, and softened tree or building silhouettes, while keeping all vehicles, road geometry, lane markings, signs, and motion unchanged.`
      },
      {
        src: `ready-to-use/autodrive/000015. Transform this front-facing autonomous-driving vidb880c9bb87.mp4`,
        prompt: `Transform this front-facing autonomous-driving video into a cold, steady rain scene at dusk. Add wet reflective asphalt, soft gray-blue overcast light, visible raindrops on the windshield, faint tire spray from vehicles, and muted roadside colors, while keeping the same lanes, vehicles, signs, camera motion, and driving trajectory unchanged.`
      }
    ]
  },
  {
    title: "Robotics",
    items: [
      {
        src: `ready-to-use/robotics/000017. Replace every visible human body part in this egocda2a76ec62.mp4`,
        prompt: `Replace every visible human body part in this egocentric manipulation video with a sleek humanoid robot body. Convert all visible hands and forearms into detailed mechanical robot hands and arms, with articulated metal fingers, exposed joints, small cables, and polished dark-silver surfaces. If legs, torso, sleeves, or other body parts appear, render them as matching robotic limbs while preserving the original pose and movement. Keep all surrounding objects, tools, furniture, lighting, camera motion, shadows, and object interactions unchanged, and make the robotic limbs naturally maintain the original contacts, timing, perspective, and temporal consistency throughout the video.`
      },
      {
        src: `ready-to-use/robotics/000251. Replace every visible human body part in this egoc469085a48e.mp4`,
        prompt: `Replace every visible human body part in this egocentric manipulation video with a sleek humanoid robot body. Convert all visible hands and forearms into detailed mechanical robot hands and arms, with articulated metal fingers, exposed joints, small cables, and polished dark-silver surfaces. If legs, torso, sleeves, or other body parts appear, render them as matching robotic limbs while preserving the original pose and movement. Keep all surrounding objects, tools, furniture, lighting, camera motion, shadows, and object interactions unchanged, and make the robotic limbs naturally maintain the original contacts, timing, perspective, and temporal consistency throughout the video.`
      }
    ]
  }
];

const cg2RealItems = [
  {
    src: `cg2real/000007. Transform the CG video to a realistic video.<image195300f9f9.mp4`,
    prompt: `Transform the CG video to a realistic video`
  },
  {
    src: `cg2real/000018. Transform the CG video to a realistic video.<imageca340d1f3b.mp4`,
    prompt: `Transform the CG video to a realistic video`
  }
];

const bibtex = `@article{zhao2026sana,
  title={SANA-Streaming: Real-time Streaming Video Editing with Hybrid Diffusion Transformer},
  author={Zhao, Yuyang and Pan, Yicheng and He, Qiyuan and Yu, Jincheng and Chen, Junsong and Ye, Tian and Liu, Haozhe and Xie, Enze and Han, Song},
  journal={arXiv preprint arXiv:2605.30409},
  year={2026}
}`;

function assetPath(path) {
  return base + path.split("/").map(encodeURIComponent).join("/");
}

function splitVideoPath(path, half) {
  return path.replace(/\.mp4$/, `__${half}.mp4`);
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function escapeAttr(value) {
  return escapeHtml(value);
}

function videoElement(src, eager = false) {
  const attr = eager ? `src="${assetPath(src)}"` : `data-src="${assetPath(src)}"`;
  return `<video ${attr} ${eager ? "autoplay" : ""} muted loop playsinline preload="${eager ? "auto" : "none"}"></video>`;
}

function pairedVideo(src, eager = false) {
  return `
    <div class="paired-video">
      ${videoElement(src, eager)}
    </div>
  `;
}

function heroSequenceDemo() {
  return `
    <div class="hero-demo" aria-label="SANA-Streaming teaser video">
      <video class="hero-teaser" src="https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets/resolve/main/Streaming/assets/teaser/sana-streaming-teaser.mp4?v=20260525ao" poster="https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets/resolve/main/Streaming/assets/logos/teaser-sana-streaming-five-institutions.png" controls playsinline preload="metadata"></video>
    </div>
  `;
}

function promptDrawerHtml(prompt, isOpen) {
  return `
    <div class="prompt-drawer" data-prompt-drawer ${isOpen ? "" : "hidden"}>
      <p>${escapeHtml(prompt)}</p>
    </div>
  `;
}

function promptButtonHtml(isOpen) {
  return `<button class="prompt-toggle" type="button" data-prompt-toggle aria-expanded="${isOpen ? "true" : "false"}">${isOpen ? "Hide prompt" : "Prompt"}</button>`;
}

function compareCard(item, index, options = {}) {
  const promptOpen = options.promptOpen ?? true;
  const isPair = Boolean(item.source && item.edit);
  const mediaAspect = isPair ? "16 / 9" : "1280 / 704";
  const source = isPair ? pairedVideo(item.source) : pairedVideo(splitVideoPath(item.src, "source"));
  const edit = isPair ? pairedVideo(item.edit) : pairedVideo(splitVideoPath(item.src, "edit"));

  return `
    <article class="compare-card" data-compare-card style="--media-aspect: ${mediaAspect}">
      <div class="compare-pair">
        <div class="compare-panel">
          <span class="compare-label compare-label-source">Source</span>
          ${source}
        </div>
        <div class="compare-panel">
          <span class="compare-label compare-label-edit">SANA-Streaming</span>
          ${edit}
        </div>
      </div>
      <div class="compare-meta">
        ${promptButtonHtml(promptOpen)}
      </div>
      ${promptDrawerHtml(item.prompt, promptOpen)}
    </article>
  `;
}

function liveDemoCard(item) {
  return `
    <article class="live-demo-card" data-compare-card>
      <video
        class="live-demo-video"
        data-src="${assetPath(item.src)}"
        aria-label="${escapeAttr(item.title)}"
        muted
        loop
        playsinline
        controls
        preload="none"
      ></video>
    </article>
  `;
}

function oneSourceMultiEditCard(items, rowIndex) {
  const source = pairedVideo(splitVideoPath(items[0].src, "source"));

  return `
    <article class="compare-card multi-edit-card" data-compare-card style="--media-aspect: 1280 / 704">
      <div class="multi-edit-row">
        <div class="compare-panel">
          <span class="compare-label compare-label-source">Source</span>
          ${source}
        </div>
        ${items.map((item, index) => `
          <div class="compare-panel">
            <span class="compare-label compare-label-edit">Edit ${index + 1}</span>
            ${pairedVideo(splitVideoPath(item.src, "edit"))}
          </div>
        `).join("")}
      </div>
      <div class="compare-meta">
        ${promptButtonHtml(true)}
      </div>
      <div class="prompt-drawer multi-edit-prompts" data-prompt-drawer>
        ${items.map((item, index) => `
          <p><strong>Edit ${index + 1}.</strong> ${escapeHtml(item.prompt)}</p>
        `).join("")}
      </div>
    </article>
  `;
}

function oneSourceMultipleEditsSection() {
  const rows = [];
  for (let i = 0; i < oneSourceMultipleEdits.length; i += 2) {
    rows.push(oneSourceMultipleEdits.slice(i, i + 2));
  }

  return `
    <div class="compare-grid one-source-grid">
      ${rows.map(oneSourceMultiEditCard).join("")}
    </div>
  `;
}

function resultGroup(group) {
  return `
    <div class="result-group">
      <h3>${escapeHtml(group.title)}</h3>
      <div class="compare-grid">
        ${group.items.map(compareCard).join("")}
      </div>
    </div>
  `;
}

function render() {
  root.innerHTML = `
    <nav class="nav" aria-label="Primary">
      <a class="brand" href="#top" aria-label="SANA-Streaming home">
        <span class="brand-mark"></span>
        SANA-STREAMING
      </a>
      <div class="nav-links">
        <a href="#features">Features</a>
        <a href="#abstract">Abstract</a>
        <a href="#efficiency">Efficiency</a>
        <a href="#live-demo">Live Demo</a>
        <a href="#streaming">Minute</a>
        <a href="#short-video">Short</a>
        <a href="#physical-ai">Physical AI</a>
        <!-- <a href="#cg2real">CG2Real</a> -->
      </div>
    </nav>

    <header class="title-section" id="top">
      <p class="paper-kicker">Real-time streaming video editing</p>
      <h1>
        <span>SANA-Streaming</span>
        <small>Real-time Streaming Video Editing with Hybrid Diffusion Transformer</small>
      </h1>
      <ul class="authors" aria-label="Authors">
        ${authors.map((author) => `
          <li>
            <a href="${escapeAttr(author.href)}" target="_blank" rel="noopener noreferrer">
              ${escapeHtml(author.name)}${author.equal ? "<sup>*</sup>" : ""}
            </a>
          </li>
          ${author.breakAfter ? `<li class="author-break" aria-hidden="true"></li>` : ""}
        `).join("")}
      </ul>
      <ul class="affiliations" aria-label="Affiliations">
        <li>NVIDIA</li>
        <li>MIT</li>
        <li>THU</li>
        <li>NUS</li>
        <li>HKU</li>
      </ul>
      <p class="equal-contribution">* Equal contribution</p>
      <div class="paper-links">
        <a href="https://arxiv.org/abs/2605.30409" target="_blank" rel="noopener noreferrer">Paper</a>
        <a href="https://github.com/NVlabs/Sana">Code</a>
        <a class="is-featured" href="https://sana-streaming.reactor.inc/" target="_blank" rel="noopener noreferrer">Online Demo <em>live</em></a>
        <button type="button" data-copy-citation>BibTeX</button>
        
      </div>
      ${heroSequenceDemo()}
    </header>

    <section class="section" id="features">
      <p class="eyebrow">Key Features</p>
      <div class="features-grid">
        ${features.map((feature) => `
          <article class="feature-card">
            <p>${escapeHtml(feature.eyebrow)}</p>
            <h2>${escapeHtml(feature.title)}</h2>
            <span>${escapeHtml(feature.body)}</span>
          </article>
        `).join("")}
      </div>
    </section>

    <section class="section abstract-section" id="abstract">
      <p class="eyebrow">Abstract</p>
      <div class="abstract-copy">
        <p>Real-time streaming video-to-video editing (V2V) is critical for interactive applications such as live broadcasting and gaming, yet it remains a formidable challenge due to the stringent requirements for temporal consistency and inference throughput. In this paper, we present <strong>SANA-Streaming</strong>, a system-algorithm co-designed framework for high-resolution, real-time streaming video editing on consumer GPUs, with the following three core designs: (1) <strong>Hybrid Diffusion Transformer architecture</strong> introduces softmax attention in part of the blocks to improve local modeling capabilities while preserving the efficiency of linear layers. (2) <strong>Cycle-Reverse Regularization</strong> is a novel training strategy that enforces semantic consistency by predicting source frames from generated content via flow matching, improving temporal consistency without requiring paired long edited videos. (3) <strong>Efficient System Co-design</strong> combines fused GDN kernels and Mixed-Precision Quantization (MPQ) optimized for the NVIDIA Blackwell (RTX 5090) architecture. By profiling real-world throughput, our MPQ maximizes Tensor Core utilization while maintaining generation quality. The resulting system achieves real-time 1280 x 704 resolution editing at <strong>24 end-to-end FPS</strong> on a single RTX 5090 GPU, with the DiT core running at <strong>58 FPS</strong>. Experimental results demonstrate that our co-design approach significantly outperforms existing SOTA methods in both temporal coherence and system throughput.</p>
      </div>
    </section>

    <section class="section" id="efficiency">
      <p class="eyebrow">Efficiency at a Glance</p>
      <div class="speed-figures">
        <figure class="speed-figure">
          <figcaption>DiT latency w.r.t. video length</figcaption>
          <div class="speed-image-frame">
            <img src="https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets/resolve/main/Streaming/assets/figures/dit-latency.png" alt="DiT latency comparison between all-softmax and SANA-Streaming hybrid attention." loading="lazy" />
          </div>
        </figure>
        <figure class="speed-figure">
          <figcaption>GPU memory w.r.t. video length</figcaption>
          <div class="speed-image-frame">
            <img src="https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets/resolve/main/Streaming/assets/figures/gpu-vram.png" alt="GPU VRAM comparison between all-softmax and SANA-Streaming hybrid attention." loading="lazy" />
          </div>
        </figure>
        <figure class="speed-figure">
          <figcaption>Latency breakdown of 45s videos</figcaption>
          <div class="speed-image-frame">
            <img src="https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets/resolve/main/Streaming/assets/figures/latency-breakdown.png" alt="Latency breakdown showing speedups from hybrid attention, GDN kernel, and MPQ." loading="lazy" />
          </div>
        </figure>
      </div>
      <p class="speed-caption"><strong>Figure 1.</strong> Efficiency measured on a single NVIDIA RTX 5090 GPU, including DiT latency, GPU memory usage, and latency breakdown for 45-second videos.</p>
    </section>

    <section class="section results-section" id="live-demo">
      <p class="eyebrow">Live Demo</p>
      <div class="result-stack">
        <div class="live-demo-grid">
          ${liveDemoVideos.map(liveDemoCard).join("")}
        </div>
      </div>
    </section>

    <section class="section results-section" id="streaming">
      <p class="eyebrow">Minute-Length Streaming Editing</p>
      <div class="result-stack">
        <div class="compare-grid">
          ${minuteLengthGroups.flatMap((group) => group.items).map(compareCard).join("")}
        </div>
      </div>
    </section>

    <section class="section results-section" id="one-source">
      <p class="eyebrow">One Source, Multiple Edits</p>
      <div class="result-stack">
        ${oneSourceMultipleEditsSection()}
      </div>
    </section>

    <section class="section results-section" id="short-video">
      <p class="eyebrow">Short Video Editing</p>
      <div class="result-stack">
        ${shortVideoGroups.map(resultGroup).join("")}
      </div>
    </section>

    <section class="section results-section" id="physical-ai">
      <p class="eyebrow">Physical AI</p>
      <div class="result-stack">
        ${physicalAiGroups.map(resultGroup).join("")}
      </div>
    </section>

    ${/*
    <section class="section results-section" id="cg2real">
      <p class="eyebrow">CG2Real</p>
      <div class="result-stack">
        <div class="compare-grid">
          ${cg2RealItems.map(compareCard).join("")}
        </div>
      </div>
    </section>
    */ ""}

    <section class="section citation-section" id="citation">
      <p class="eyebrow">Citation</p>
      <div class="citation-block">
        <button class="copy-btn" type="button" data-copy-citation>Copy</button>
        <pre><code>${escapeHtml(bibtex)}</code></pre>
      </div>
    </section>
  `;
}

function setupPromptButtons() {
  document.querySelectorAll("[data-prompt-toggle]").forEach((button) => {
    const card = button.closest("[data-compare-card]");
    const drawer = card.querySelector("[data-prompt-drawer]");

    button.addEventListener("click", () => {
      const isOpen = button.getAttribute("aria-expanded") === "true";
      const nextOpen = !isOpen;
      button.setAttribute("aria-expanded", String(nextOpen));
      button.textContent = nextOpen ? "Hide prompt" : "Prompt";
      drawer.hidden = !nextOpen;
      card.classList.toggle("is-prompt-open", nextOpen);
    });
  });
}

function setupLazyVideos() {
  const cards = [...document.querySelectorAll("[data-compare-card]")];

  const loadCard = (card) => {
    card.querySelectorAll("video[data-src]").forEach((video) => {
      if (!video.src) {
        video.src = video.dataset.src;
        video.load();
      }
    });
  };

  const playCard = (card) => {
    const videos = [...card.querySelectorAll("video")];
    videos.forEach((video) => video.play().catch(() => {}));
  };

  const pauseCard = (card) => {
    card.querySelectorAll("video").forEach((video) => video.pause());
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      const card = entry.target;
      if (entry.isIntersecting) {
        loadCard(card);
        playCard(card);
        card.dataset.visible = "true";
      } else {
        pauseCard(card);
        card.dataset.visible = "false";
      }
    });
  }, { rootMargin: "360px 0px", threshold: 0.08 });

  cards.forEach((card) => observer.observe(card));
}

function setupVideoSync() {
  const sync = () => {
    const cards = [
      ...document.querySelectorAll("[data-sync-card]"),
      ...document.querySelectorAll('[data-compare-card][data-visible="true"]')
    ];

    cards.forEach((card) => {
      const videos = [...card.querySelectorAll("video")];
      if (videos.length < 2 || !videos[0].src) return;
      const leader = videos[0];

      videos.slice(1).forEach((video) => {
        if (!video.src) return;
        if (Math.abs(leader.currentTime - video.currentTime) > 0.08) {
          video.currentTime = leader.currentTime;
        }
      });
    });
    requestAnimationFrame(sync);
  };
  requestAnimationFrame(sync);
}

function setupActions() {
  document.querySelectorAll("[data-copy-citation]").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(bibtex);
        const old = button.textContent;
        button.textContent = "Copied";
        window.setTimeout(() => {
          button.textContent = old;
        }, 1500);
      } catch {
        window.prompt("Copy BibTeX", bibtex);
      }
    });
  });

  document.querySelectorAll(".is-disabled").forEach((link) => {
    link.addEventListener("click", (event) => event.preventDefault());
  });
}

function scrollToInitialHash() {
  const jump = new URLSearchParams(window.location.search).get("jump");
  const selector = jump ? `#${CSS.escape(jump)}` : window.location.hash;
  if (!selector) return;
  const target = document.querySelector(selector);
  if (!target) return;
  window.scrollTo({
    top: target.getBoundingClientRect().top + window.scrollY,
    left: 0,
    behavior: "auto"
  });
}

render();
setupPromptButtons();
setupLazyVideos();
setupVideoSync();
setupActions();
scrollToInitialHash();
window.addEventListener("load", () => window.setTimeout(scrollToInitialHash, 80));
