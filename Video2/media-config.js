/* Add HLS/MP4 URLs here when final media is ready. Posters remain visible until playback begins. */
window.SANA_VIDEO_MEDIA = {
  assetBase: "https://huggingface.co/datasets/Efficient-Large-Model/Sana-assets/resolve/main/Video2",
  curated: {
    root: "assets/curated-20260812",
    metadata: "assets/curated-20260812/metadata/dataset.json",
    cinematicSamples: [0, 2, 4, 5, 8, 9],
    motionSamples: [1, 3, 6, 7],
    titles: [
      "Lunar astronaut",
      "Iridescent dragon",
      "Starlit horse",
      "Rooster at ease",
      "Garden stillness",
      "Alien laboratory",
      "Joyful portrait",
      "Coastal Jaguar",
      "Studio mix",
      "Night-sky figure"
    ]
  },
  intro: [
    { title: "Village in a bottle", poster: "assets/posters/cinematic-01.webp", duration: "8 s", hls: "assets/hls/cinematic-01/index.m3u8", mp4: "" },
    { title: "Snowbound cottage", poster: "assets/posters/cinematic-02.webp", duration: "8 s", hls: "assets/hls/cinematic-02/index.m3u8", mp4: "" },
    { title: "Castle approach", poster: "assets/posters/cinematic-03.webp", duration: "8 s", hls: "assets/hls/cinematic-03/index.m3u8", mp4: "" },
    { title: "Morning close-up", poster: "assets/posters/cinematic-04.webp", duration: "8 s", hls: "assets/hls/cinematic-04/index.m3u8", mp4: "" },
    { title: "Hawk in flight", poster: "assets/posters/cinematic-05.webp", duration: "8 s", hls: "assets/hls/cinematic-05/index.m3u8", mp4: "" },
    { title: "Tropical flyover", poster: "assets/posters/cinematic-06.webp", duration: "8 s", hls: "assets/hls/cinematic-06/index.m3u8", mp4: "" },
    { title: "Popcorn burst", poster: "assets/posters/motion-01.webp", duration: "8 s", hls: "assets/hls/motion-01/index.m3u8", mp4: "" },
    { title: "Poodle sprint", poster: "assets/posters/motion-02.webp", duration: "8 s", hls: "assets/hls/motion-02/index.m3u8", mp4: "" },
    { title: "Armored warrior", poster: "assets/posters/motion-03.webp", duration: "8 s", hls: "assets/hls/motion-03/index.m3u8", mp4: "" },
    { title: "Pack in motion", poster: "assets/posters/motion-04.webp", duration: "8 s", hls: "assets/hls/motion-04/index.m3u8", mp4: "" },
    { title: "Anime wave", poster: "assets/posters/motion-05.webp", duration: "8 s", hls: "assets/hls/motion-05/index.m3u8", mp4: "" },
    { title: "Goldfish cyclist", poster: "assets/posters/motion-06.webp", duration: "8 s", hls: "assets/hls/motion-06/index.m3u8", mp4: "" }
  ],
  sections: [
    {
      id: "cinematic",
      kicker: "01 · Generated results",
      title: "Cinematic Stories",
      description: "Character, expression, lighting, and coherent visual storytelling at 720p.",
      gridColumns: 3,
      batches: [
        [
          { title: "Village in a bottle", prompt: "A large glass bottle, sealed with a cork, drifts on a turbulent, dark sea. Inside is a serene, miniature Scandinavian village with red wooden houses, a church, green trees, and blue skies with white clouds. This tranquil scene contrasts sharply with the stormy sea outside, symbolizing a peaceful refuge. Dramatic aerial shot, close-up of the bottle then pull back to show the tumultuous sea.", poster: "assets/posters/cinematic-01.webp", duration: "8 s", resolution: "1280×736", width: 1280, height: 736, hls: "assets/hls/cinematic-01/index.m3u8", mp4: "assets/mp4/cinematic-01.mp4" },
          { title: "Snowbound cottage", prompt: "At sunset, a medium shot of a pineapple-shaped cottage with detailed leaf textures, covered in snow, and outlined by the edge light of the orange and purple sky. Behind it, a dark coniferous forest with mist diffusing through, creating layers and mystery. Fireworks bloom above, casting cool-toned reflections on the thick snow in front of the slightly ajar, colorful door. Warm light glows softly around the doorframe. Eye-level camera angle, center composition, with snow extending into the foreground to enhance depth.", poster: "assets/posters/cinematic-02.webp", duration: "8 s", resolution: "1280×736", width: 1280, height: 736, hls: "assets/hls/cinematic-02/index.m3u8", mp4: "assets/mp4/cinematic-02.mp4" },
          { title: "Castle approach", prompt: "A formidable medieval stone castle rises above a green river valley as the camera flies steadily toward its towers.", poster: "assets/posters/cinematic-03.webp", duration: "8 s", resolution: "1280×736", width: 1280, height: 736, hls: "assets/hls/cinematic-03/index.m3u8", mp4: "assets/mp4/cinematic-03.mp4" }
        ],
        [
          { title: "Morning close-up", prompt: "A close-up focuses on a woman resting against a pillow as she slowly opens her eyes in warm morning light.", poster: "assets/posters/cinematic-04.webp", duration: "8 s", resolution: "1280×736", width: 1280, height: 736, hls: "assets/hls/cinematic-04/index.m3u8", mp4: "assets/mp4/cinematic-04.mp4" },
          { title: "Hawk in flight", prompt: "A majestic hawk with detailed brown-and-white plumage glides through the open sky above a distant landscape.", poster: "assets/posters/cinematic-05.webp", duration: "8 s", resolution: "1280×736", width: 1280, height: 736, hls: "assets/hls/cinematic-05/index.m3u8", mp4: "assets/mp4/cinematic-05.mp4" },
          { title: "Tropical flyover", prompt: "An aerial shot circles a small verdant tropical island surrounded by clear turquoise water and bright white sand.", poster: "assets/posters/cinematic-06.webp", duration: "8 s", resolution: "1280×736", width: 1280, height: 736, hls: "assets/hls/cinematic-06/index.m3u8", mp4: "assets/mp4/cinematic-06.mp4" }
        ]
      ]
    },
    {
      id: "motion",
      kicker: "02 · Generated results",
      title: "Motion & Dynamics",
      description: "Fast subjects, fluid motion, and temporally consistent scene evolution.",
      gridColumns: 2,
      batches: [
        [
          { title: "Popcorn burst", prompt: "Numerous pieces of freshly popped popcorn burst upward in a crisp slow-motion macro shot.", poster: "assets/posters/motion-01.webp", duration: "8 s", resolution: "1280×704", width: 1280, height: 704, hls: "assets/hls/motion-01/index.m3u8", mp4: "assets/mp4/motion-01.mp4" },
          { title: "Poodle sprint", prompt: "A fluffy white poodle, meticulously groomed, runs toward the camera across a sunlit wooden floor.", poster: "assets/posters/motion-02.webp", duration: "8 s", resolution: "1280×704", width: 1280, height: 704, hls: "assets/hls/motion-02/index.m3u8", mp4: "assets/mp4/motion-02.mp4" },
          { title: "Armored warrior", prompt: "A middle-aged warrior clad in worn leather and metal armor walks steadily across a sunlit dusty field.", poster: "assets/posters/motion-03.webp", duration: "8 s", resolution: "1280×704", width: 1280, height: 704, hls: "assets/hls/motion-03/index.m3u8", mp4: "assets/mp4/motion-03.mp4" }
        ],
        [
          { title: "Pack in motion", prompt: "Hundreds of diverse dogs of many breeds and sizes run together across an open green field.", poster: "assets/posters/motion-04.webp", duration: "8 s", resolution: "1280×704", width: 1280, height: 704, hls: "assets/hls/motion-04/index.m3u8", mp4: "assets/mp4/motion-04.mp4" },
          { title: "Anime wave", prompt: "A young woman animated in a distinct anime style smiles and waves toward the camera.", poster: "assets/posters/motion-05.webp", duration: "8 s", resolution: "1280×736", width: 1280, height: 736, hls: "assets/hls/motion-05/index.m3u8", mp4: "assets/mp4/motion-05.mp4" },
          { title: "Goldfish cyclist", prompt: "A vibrant orange goldfish enclosed in a perfectly clear water bubble rides a small red bicycle through a park.", poster: "assets/posters/motion-06.webp", duration: "8 s", resolution: "1280×736", width: 1280, height: 736, hls: "assets/hls/motion-06/index.m3u8", mp4: "assets/mp4/motion-06.mp4" }
        ]
      ]
    },
    {
      id: "physical-ai",
      kicker: "04 · Embodied intelligence",
      title: "Physical AI",
      description: "Bimanual robot manipulation across everyday kitchen tasks.",
      gridColumns: 3,
      batches: [
        [
          { title: "Sugar container", prompt: "Grab the lid of the canned sugar on the table with the right arm.", poster: "assets/posters/physical-ai-01.webp", duration: "5 s", resolution: "640×360", width: 640, height: 360, hls: "assets/hls/physical-ai-01/index.m3u8", mp4: "assets/mp4/physical-ai-01.mp4" },
          { title: "Medicine box", prompt: "Place the lifted medicine box cover onto the medicine box with the right arm.", poster: "assets/posters/physical-ai-02.webp", duration: "5 s", resolution: "640×360", width: 640, height: 360, hls: "assets/hls/physical-ai-02/index.m3u8", mp4: "assets/mp4/physical-ai-02.mp4" },
          { title: "Dough cutting", prompt: "Cut the dough into small pieces with a knife using both hands.", poster: "assets/posters/physical-ai-03.webp", duration: "5 s", resolution: "640×360", width: 640, height: 360, hls: "assets/hls/physical-ai-03/index.m3u8", mp4: "assets/mp4/physical-ai-03.mp4" }
        ],
        [
          { title: "Ketchup bottle", prompt: "Flip the picked-up ketchup bottle with the left arm.", poster: "assets/posters/physical-ai-04.webp", duration: "5 s", resolution: "640×360", width: 640, height: 360, hls: "assets/hls/physical-ai-04/index.m3u8", mp4: "assets/mp4/physical-ai-04.mp4" },
          { title: "Pineapple bun", prompt: "Use the picked-up dessert spatula with the right arm to scoop up mini pineapple bun.", poster: "assets/posters/physical-ai-05.webp", duration: "5 s", resolution: "640×360", width: 640, height: 360, hls: "assets/hls/physical-ai-05/index.m3u8", mp4: "assets/mp4/physical-ai-05.mp4" },
          { title: "Rice cooker", prompt: "Press the lid-opening button with the left arm to open the rice cooker lid.", poster: "assets/posters/physical-ai-06.webp", duration: "5 s", resolution: "640×360", width: 640, height: 360, hls: "assets/hls/physical-ai-06/index.m3u8", mp4: "assets/mp4/physical-ai-06.mp4" }
        ]
      ]
    }
  ]
};
