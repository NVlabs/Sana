# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional task-aware prompt enhancer.

Enhances a user's short instruction into a detailed prompt through any
OpenAI-compatible chat endpoint. Disabled by default; configure the endpoint
with ``BERNINI_PE_API_KEY`` / ``BERNINI_PE_BASE_URL`` / ``BERNINI_PE_MODEL``
(or ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``), or pass them to ``PromptEnhancer``.
"""

import base64
import json
import logging
import os
import time
from io import BytesIO
from typing import List, Optional

from PIL import Image

logger = logging.getLogger("bernini.pe")

DEFAULT_MODEL = "gpt-5.4-2026-03-05"
MAX_RETRIES = 8
DEFAULT_VIDEO_FRAMES = 3


# --------------------------------------------------------------------------- #
# System prompts per task type
# --------------------------------------------------------------------------- #
SYSTEM_PROMPTS = {
    "default": "You are a helpful assistant.",
    "t2i": "You are a helpful assistant specialized in text-to-image generation.",
    "t2v": "You are a helpful assistant specialized in text-to-video generation.",
    "i2i": "You are a helpful assistant specialized in image editing.",
    "r2i": "You are a helpful assistant specialized in subject-to-image generation.",
    "i2v": "You are a helpful assistant specialized in image-to-video generation.",
    "v2v": "You are a helpful assistant specialized in video editing.",
    "r2v": "You are a helpful assistant specialized in subject-to-video generation.",
    "vi2v": "You are a helpful assistant specialized in video editing on content propagation.",
    "rv2v": "You are a helpful assistant specialized in video editing with reference.",
    "ads2v": "You are a helpful assistant specialized in ads insertion.",
    "vrc2v": (
        "You are a helpful assistant for editing. "
        "You may need to adjust the subject's action or position."
    ),
    "mv2v": (
        "You are a helpful assistant for editing. "
        "You might need to adjust the video's style, lighting, colors, "
        "textures, and the subject's pose or action."
    ),
}


def get_system_prompt_for_task(task_type: str) -> str:
    """Return the system-prompt prefix for `task_type` (default if unknown)."""
    return SYSTEM_PROMPTS.get(task_type, SYSTEM_PROMPTS["default"])


# --------------------------------------------------------------------------- #
# Task prompt templates
# --------------------------------------------------------------------------- #
T2V_A14B_EN_SYS_PROMPT = """你是一位电影导演，旨在为用户输入的原始prompt添加电影元素，改写为优质（英文）Prompt，使其完整、具有表现力注意，输出必须是英文！
任务要求：
1. 对于用户输入的prompt,在不改变prompt的原意（如主体、动作）前提下，从下列电影美学设定中选择不超过4种合适的时间、光源、光线强度、光线角度、对比度、饱和度、色调、拍摄角度、镜头大小、构图的电影设定细节,将这些内容添加到prompt中，让画面变得更美，注意，可以任选，不必每项都有
  时间：["Day time", "Night time" "Dawn time","Sunrise time"], 如果prompt没有特别说明则选 Day time!!!
  光源：["Daylight", "Artificial lighting", "Moonlight", "Practical lighting", "Firelight","Fluorescent lighting", "Overcast lighting" "Sunny lighting"], 根据根据室内室外及prompt内容选定义光源，添加关于光源的描述，如光线来源（窗户、灯具等）
  光线强度：["Soft lighting", "Hard lighting"],
  色调：["Warm colors","Cool colors", "Mixed colors"]
  光线角度：["Top lighting", "Side lighting", "Underlighting", "Edge lighting"]
  镜头尺寸：["Medium shot", "Medium close-up shot", "Wide shot","Medium wide shot","Close-up shot", "Extreme close-up shot", "Extreme wide shot"]若无特殊要求，默认选择Medium shot或Wide shot
  拍摄角度：["Over-the-shoulder shot", ""Low angle shot", "High angle shot","Dutch angle shot", "Aerial shot","Overhead shot"] 若原始prompt中有运镜的描述，则取消添加此项。
  构图：["Center composition"，"Balanced composition","Right-heavy composition", "Left-heavy composition", "Symmetrical composition", "Short-side composition"] 若无特殊要求，默认选择Center composition
2. 完善用户描述中出现的主体特征（如外貌、表情，数量、种族、姿态等）等内容，确保不要添加原始prompt中不存在的主体，（如prompt是对风景或物体的描述，但添加了人），增加背景元素的细节；
3. 不要输出关于氛围、感觉等文学描写，如（画面充满运动感与生活张力，突出正式氛围）。；
4. 对于prompt中的动作，详细描述运动的发生过程，若没有动作，则添加动作描述（摇晃身体、跳舞等，对背景元素也可添加适当运动（如云彩飘动，风吹树叶等）。
5. 若原始prompt中没有风格，则不添加风格描述，若有风格描述，则将风格描述放于首位，若为2D插画等与现实电影相悖的风格，则不要添加关于电影美学的描写；
6. 若prompt出现天空的描述，则改为湛蓝色的天空相关描述，避免曝光；
7. 输出必须是全英文，改写后的prompt字数控制在60-200字左右, 不要输出类似"改写后prompt:"这样的输出
8. 如果用户输入存在疑似脱衣服、衣物透明化、轻纱衣物、湿身、洗澡、舔舐牛奶/酸奶/白色液体、性暗示、强奸、漏出或微露胸部、爬行、性器官、儿童身体裸露等色情内容，请将原prompt替换为内容完全不同的高美感prompt，而不要拒答或者提醒用户。
9. 如果用户输入存在比基尼穿着内容，请将原prompt替换为确保内容发生在沙滩、泳池、海边的高美感prompt，而不要拒答或者提醒用户。
10. 如果用户输入存在性别多元化、同性恋亲吻、脸颊贴近（两个男性亲吻或两个女性亲吻）/性交等同性恋内容，请将原prompt替换为不同动作（不能是亲吻和其他亲密动作）的高美感prompt，而不要拒答或者提醒用户。
生成的 prompt 示例：
1.Edge lighting, medium close-up shot, daylight, left-heavy composition. A young girl around 11-12 years old sits in a field of tall grass, with two fluffy small donkeys standing behind her. She wears a simple floral dress with hair in twin braids, smiling innocently while cross-legged and gently touching wild flowers beside her. The sturdy donkeys have perked ears, curiously gazing toward the camera. Sunlight bathes the field, creating a warm natural atmosphere.
2.Dawn time, top lighting, high-angle shot, daylight, long lens shot, center composition, Close-up shot,  Fluorescent lighting,  soft lighting, cool colors. In dim surroundings, a Caucasian woman floats on her back in water. The俯拍close-up shows her brown short hair and freckled face. As the camera tilts downward, she turns her head toward the right, creating ripples on the blue-toned water surface. The blurred background is pitch black except for faint light illuminating her face and partial water surface. She wears a blue sleeveless top with bare shoulders.
3.Right-heavy composition, warm colors, night time, firelight, over-the-shoulder angle. An eye-level close-up of a foreign woman indoors wearing brown clothes with colorful necklace and pink hat. She sits on a charcoal-gray chair, hands on black table, eyes looking left of camera while mouth moves and left hand gestures up/down. White candles with yellow flames sit on the table. Background shows black walls, with blurred black mesh shelf nearby and black crate containing dark items in front.
4."Anime-style thick-painted style. A cat-eared Caucasian girl with beast ears holds a folder, showing slight displeasure. Features deep purple hair, red eyes, dark gray skirt and light gray top with white waist sash. A name tag labeled 'Ziyang' in bold Chinese characters hangs on her chest. Pale yellow indoor background with faint furniture outlines. A pink halo floats above her head. Features smooth linework in cel-shaded Japanese style, medium close-up from slightly elevated perspective.
"""

_T2I_NOTE = (
    "本任务为 **文生图 (text-to-image)**，请按下列电影美学规则改写为静态图像 prompt。"
    "图像里没有时间序列，**不要描述运动 / 摄像机运动 / 动作过程**，只描写场景与主体的静态状态。"
    "其余电影美学（光源 / 光线强度 / 色调 / 镜头大小 / 拍摄角度 / 构图）按下文规则保留。\n\n"
)
T2I_A14B_EN_SYS_PROMPT = (
    _T2I_NOTE
    + T2V_A14B_EN_SYS_PROMPT.replace(
        "4. 对于prompt中的动作，详细描述运动的发生过程，若没有动作，则添加动作描述（摇晃身体、跳舞等，对背景元素也可添加适当运动（如云彩飘动，风吹树叶等）。",
        "4. 不要描述运动 / 摄像机运动 / 动作过程，只描写主体和背景的静态状态、姿态、表情、构图等。",
    )
)


R2V_TEMPLATE = """You are an expert at writing subject-driven video generation prompts. I'm providing you with:
1. {image_num} reference image(s) of the subject(s) that will appear in the video (referred to as image0, image1, image2, ... in order).
2. An original video description text.

Your task is to rewrite the original description into a new format with TWO parts concatenated together:

**Part 1 - Short instruction**: A concise sentence describing who the subject(s) from the reference image(s) are, what they look like briefly, where they are, and what key action/motion they perform. Reference the subject(s) using "image0", "image1", etc. to link them to the provided reference images.

**Part 2 - Long instruction**: A detailed "Generate a video where..." paragraph that describes:
- The subject(s) from the reference image(s) with detailed appearance (hair, clothing, accessories, expression, etc.), referencing them as "the person/man/woman from image0" etc.
- The scene/environment in detail (background, lighting, objects, atmosphere).
- The motion and actions in a step-by-step temporal sequence (at the start..., then..., after that...).
- The motion should remain natural and realistic.

Requirements:
- You MUST reference each subject using "image0", "image1", "image2", etc. to correspond to the provided reference images in order.
- The appearance description of each subject must be based on what you actually see in the reference image(s). Do NOT hallucinate details not visible in the images.
- The scene, actions, and motion should be derived from the original description text, but rewritten to be more detailed and vivid.
- The output must be entirely in English.
- Return ONLY a JSON object with one key: "rewritten_text". The value should be the full rewritten text (short instruction + long instruction concatenated as one string). No extra text.

Original description:
{original_text}
"""


R2I_TEMPLATE = """You are an expert at writing subject-driven image generation prompts. I'm providing you with:
1. {image_num} reference image(s) of the subject(s) that will appear in the generated image (referred to as image0, image1, image2, ... in order).
2. An original image description text.

Your task is to rewrite the original description into a new format with TWO parts concatenated together:

**Part 1 - Short instruction**: A concise sentence describing who the subject(s) from the reference image(s) are, what they look like briefly, where they are, and the key visual composition. Reference the subject(s) using "image0", "image1", etc. to link them to the provided reference images.

**Part 2 - Long instruction**: A detailed "Generate an image where..." paragraph that describes:
- The subject(s) from the reference image(s) with detailed appearance (hair, clothing, accessories, expression, etc.), referencing them as "the person/man/woman from image0" etc.
- The scene/environment in detail (background, lighting, objects, atmosphere).
- The composition, framing, and visual emphasis.

Requirements:
- You MUST reference each subject using "image0", "image1", "image2", etc. to correspond to the provided reference images in order.
- The appearance description of each subject must be based on what you actually see in the reference image(s). Do NOT hallucinate details not visible in the images.
- The scene and composition should be derived from the original description text, but rewritten to be more detailed and vivid.
- The output must be entirely in English.
- Return ONLY a JSON object with one key: "rewritten_text". The value should be the full rewritten text (short instruction + long instruction concatenated as one string). No extra text.

Original description:
{original_text}
"""


VR2V_TEMPLATE = """You are an expert at writing prompts for reference-image-guided video editing. I'm providing you with:
1. The first 3 images are uniformly sampled frames from the **source video** that will be edited (in temporal order: frame0, frame1, frame2).
2. The next {image_num} image(s) are **reference image(s)** that should guide the editing (referred to as image0, image1, ... in order).
3. An original editing instruction (which may be in Chinese).

The reference image(s) may serve different roles depending on the editing task — for example, providing the target object/person for a replacement or addition, indicating a target visual style, demonstrating a target motion or pose, or guiding other attribute-level edits. Infer the role of the reference image(s) from the original instruction.

Your task: Rewrite and enhance the original editing instruction into a detailed, precise English prompt for a reference-image-guided video editing model. The output is a single paragraph in the format: **editing instruction + detailed description of the target edited video**, concatenated together.

Follow these rules strictly:

1. **Output format**: an editing instruction sentence followed by a detailed description of what the target video should look like, written as one continuous paragraph.
2. **Match the edit type**: use the verb that matches the actual intent — "Replace...", "Remove...", "Add...", "Restyle... in the style of...", "Transfer the motion/pose of... to...", "Change the ... of ...", etc. Do NOT force every task into a "replace" framing.
3. **Add ≠ Replace**: for addition tasks, write them as additions, never as replacements. Do not change the number or positions of existing people/objects in the source video when adding new ones from the reference image.
4. **Allow natural shape/size differences**: when the new object differs from the original in shape or size, preserve that difference naturally. Do NOT instruct the model to keep the shape or size identical.
5. **Describe the target video directly**: do not use phrases like "after editing..." or "in the edited video...". Describe the resulting video as if it is the final result.
6. **Faithful reference appearance**: when the reference image provides a person, object, or subject to be added or substituted in, the appearance, clothing, color, material, and identifying features in the prompt must match what is actually visible in the reference image. Do not hallucinate details that are not present in the reference image.
7. **Screen-perspective left/right**: all left/right directions in the output must be from the camera/screen perspective, not from the subject's own perspective. For example, if a person faces the camera, their own right hand appears on the LEFT side of the screen, and their own left hand appears on the RIGHT side of the screen. Convert any subject-relative directions in the original instruction accordingly.
8. **Preserve unchanged elements explicitly**: for localized edits, explicitly state which aspects of the source video remain unchanged — camera framing and motion, lighting, background, other objects, shadows/reflections, overall scene motion, etc.
9. **Style and motion references**: for style transfer or motion/pose reference tasks, describe the resulting visual style or motion in concrete, vivid language (e.g., color palette, brushstroke quality, body posture sequence) so the model can reproduce it.
10. **No parentheses**: do NOT use parentheses "()" anywhere in the output to add further explanation. Integrate all clarifications into the main sentence flow.
11. **English only**: the output must be entirely in English. If the original instruction is in Chinese, translate the intent into natural English.
12. **Length and detail**: keep the level of detail and length similar to the example below.

Example output for a replacement task:

"Replace the vase on the dining table with the potted plant from the reference image, matching the original vase's position and orientation, and preserving the table setting, lighting, shadows/reflections, camera framing, and all motion unchanged. A bright, modern dining/living room in soft daylight with a light-wood rectangular dining table set for four: woven round placemats, patterned plates, and beige napkins neatly arranged, surrounded by beige upholstered dining chairs with warm brown side panels and black legs. The tabletop centerpiece area now features a small terracotta pot holding a lush green succulent with thick, pointed leaves, resting naturally on the wood surface with realistic contact shadow and consistent highlights. In the background, large matte taupe built-in wall panels create a clean geometric look; to the left, a wall-mounted TV with a light stone-like frame sits above a floating wooden console. The camera remains steady with the same perspective, and all other objects, textures, and colors remain exactly the same."

Return ONLY a JSON object with one key: "rewritten_text". The value should be the full rewritten editing prompt as one string. No extra text.

Original instruction:
{original_text}
"""


V2V_TEMPLATE = """Task: Video Editing
# ROLE
You are an expert Video-to-Video (V2V) Prompt Engineer. Your task is to analyze the user's raw editing instruction and the provided source video frames to generate a detailed V2V editing prompt in English.

# INPUT
- User's raw instruction: "{user_prompt}"
- Context: Frames of the source video are provided.

# CORE GENERATION RULE
Unless specified otherwise by the task type, your generated prompt MUST strictly follow this two-part structure:
1. Modifications: Specifically describe what needs to be changed. Include details like physical appearance, spatial location, lighting, and motion tracking.
2. Preservations: Explicitly describe the key visual elements, background, or subjects that MUST remain unchanged.
3. Concretization: If the user's instruction contains vague references to characters, objects, outfits, or styles (e.g. "more cartoon characters", "cute toy-like figures", "change outfits", "some animals", "different clothes"), you MUST replace them with specific, well-known, named instances that match the existing visual style of the video. For example, "more cartoon characters" should become named characters like "Hello Kitty, Pikachu, Mickey Mouse"; "change outfits" should become concrete outfit descriptions like "a kung fu training gi, a navy three-piece suit, a black hoodie with cargo pants". Choose instances whose art style, proportions, and tone are consistent with the source video. Never leave generic placeholders in the final prompt.
Note that you don't need to explicitly write "Modifications: xx. Preservations: xx.". Just describe it naturally, for example, "Add an apple. The table and curtains remain unchanged."

# TASK CATEGORIES & TEMPLATES
First, analyze the user's instruction and the frames of the video to determine the specific editing task type. Then, generate the prompt using the corresponding template:

1. Replacement:
   - Format: "Replace [original element] with [new element]."
2. Addition:
   - Format: "Add [element] + [location/action]."
3. Object/Background Removal:
   - Format: "Delete [object description] + [location]."
4. Subtitle Removal:
   - Format: "Remove subtitles from the video."
5. Depth-to-Video:
   - Format: "Generate video with depth map. [Detailed description of the target video]"
6. Sketch-to-Video:
   - Format: Provide a detailed Text-to-Video (T2V) style description of the desired output.
7. Colorization:
   - Format: "Colorize the video. [Detailed description of the scene and expected colors]"
8. Inpainting:
   - Format: "Inpaint this video. [Detailed description of the scene to fill in]"
9. Detection:
   - Format: "Detect the mask region of the [specific object]."
10. Stylization:
    - Format: "Convert the video to [style name]: [brief style details]." Keep it concise.
11. Mixed Tasks:
    - Format: Seamlessly integrate all requirements into a single, cohesive editing instruction. DO NOT list subtasks separately.
12. Camera Movement (Cinematography):
    - Format: Apply camera motion: [Camera Movement Description]
    - Example: Apply camera motion: orbit down
13. Change Camera Perspective (Note: this is changing the camera's viewpoint, not camera movement):
    - Type 1: First-Third Person Change
        - Format: Switch the camera to a [first/third]-person perspective
    - Others:
        - Format: Move the camera [How the camera moves from the current angle to the desired angle]
        - Example: Move the camera forward and slightly to the left, tilting it upward and rotating to the right for a more dynamic urban perspective.
14. Change the focus of the video:
    - Format: Shift the focus to [describe the subjects to be focused on], making her/him/it sharp. Blur [the objects to be blurred].
15. Other Tasks:
    - Format: Generate logically based on the specific situation while adhering to the Core Generation Rule.

# EXAMPLE OF A HIGH-QUALITY PROMPT
Add a pair of realistic sunglasses to the man centered in the frame: thin matte-black rectangular frame with straight temples and dark neutral-gray mirror lenses (10–15% VLT) that subtly reflect the green foliage and sky. Fit proportionally, browline just above the eyebrows; nose pads rest on the bridge; temple arms sit over the ears and tuck under hair if needed. Match the soft outdoor daylight: add gentle environment reflections on the lenses and soft contact shadows on the nose bridge and upper cheeks where the frame rests. Maintain proper occlusion with hair or hands, crisp anti-aliased edges, no jitter/flicker/warping, no clipping into skin, and do not alter other scene elements or reflect the camera.

# OUTPUT REQUIREMENT
Output ONLY the final enhanced English prompt. Do not include any explanations, greetings, or the category name.
Do not imagine things that do not appear in the video.
For camera movement and camera perspective change cases, only describe the camera transformation in one sentence, without describing anything else.
"""


I2I_TEMPLATE = """Task: Image Editing
# ROLE
You are an expert Image-to-Image (I2I) Prompt Engineer. Your task is to analyze the user's raw editing instruction and the provided source image to generate a detailed I2I editing prompt in English.

# INPUT
- User's raw instruction: "{user_prompt}"
- Context: The source image is provided.

# CORE GENERATION RULE
Unless specified otherwise by the task type, your generated prompt MUST strictly follow this two-part structure:
1. Modifications: Specifically describe what needs to be changed. Include details like physical appearance, spatial location, lighting, shadows, and perspective consistency.
2. Preservations: Explicitly describe the key visual elements, background, or subjects that MUST remain unchanged.
3. Concretization: If the user's instruction contains vague references to characters, objects, outfits, or styles (e.g. "more cartoon characters", "cute toy-like figures", "change outfits", "some animals", "different clothes"), you MUST replace them with specific, well-known, named instances that match the existing visual style of the image. For example, "more cartoon characters" should become named characters like "Hello Kitty, Pikachu, Mickey Mouse"; "change outfits" should become concrete outfit descriptions like "a kung fu training gi, a navy three-piece suit, a black hoodie with cargo pants". Choose instances whose art style, proportions, and tone are consistent with the source image. Never leave generic placeholders in the final prompt.
Note that you don't need to explicitly write "Modifications: xx. Preservations: xx.". Just describe it naturally, for example, "Add an apple. The table and curtains remain unchanged."

# TASK CATEGORIES & TEMPLATES
First, analyze the user's instruction and the source image to determine the specific editing task type. Then, generate the prompt using the corresponding template:

1. Replacement:
   - Format: "Replace [original element] with [new element]."
2. Addition:
   - Format: "Add [element] + [location/placement]."
3. Object/Background Removal:
   - Format: "Delete [object description] + [location]."
4. Text/Watermark Removal:
   - Format: "Remove [text/watermark/logo description] from the image."
5. Depth-to-Image:
   - Format: "Generate image with depth map. [Detailed description of the target image]"
6. Sketch-to-Image:
   - Format: Provide a detailed Text-to-Image (T2I) style description of the desired output.
7. Colorization:
   - Format: "Colorize the image. [Detailed description of the scene and expected colors]"
8. Inpainting:
   - Format: "Inpaint this image. [Detailed description of the region to fill in]"
9. Outpainting / Image Extension:
   - Format: "Extend the image [direction]. [Detailed description of the extended content]"
10. Detection:
    - Format: "Detect the mask region of the [specific object]."
11. Stylization:
    - Format: "Convert the image to [style name]: [brief style details]." Keep it concise.
12. Relighting:
    - Format: "Relight the image: [light direction, color temperature, intensity, and resulting shadow description]."
13. Pose / Expression Change:
    - Format: "Change the [subject]'s [pose/expression] to [target pose/expression description]." Keep identity, clothing, and background consistent unless otherwise specified.
14. Change Viewpoint / Camera Angle (novel view of the same scene):
    - Format: View the scene from [target viewpoint description, e.g. a low-angle front view, a top-down 45-degree view].
    - Example: View the scene from a slight low angle, rotated about 30 degrees to the right of the original viewpoint.
15. Change the focus of the image (depth of field):
    - Format: Shift the focus to [describe the subject to be focused on], making her/him/it sharp. Blur [the objects to be blurred] with a shallow depth-of-field bokeh effect.
16. Mixed Tasks:
    - Format: Seamlessly integrate all requirements into a single, cohesive editing instruction. DO NOT list subtasks separately.
17. Other Tasks:
    - Format: Generate logically based on the specific situation while adhering to the Core Generation Rule.

# EXAMPLE OF A HIGH-QUALITY PROMPT
Add a pair of realistic sunglasses to the man centered in the frame: thin matte-black rectangular frame with straight temples and dark neutral-gray mirror lenses (10-15% VLT) that subtly reflect the green foliage and sky. Fit proportionally, browline just above the eyebrows; nose pads rest on the bridge; temple arms sit over the ears and tuck under hair if needed. Match the soft outdoor daylight: add gentle environment reflections on the lenses and soft contact shadows on the nose bridge and upper cheeks where the frame rests. Maintain proper occlusion with hair or hands, crisp anti-aliased edges, no clipping into skin, and do not alter other scene elements or reflect the camera.

# OUTPUT REQUIREMENT
Output ONLY the final enhanced English prompt. Do not include any explanations, greetings, or the category name.
Do not imagine things that do not appear in the image.
For viewpoint change cases, only describe the viewpoint transformation in one sentence, without describing anything else.
"""


I2V_TEMPLATE = """Task: Image-to-Video Generation
User's prompt: "{user_prompt}"
I'm providing {image_num} reference image(s) used as input frames.

可能是单图 / 多图 I2V 任务，根据图像数量和 prompt 判定，返回英文 prompt：
* 单图 I2V：直接生成英文 prompt 描述视频内容（动作、镜头、场景），参考 T2V prompt 的格式。
* 首尾帧 I2V 任务（2 张图）：返回 "Generate a video based on the first and last frames. " + 视频描述
* 首帧+中间+尾帧的 I2V 任务（3 张图）：返回 "Generate a video based on the first, middle, and last frames. " + 视频描述

只输出最终的英文 prompt，不要其它说明。
"""


VI2V_TEMPLATE = """Task: Video Editing with Reference Image (vi2v)
User's editing instruction: "{user_prompt}"
I'm providing:
1. 3 uniformly sampled frames of the source video
2. {image_num} reference image(s) that should guide the editing

可能是 propagation / reference insertion / reference replacement 任务之一，根据输入的图像和 prompt 判定，返回英文 prompt：
* propagation 任务：直接返回下面这条指令，不要有任何其它内容 — "edit the video following the first frame."
* reference insertion 任务：参考该示例的格式生成 — "Integrate the tree from the image into the video in a reasonable way."
* reference replacement 任务：参考类似格式生成 — 描述用 reference 中的物体替换视频中的对应物体。

只输出最终的英文 prompt，不要其它说明。
"""


ADS2V_TEMPLATE = """Task: Ads Insertion in Video
User's instruction: "{user_prompt}"
I'm providing 3 uniformly sampled frames of the source video for context.

参考下面这条示例的格式生成简洁的英文广告植入指令（一句话即可）：
"Add Starbucks Latte wallpaper on the second floor across the street"

只输出最终的英文 prompt，不要其它说明。
"""


# --------------------------------------------------------------------------- #
# Media helpers
# --------------------------------------------------------------------------- #
def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_uniform_frames(video_path: str, num_frames: int = 1) -> List[str]:
    """Sample `num_frames` frames uniformly and return them as base64 PNGs."""
    import decord

    vr = decord.VideoReader(video_path, num_threads=1, ctx=decord.cpu(0), fault_tol=1)
    total = len(vr)
    if total <= 0:
        return []
    if num_frames == 1:
        indices = [total // 2]
    else:
        indices = [round(i * (total - 1) / (num_frames - 1)) for i in range(num_frames)]
    indices = [max(0, min(i, total - 1)) for i in indices]
    out = []
    for frame in vr.get_batch(indices).asnumpy():
        buf = BytesIO()
        Image.fromarray(frame).save(buf, format="PNG")
        out.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return out


def _img_to_b64(item) -> Optional[str]:
    if item is None:
        return None
    if isinstance(item, tuple):
        item = item[0]
    if isinstance(item, str) and os.path.exists(item):
        return encode_image_to_base64(item)
    if isinstance(item, str):
        return item  # already base64
    return None


def _images_to_b64(images) -> List[str]:
    out = []
    for item in images or []:
        b64 = _img_to_b64(item)
        if b64:
            out.append(b64)
    return out


def _video_frames_to_b64(video, num_frames: int) -> List[str]:
    if not video:
        return []
    paths = video if isinstance(video, list) else [video]
    out = []
    for vp in paths:
        out.extend(extract_uniform_frames(vp, num_frames=num_frames))
    return out


def _build_messages(system_prompt: str, user_text: str, images_b64: List[str]):
    content = [{"type": "text", "text": user_text}]
    for i, b64 in enumerate(images_b64):
        content.append({"type": "text", "text": f"\n[Image {i}]:"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


class PromptEnhancer:
    """Task-aware prompt enhancer backed by an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        max_retries: int = MAX_RETRIES,
        video_frames: int = DEFAULT_VIDEO_FRAMES,
    ):
        from openai import OpenAI

        api_key = api_key or os.environ.get("BERNINI_PE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = base_url or os.environ.get("BERNINI_PE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        self.model = model or os.environ.get("BERNINI_PE_MODEL", DEFAULT_MODEL)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.max_retries = max_retries
        self.video_frames = video_frames

    def _chat(self, system_prompt, user_text, images_b64, json_mode=False) -> Optional[str]:
        messages = _build_messages(system_prompt, user_text, images_b64)
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                kwargs = dict(model=self.model, messages=messages, max_completion_tokens=8192)
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = self.client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                if json_mode:
                    return json.loads(text).get("rewritten_text", text).strip()
                return text.strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning("PE attempt %d/%d failed: %s", attempt, self.max_retries, e)
                time.sleep(min(attempt, 5))
        logger.error("PE failed after %d attempts: %s", self.max_retries, last_err)
        return None

    def __call__(self, task_type, user_prompt, video=None, image=None, images=None) -> Optional[str]:
        """Rewrite `user_prompt` for `task_type`; returns the raw prompt on failure."""
        if not user_prompt or not user_prompt.strip():
            return user_prompt

        video_frames = _video_frames_to_b64(video, self.video_frames)
        ref_imgs = []
        single = _img_to_b64(image)
        if single:
            ref_imgs.append(single)
        ref_imgs += _images_to_b64(images)
        image_num = len(ref_imgs)
        base_sys = SYSTEM_PROMPTS["default"]

        if task_type == "t2v":
            return self._chat(T2V_A14B_EN_SYS_PROMPT, user_prompt, []) or user_prompt
        if task_type == "t2i":
            return self._chat(T2I_A14B_EN_SYS_PROMPT, user_prompt, []) or user_prompt
        if task_type in ("v2v", "mv2v"):
            return self._chat(base_sys, V2V_TEMPLATE.format(user_prompt=user_prompt), video_frames) or user_prompt
        if task_type == "i2i":
            return self._chat(base_sys, I2I_TEMPLATE.format(user_prompt=user_prompt), ref_imgs) or user_prompt
        if task_type == "i2v":
            imgs = ref_imgs if ref_imgs else video_frames[:1]
            text = I2V_TEMPLATE.format(user_prompt=user_prompt, image_num=len(imgs))
            return self._chat(base_sys, text, imgs) or user_prompt
        if task_type == "ads2v":
            return self._chat(base_sys, ADS2V_TEMPLATE.format(user_prompt=user_prompt), video_frames) or user_prompt
        if task_type == "vi2v":
            text = VI2V_TEMPLATE.format(user_prompt=user_prompt, image_num=image_num)
            return self._chat(base_sys, text, video_frames + ref_imgs) or user_prompt
        if task_type == "r2v":
            text = R2V_TEMPLATE.format(image_num=max(image_num, 1), original_text=user_prompt)
            return self._chat(base_sys, text, ref_imgs, json_mode=True) or user_prompt
        if task_type == "r2i":
            text = R2I_TEMPLATE.format(image_num=max(image_num, 1), original_text=user_prompt)
            return self._chat(base_sys, text, ref_imgs, json_mode=True) or user_prompt
        if task_type in ("rv2v", "vrc2v"):
            text = VR2V_TEMPLATE.format(image_num=max(image_num, 1), original_text=user_prompt)
            return self._chat(base_sys, text, video_frames + ref_imgs, json_mode=True) or user_prompt

        logger.warning("unknown task_type=%r; using the raw prompt", task_type)
        return user_prompt
