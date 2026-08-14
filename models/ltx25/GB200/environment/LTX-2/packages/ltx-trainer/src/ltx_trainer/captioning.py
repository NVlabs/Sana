"""
Audio-visual media captioning using multimodal models.
This module provides captioning capabilities for videos with audio using:
- Qwen3-Omni via a local vLLM server (default)
- Gemini Flash 3.5 (cloud API)
Both produce a single combined English caption per video as a single
continuous paragraph of prose.
The Qwen3-Omni backend runs in a separately-launched vLLM server rather than
in-process, so vLLM's heavy CUDA dependencies stay out of this package. The
captioner talks to it over the OpenAI-compatible HTTP API.
Launch the server once (in an isolated environment) with:
.. code-block:: bash
    uv run python scripts/serve_captioner.py
That helper picks BF16 vs FP8 dynamic quantization based on the GPU's free
memory and forwards everything else to ``vllm serve``. To check the recommended
command without running it, pass ``--print-cmd``.
To use Gemini instead, install ``google-genai`` and either set ``GEMINI_API_KEY``
(Gemini Developer API) or have Google Cloud credentials available (gcloud / an
attached service account), in which case it uses Vertex AI automatically.
"""

import json
import os
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import ClassVar

DEFAULT_VIDEO_CAPTION_INSTRUCTION = """\
Analyze this video and produce a single detailed caption covering both its visual content and its audio. Be \
detailed enough that someone reading the caption could form an accurate mental picture of what happens on screen \
and what can be heard. Be exhaustive: include every meaningful detail you can see and hear, including small \
objects, textures, secondary movements, and minor background sounds.

Begin the caption directly with the action or visual detail; do not preface it with phrases like \
"The video opens with...", "The scene shows...", "We see...", or "There is...".

For every shot, include:
- The shot type and framing (extreme wide / wide / medium / medium close-up / close-up / extreme close-up) and any \
camera motion.
- Characters' clothing, appearance, posture, and movement (direction, speed, quality).
- The environment's materials, textures, lighting, and colors.
- All audio: spoken dialogue (quoted exactly in the original language), tone of voice, music (style, mood, \
volume changes), and environmental sounds. If a category is absent -- for example no music is playing, or no one is \
speaking -- state that explicitly. Do not invent specific instruments, music genres, moods, or ambient sounds \
that are not actually present.
- Any on-screen text (signs, titles, labels).

Describe only what is visible or audible. Do not infer emotions, intentions, or anything outside the segment. \
Refer to people descriptively (e.g., "the man in the blue jacket"). Narrate strictly in chronological order; if \
the video contains multiple shots, describe each one in turn.

Write everything as a single continuous paragraph of prose. Do not use section headers, bullet points, or labels \
like "Audio:" / "Visual:" / "Shot:". Integrate visual and audio details naturally within the same sentences.

Return a JSON object with exactly one key:

{"combined_caption_english": "<your caption here>"}"""


DEFAULT_IMAGE_CAPTION_INSTRUCTION = """\
Analyze this image and produce a single detailed caption of its visual content. Be detailed enough that \
someone reading the caption could form an accurate mental picture of the image. Be thorough: include every meaningful \
detail that is actually present, including small objects, textures, and background elements.

Begin the caption directly with the main subject or a visual detail; do not preface it with phrases like \
"The image shows...", "This is a photo of...", "We see...", or "There is...".

Include:
- The framing and composition (close-up / medium / wide / overhead, etc.) and the vantage point.
- The medium or style if distinctive (photograph, illustration, 3D render, painting).
- People's clothing, appearance, and posture, and what they are doing.
- The setting's materials, textures, lighting, and colors.
- Transcribe any visible text verbatim (signs, labels, titles, captions).

Describe only what is visible. Do not infer emotions or intentions, and do not describe sounds, motion, or \
events before or after the moment shown -- this is a single still image. When something is ambiguous, describe \
the visible cue (e.g., "warm low-angle light") rather than guessing the underlying fact (e.g., "sunrise"). \
Refer to people descriptively (e.g., "the man in the blue jacket").

Only describe what is present. Never state that something is absent or missing -- do not write phrases like \
"there is no text", "no people are present", or "no other objects". If a category such as people or text does \
not appear, simply leave it out.

Write everything as a single continuous paragraph of prose. Do not use section headers, bullet points, or \
labels.

Return a JSON object with exactly one key:

{"combined_caption_english": "<your caption here>"}"""


# Default model served by ``scripts/serve_captioner.py``. The captioner does not
# download or load this model itself -- it just sends requests to the vLLM
# server, which already has the model loaded.
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8001/v1"

# Key the combined-caption prompt asks the model to return its caption under.
_CAPTION_JSON_KEY = "combined_caption_english"


class CaptionerType(str, Enum):
    """Enum for different types of media captioners."""

    QWEN_OMNI = "qwen_omni"  # Qwen3-Omni via local vLLM HTTP server
    GEMINI_FLASH = "gemini_flash"  # Gemini Flash 3.5 cloud API


def create_captioner(captioner_type: CaptionerType, **kwargs) -> "MediaCaptioningModel":
    """Factory function to create a media captioner."""
    match captioner_type:
        case CaptionerType.QWEN_OMNI:
            return QwenOmniCaptioner(**kwargs)
        case CaptionerType.GEMINI_FLASH:
            return GeminiFlashCaptioner(**kwargs)
        case _:
            raise ValueError(f"Unsupported captioner type: {captioner_type}")


class MediaCaptioningModel(ABC):
    """Abstract base class for audio-visual media captioning models."""

    instruction: str | None = None

    @abstractmethod
    def caption(self, path: str | Path, **kwargs) -> str:
        """Generate a caption for the given video or image."""

    def _resolve_instruction(self, path: str | Path) -> str:
        """Return the custom instruction, or the image/video default for this input."""
        if self.instruction is not None:
            return self.instruction
        return DEFAULT_IMAGE_CAPTION_INSTRUCTION if self._is_image_file(path) else DEFAULT_VIDEO_CAPTION_INSTRUCTION

    @staticmethod
    def _is_image_file(path: str | Path) -> bool:
        return str(path).lower().endswith((".png", ".jpg", ".jpeg", ".heic", ".heif", ".webp"))

    @staticmethod
    def _is_video_file(path: str | Path) -> bool:
        return str(path).lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))


class QwenOmniCaptioner(MediaCaptioningModel):
    """Audio-visual captioning via a local vLLM server running Qwen3-Omni.
    The vLLM server must already be running. See ``scripts/serve_captioner.py``
    for a helper that launches one in an isolated environment (no impact on
    this package's dependency tree).
    The captioner uses the OpenAI-compatible chat completions API. It sends
    a ``file://`` URL pointing at the local video, the default combined-caption
    prompt, and parses the JSON-wrapped response.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_VLLM_BASE_URL,
        model: str = DEFAULT_QWEN_MODEL,
        api_key: str = "EMPTY",
        instruction: str | None = None,
        max_tokens: int = 4096,
        enable_thinking: bool = False,
        timeout_s: float = 600.0,
    ):
        """Initialize the Qwen3-Omni captioner.
        Args:
            base_url: Base URL of the vLLM OpenAI-compatible server (default
                ``http://127.0.0.1:8001/v1``).
            model: Model identifier the server is serving. Must match the
                server's ``--served-model-name`` (defaults to the HuggingFace
                model ID).
            api_key: Token sent in the ``Authorization`` header. vLLM accepts
                any value by default.
            instruction: Custom instruction prompt. If ``None``, uses the
                default combined-caption prompt.
            max_tokens: Maximum new tokens to generate per caption. 4096 leaves
                comfortable headroom for both ``enable_thinking`` modes.
            enable_thinking: Whether to let the Thinking model produce a
                ``<think>...</think>`` chain-of-thought before the caption.
                Off by default: it makes captioning ~5x slower with little
                quality benefit and occasionally introduces hallucinations
                (e.g., inventing dialogue or background music).
            timeout_s: Per-request HTTP timeout.
        """
        from openai import OpenAI  # noqa: PLC0415

        self.model = model
        self.instruction = instruction
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)

    def caption(
        self,
        path: str | Path,
        fps: int = 2,
    ) -> str:
        """Generate a caption for the given video or image.
        Args:
            path: Path to the video/image file to caption.
            fps: Frames per second to sample from the video. Passed through to
                vLLM's multimodal processor (``mm_processor_kwargs.fps``).
                Default 2 is a typical choice for video MLLMs at this resolution.
                Ignored for image inputs.
        Returns:
            The extracted caption string.
        """
        path = Path(path)
        is_image = self._is_image_file(path)
        is_video = self._is_video_file(path)
        if not (is_image or is_video):
            raise ValueError(f"Unsupported media file: {path}")

        instruction = self._resolve_instruction(path)

        if is_image:
            content = [
                {"type": "image_url", "image_url": {"url": f"file://{path.resolve()}"}},
                {"type": "text", "text": instruction},
            ]
            return _parse_caption_response(self._chat(content)).strip()

        return self._caption_video(path, instruction, fps)

    def _chat(self, content: list[dict], mm_kwargs: dict | None = None) -> str:
        """Send one chat-completions request and return the raw response text."""
        extra_body: dict = {
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        if mm_kwargs:
            extra_body["mm_processor_kwargs"] = mm_kwargs
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=self.max_tokens,
            temperature=0.0,
            extra_body=extra_body,
        )
        return response.choices[0].message.content or ""

    def _caption_video(self, path: Path, instruction: str, fps: int) -> str:
        """Caption a video, sending its audio track as a separate modality.
        vLLM does not extract a video's audio on its own (and its
        ``use_audio_in_video`` path is broken server-side), so we pull the audio
        into a 16 kHz mono WAV and send it alongside the video -- otherwise the
        model only sees frames and fabricates any spoken content.
        """
        with tempfile.TemporaryDirectory(prefix="qwencap_") as tmp:
            work = Path(tmp)

            # Best-effort: ffmpeg fails (and we send video only) if there's no audio.
            audio_url: str | None = None
            try:
                wav = work / "audio.wav"
                _extract_audio_wav(path, wav)
                audio_url = f"file://{wav.resolve()}"
            except subprocess.CalledProcessError:
                pass

            def content(video: Path) -> list[dict]:
                parts: list[dict] = [{"type": "video_url", "video_url": {"url": f"file://{video.resolve()}"}}]
                if audio_url:
                    parts.append({"type": "audio_url", "audio_url": {"url": audio_url}})
                parts.append({"type": "text", "text": instruction})
                return parts

            mm_kwargs = {"fps": fps}
            try:
                raw = self._chat(content(path), mm_kwargs)
            except Exception as e:
                # Raw / variable-frame-rate videos over-report their frame count, which
                # breaks the server's frame sampler ("... frames from video"). Re-encode
                # to a constant frame rate and retry once.
                if "frames from video" not in str(e):
                    raise
                cfr = work / "video_cfr.mp4"
                _transcode_cfr(path, cfr)
                raw = self._chat(content(cfr), mm_kwargs)

            return _parse_caption_response(raw).strip()


class GeminiFlashCaptioner(MediaCaptioningModel):
    """Audio-visual captioning using Google's Gemini via the Google Gen AI SDK.
    Uses the ``google-genai`` package (the current SDK; ``google-generativeai``
    is deprecated). Auth is resolved automatically:
    1. If an API key is given (``api_key`` argument, or ``GEMINI_API_KEY`` /
       ``GOOGLE_API_KEY`` in the environment) -> the Gemini Developer API (AI Studio).
    2. Otherwise, if Google Cloud Application Default Credentials are available
       (an attached service account or ``gcloud auth application-default login``)
       -> Vertex AI. The project comes from ADC (or ``GOOGLE_CLOUD_PROJECT``) and
       the location defaults to ``global`` (override with ``GOOGLE_CLOUD_LOCATION``).
       This means it "just works" on a gcloud-authed GCP VM with no env vars.
    If neither is available, a clear error explains how to authenticate.
    Media is sent inline (``Part.from_bytes``), which works on both backends.
    """

    MODEL_ID = "gemini-3.5-flash"

    _MIME_TYPES: ClassVar[dict[str, str]] = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".heic": "image/heic",
        ".heif": "image/heif",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
    }

    def __init__(
        self,
        api_key: str | None = None,
        instruction: str | None = None,
        model: str | None = None,
    ):
        """Initialize the Gemini captioner.
        Args:
            api_key: Gemini Developer API key. If ``None``, falls back to
                ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``; if no key is set at all,
                uses Vertex AI via Application Default Credentials.
            instruction: Custom instruction prompt. If ``None``, uses the default
                image or video prompt depending on the input.
            model: Override the served model id (defaults to ``MODEL_ID``).
        """
        self.instruction = instruction
        self.model = model or self.MODEL_ID
        self._client = self._make_client(api_key)

    def caption(
        self,
        path: str | Path,
        fps: int = 2,  # noqa: ARG002 - kept for API compatibility
    ) -> str:
        from google.genai import types  # noqa: PLC0415

        path = Path(path)
        instruction = self._resolve_instruction(path)
        media = types.Part.from_bytes(data=path.read_bytes(), mime_type=self._mime_type(path))
        response = self._client.models.generate_content(
            model=self.model,
            contents=[media, instruction],
            config=types.GenerateContentConfig(temperature=0.0),
        )

        # Gemini may also return JSON if it followed our prompt format.
        return _parse_caption_response(response.text or "").strip()

    @classmethod
    def _mime_type(cls, path: Path) -> str:
        try:
            return cls._MIME_TYPES[path.suffix.lower()]
        except KeyError:
            raise ValueError(f"Unsupported media type for Gemini: {path.suffix}") from None

    def _make_client(self, api_key: str | None):  # noqa: ANN202 - genai.Client type is lazy-imported
        from google import genai  # noqa: PLC0415

        # 1. API key (explicit arg or env) -> Gemini Developer API.
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if key:
            return genai.Client(api_key=key)

        # 2. No key -> Vertex AI via Application Default Credentials (gcloud / service account).
        import google.auth  # noqa: PLC0415

        try:
            _, adc_project = google.auth.default()
        except Exception as e:
            raise ValueError(
                "No Gemini credentials found. Provide an API key (--api-key, or "
                "GEMINI_API_KEY / GOOGLE_API_KEY), or set up Google Cloud credentials "
                "for Vertex AI (e.g. `gcloud auth application-default login` or an "
                "attached service account)."
            ) from e

        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or adc_project
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        return genai.Client(vertexai=True, project=project, location=location)


def _parse_caption_response(raw: str) -> str:
    """Extract the caption text from a model response.
    Backend-agnostic: works for any model that follows the combined-caption
    prompt. Handles the formats a model may produce:
    - Plain caption text
    - JSON ``{"combined_caption_english": "..."}``
    - ``<think>...</think>`` chain-of-thought followed by either of the above
    - Truncated JSON (when generation hits a token limit mid-string)
    """
    text = re.sub(r"<think>[\s\S]*?</think>", "", raw).strip()

    # Thinking models (e.g. Qwen3-Omni-*-Thinking) emit the reasoning trace
    # without an opening ``<think>`` tag, because the chat template injects it
    # for them -- so the response starts mid-thought and is terminated by a lone
    # ``</think>`` before the real answer. Drop everything up to that closer.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()

    if not text:
        return raw.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and _CAPTION_JSON_KEY in parsed:
            return parsed[_CAPTION_JSON_KEY]
    except (json.JSONDecodeError, ValueError):
        pass

    match = re.search(rf"\{{[^{{}}]*\"{_CAPTION_JSON_KEY}\"[^{{}}]*\}}", text)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict) and _CAPTION_JSON_KEY in parsed:
                return parsed[_CAPTION_JSON_KEY]
        except (json.JSONDecodeError, ValueError):
            pass

    # Truncated JSON: extract the string value even if the closing quote/brace is missing.
    match = re.search(rf'"{_CAPTION_JSON_KEY}"\s*:\s*"((?:[^"\\]|\\.)*)', text)
    if match:
        try:
            return json.loads('"' + match.group(1) + '"')
        except (json.JSONDecodeError, ValueError):
            return match.group(1)

    return text


def _run_ffmpeg(args: list[str]) -> None:
    """Run the ffmpeg binary bundled with ``imageio-ffmpeg`` (a dependency)."""
    import imageio_ffmpeg  # noqa: PLC0415

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", *args]
    subprocess.run(cmd, check=True, capture_output=True)


def _extract_audio_wav(src: Path, dest: Path) -> None:
    """Extract the audio track to a 16 kHz mono PCM WAV (matches pretraining).
    Raises ``CalledProcessError`` when the video has no audio stream.
    """
    _run_ffmpeg(["-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dest)])


def _transcode_cfr(src: Path, dest: Path) -> None:
    """Re-encode the video to a constant frame rate so the server's frame sampler can
    read every requested index (raw / variable-frame-rate videos over-report frames)."""
    _run_ffmpeg(["-i", str(src), "-fps_mode", "cfr", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(dest)])
