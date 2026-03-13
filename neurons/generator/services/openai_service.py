import os
import time
from typing import Dict, Any, Optional

import bittensor as bt
import requests

from .base_service import BaseGenerationService
from ..task_manager import GenerationTask


class OpenAIService(BaseGenerationService):
    """
    OpenAI API service for media generation.

    - Images via DALL-E 3 (with C2PA content credentials preserved by
      downloading the image bytes).
    - Video via Sora 2 (sora-2 / sora-2-pro), returning MP4 bytes.
    """

    def __init__(self, config: Any = None):
        super().__init__(config)
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = None

        if self.api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                bt.logging.info("OpenAI client initialized")
            except ImportError:
                bt.logging.warning("OpenAI package not installed. Run: pip install openai")
            except Exception as e:
                bt.logging.error(f"Failed to initialize OpenAI client: {e}")

    def is_available(self) -> bool:
        """Check if OpenAI service is available."""
        return (
            self.api_key is not None
            and self.api_key.strip() != ""
            and self.client is not None
        )

    def supports_modality(self, modality: str) -> bool:
        """Check if this service supports the given modality."""
        # Now supports both images (DALL-E) and video (Sora)
        return modality in {"image", "video"}

    def get_supported_tasks(self) -> Dict[str, list]:
        """Return supported tasks by modality."""
        return {
            "image": ["image_generation"],
            "video": ["video_generation"],
        }

    def get_api_key_requirements(self) -> Dict[str, str]:
        """Return OpenAI API key requirements."""
        return {
            "OPENAI_API_KEY": "OpenAI API key for DALL-E image and Sora video generation"
        }

    def process(self, task: GenerationTask) -> Dict[str, Any]:
        """Dispatch task by modality."""
        if task.modality == "image":
            return self._generate_image(task)
        elif task.modality == "video":
            return self._generate_video(task)
        else:
            raise ValueError(f"Unsupported modality: {task.modality}")


    def _generate_image(self, task: GenerationTask) -> Dict[str, Any]:
        """Generate an image using DALL-E and download the bytes with C2PA metadata."""
        try:
            bt.logging.info(f"Generating image with DALL-E: {task.prompt[:50]}...")
            params = task.parameters or {}
            width = params.get("width", 1024)
            height = params.get("height", 1024)
            quality = params.get("quality", "standard")
    
            size = self._get_valid_image_size(width, height)

            response = self.client.images.generate(
                model="dall-e-3",
                prompt=task.prompt,
                size=size,
                quality=quality,
                n=1,
            )

            image_url = response.data[0].url
            bt.logging.info("DALL-E generated image, downloading from URL...")

            img_response = requests.get(image_url, timeout=60)
            img_response.raise_for_status()
            image_data = img_response.content

            bt.logging.success(f"Downloaded {len(image_data)} bytes with C2PA metadata")

            return {
                "data": image_data,
                "metadata": {
                    "model": "dall-e-3",
                    "size": size,
                    "quality": quality,
                    "provider": "openai",
                    "mime_type": "image/png",  # DALL-E 3 outputs PNG
                },
            }

        except Exception as e:
            bt.logging.error(f"OpenAI image generation failed: {e}")
            raise

    def _get_valid_image_size(self, width: int, height: int) -> str:
        """Convert width/height to valid DALL-E size."""
        # DALL-E 3 supports: 1024x1024, 1024x1792, 1792x1024
        if width == height:
            return "1024x1024"
        elif width > height:
            return "1792x1024"
        else:
            return "1024x1792"

    def _generate_video(self, task: GenerationTask) -> Dict[str, Any]:
        """
        Generate a video using Sora 2 and return the raw MP4 bytes.

        Adds retry logic for transient Sora/internal errors:
        - Creates a fresh job per attempt
        - Polls until completion or failure/timeout
        - Retries on specific error codes with exponential backoff
        """
        if self.client is None:
            raise RuntimeError("OpenAI client not initialized")

        params: Dict[str, Any] = task.parameters or {}

        model: str = params.get("model", "sora-2")  # or "sora-2-pro"
        size: str = params.get("size", "720x1280")  # e.g. "1280x720"
        seconds_value = params.get("seconds", params.get("duration", 4))
        seconds: str = str(seconds_value)

        poll_interval: float = float(params.get("poll_interval", 5.0))
        timeout: float = float(params.get("timeout", 600.0))  # per attempt

        max_retries: int = int(params.get("max_retries", 2))
        backoff_base: float = float(params.get("backoff_base", 5.0))  # seconds

        bt.logging.info(
            f"Sora video generation requested: "
            f"model={model}, size={size}, seconds={seconds}, "
            f"max_retries={max_retries}, timeout={timeout}"
        )

        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            attempt_num = attempt + 1
            bt.logging.info(
                f"Starting Sora attempt {attempt_num}/{max_retries + 1} "
                f"for prompt={task.prompt[:60]!r}"
            )

            try:
                # 1) Create the video job
                job = self.client.videos.create(
                    model=model,
                    prompt=task.prompt,
                    size=size,
                    seconds=seconds,
                )

                video_id = getattr(job, "id", None)
                status = getattr(job, "status", None)

                if not video_id:
                    raise RuntimeError(f"Sora job did not return an id: {job}")

                bt.logging.info(
                    f"Sora job created: id={video_id}, status={status}, "
                    f"attempt={attempt_num}"
                )

                # 2) Poll until completion or timeout for THIS job
                start_time = time.monotonic()

                while status not in ("completed", "failed", "cancelled"):
                    elapsed = time.monotonic() - start_time
                    if elapsed > timeout:
                        raise TimeoutError(
                            f"Sora video generation timed out after {timeout} seconds "
                            f"(status={status}, id={video_id}, attempt={attempt_num})"
                        )

                    progress = getattr(job, "progress", None)
                    bt.logging.info(
                        f"Sora job {video_id}: status={status}, "
                        f"progress={progress}%, elapsed={elapsed:.1f}s, "
                        f"attempt={attempt_num}"
                    )

                    time.sleep(poll_interval)
                    job = self.client.videos.retrieve(video_id)
                    status = getattr(job, "status", None)

                # 3) Handle terminal status
                if status == "completed":
                    bt.logging.success(
                        f"Sora job {video_id} completed on attempt {attempt_num}, "
                        f"downloading content..."
                    )
                    stream = self.client.videos.download_content(video_id=video_id)
                    video_bytes = stream.read()

                    bt.logging.success(
                        f"Downloaded {len(video_bytes)} bytes for "
                        f"Sora video id={video_id}"
                    )

                    return {
                        "data": video_bytes,
                        "metadata": {
                            "model": model,
                            "size": size,
                            "seconds": seconds,
                            "provider": "openai",
                            "mime_type": "video/mp4",
                            "video_id": video_id,
                            "status": status,
                            "progress": getattr(job, "progress", None),
                            "created_at": getattr(job, "created_at", None),
                            "completed_at": getattr(job, "completed_at", None),
                            "attempt": attempt_num,
                        },
                    }

                # failed / cancelled
                error_obj = getattr(job, "error", None)
                error_code = getattr(error_obj, "code", None) or getattr(
                    error_obj, "type", None
                )
                error_msg = getattr(error_obj, "message", None) or str(error_obj)

                bt.logging.error(
                    f"Sora job {video_id} finished with status={status}, "
                    f"error_code={error_code}, error_msg={error_msg}, "
                    f"attempt={attempt_num}"
                )

                # Decide if we should retry a NEW job
                if self._is_retryable_sora_error(error_code, status):
                    last_error = RuntimeError(
                        f"Sora video generation failed (transient): "
                        f"status={status}, code={error_code}, msg={error_msg}"
                    )
                    if attempt < max_retries:
                        delay = backoff_base * (2 ** attempt)
                        bt.logging.warning(
                            f"Retryable Sora error (attempt {attempt_num}). "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        continue
                    else:
                        bt.logging.error(
                            "Max retries reached for Sora video generation; "
                            "giving up on this task."
                        )
                        raise last_error
                else:
                    # Non-retryable failure: bail out immediately
                    raise RuntimeError(
                        f"Sora video generation failed (non-retryable): "
                        f"status={status}, code={error_code}, msg={error_msg}"
                    )

            except Exception as e:
                # Catch any exception in this attempt and decide whether to retry
                last_error = e
                # Try to extract an error_code if this is an OpenAI error-like obj
                error_code = getattr(e, "code", None)
                if self._is_retryable_sora_error(error_code, status=None):
                    if attempt < max_retries:
                        delay = backoff_base * (2 ** attempt)
                        bt.logging.warning(
                            f"Sora attempt {attempt_num} raised {e!r} "
                            f"(code={error_code}); retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        continue
                    else:
                        bt.logging.error(
                            f"Max retries reached after exception in Sora video "
                            f"generation: {e!r}"
                        )
                        break
                else:
                    bt.logging.error(
                        f"Sora video generation failed with non-retryable exception "
                        f"on attempt {attempt_num}: {e!r}"
                    )
                    break

        # If we get here, all attempts failed
        bt.logging.error(
            f"Sora video generation exhausted {max_retries + 1} attempts "
            f"for prompt={task.prompt[:60]!r}"
        )
        if last_error:
            raise last_error
        else:
            raise RuntimeError("Sora video generation failed for unknown reasons.")

    @staticmethod
    def _is_retryable_sora_error(
        error_code: Optional[str],
        status: Optional[str],
    ) -> bool:
        """
        Heuristic: which Sora errors are safe to retry with a fresh job?

        We treat typical transient/backend errors as retryable:
        - "video_generation_failed"  (generic internal failure, like you saw)
        - "server_error"
        - "timeout"
        - "rate_limit_exceeded"
        - "overloaded"
        """
        if status == "cancelled":
            # Usually user cancellation; don't retry automatically
            return False

        if not error_code:
            return False

        error_code = str(error_code)
        retryable_codes = {
            "video_generation_failed",  # internal error (your case)
            "server_error",
            "timeout",
            "rate_limit_exceeded",
            "overloaded",
        }
        return error_code in retryable_codes