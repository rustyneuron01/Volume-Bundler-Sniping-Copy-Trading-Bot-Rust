import os
import io
import time
import requests
import bittensor as bt
from typing import Dict, Any, Optional

from PIL import Image
import c2pa

from .base_service import BaseGenerationService
from ..task_manager import GenerationTask


class Models:
    """Symbolic constants for StabilityAI models."""

    SD35_MEDIUM = "sd3.5-medium"
    SD35_LARGE = "sd3.5-large"
    ULTRA = "ultra"
    CORE = "core"


MODEL_INFO = {
    Models.SD35_MEDIUM: {
        "endpoint": "https://api.stability.ai/v2beta/stable-image/generate/sd3",
        "name": "Stable Diffusion 3.5 Medium",
        "family": "sd3.5",
        "description": "Balanced high-quality diffusion model",
    },
    Models.SD35_LARGE: {
        "endpoint": "https://api.stability.ai/v2beta/stable-image/generate/sd3",
        "name": "Stable Diffusion 3.5 Large",
        "family": "sd3.5",
        "description": "Higher-quality large model",
    },
    Models.ULTRA: {
        "endpoint": "https://api.stability.ai/v2beta/stable-image/generate/ultra",
        "name": "Stability Ultra",
        "family": "ultra",
        "description": "Highest-quality proprietary model",
    },
    Models.CORE: {
        "endpoint": "https://api.stability.ai/v2beta/stable-image/generate/core",
        "name": "Stability Core",
        "family": "core",
        "description": "Fast, efficient core model",
    },
}

class StabilityAIService(BaseGenerationService):
    """
    Stability AI generation service for Ultra, Core, and SD 3.5 family models.

    Features:
    - Binary image generation
    - Embedded C2PA manifest extraction
    - Full MIME-type automatic detection
    - Model → endpoint mapping
    """

    # Allowed output formats
    VALID_FORMATS = {"png", "jpeg", "webp"}

    def __init__(self, config: Any = None):
        super().__init__(config)

        self.api_key = os.getenv("STABILITY_API_KEY")
        self.timeout = 90
        self.default_model = Models.SD35_MEDIUM

        if not self.api_key:
            bt.logging.warning("STABILITY_API_KEY not found.")
        else:
            bt.logging.info("StabilityAIService initialized with API key")

    # ---------------------------------------------------------------------
    # Base methods
    # ---------------------------------------------------------------------
    def is_available(self) -> bool:
        return self.api_key is not None and self.api_key.strip() != ""

    def supports_modality(self, modality: str) -> bool:
        return modality == "image"

    def get_supported_tasks(self) -> Dict[str, list]:
        return {
            "image": ["image_generation"],
            "video": []  # StabilityAI doesn't support video generation yet
        }

    def get_api_key_requirements(self) -> Dict[str, str]:
        return {"STABILITY_API_KEY": "API key for StabilityAI image generation"}

    # ---------------------------------------------------------------------
    # Processing logic
    # ---------------------------------------------------------------------
    def process(self, task: GenerationTask) -> Dict[str, Any]:
        if task.modality != "image":
            raise ValueError(f"StabilityAIService does not support modality: {task.modality}")

        return self._generate_image(task)

    def _check_c2pa(self, img_bytes: bytes, output_format: str) -> bool:
        """Check if image has embedded C2PA credentials and log result."""
        mime_map = {
            "png": "image/png",
            "jpeg": "image/jpeg",
            "jpg": "image/jpeg",
            "webp": "image/webp",
        }
        mime_type = mime_map.get(output_format.lower(), "application/octet-stream")

        try:
            with io.BytesIO(img_bytes) as f:
                with c2pa.Reader(mime_type, f) as reader:
                    manifest = reader.json()
                    if manifest:
                        bt.logging.debug("StabilityAI image has embedded C2PA credentials")
                        return True
        except Exception:
            pass
        return False

    def _get_endpoint(self, model: str) -> str:
        if model not in MODEL_INFO:
            raise ValueError(f"Unknown StabilityAI model: {model}")
        return MODEL_INFO[model]["endpoint"]

    # ---------------------------------------------------------------------
    # Image generation core
    # ---------------------------------------------------------------------
    def _generate_image(self, task: GenerationTask) -> Dict[str, Any]:
        try:
            params = task.parameters or {}

            model = params.get("model", self.default_model)
            prompt = task.prompt

            bt.logging.info(f"StabilityAI generating image with model={model}")

            if model not in MODEL_INFO:
                raise ValueError(f"Unknown StabilityAI model: {model}. "
                                f"Available models: {list(MODEL_INFO.keys())}")

            url = self._get_endpoint(model)

            output_format = params.get("format", "png")
            if output_format not in self.VALID_FORMATS:
                output_format = "png"

            api_data = {
                "prompt": prompt,
                "output_format": output_format,
                "model": model,
            }

            # Optional parameters
            if "negative_prompt" in params:
                api_data["negative_prompt"] = params["negative_prompt"]
            if "aspect_ratio" in params:
                api_data["aspect_ratio"] = params["aspect_ratio"]
            if "seed" in params:
                api_data["seed"] = str(params["seed"])

            headers = {
                "authorization": f"Bearer {self.api_key}",
                "accept": "image/*",
            }

            # Multipart requirement
            files = {"none": ""}

            start_time = time.time()
            response = requests.post(
                url,
                headers=headers,
                data=api_data,
                files=files,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                raise RuntimeError(f"Stability API error {response.status_code}: {response.text}")

            img_bytes = response.content
            gen_time = time.time() - start_time

            # Log C2PA detection (credentials are embedded in binary, validator extracts them)
            self._check_c2pa(img_bytes, output_format)

            # Return final miner-compatible result
            return {
                "data": img_bytes,
                "metadata": {
                    "model": model,
                    "provider": "stability.ai",
                    "format": output_format.upper(),
                    "generation_time": gen_time,
                }
            }

        except Exception as e:
            bt.logging.error(f"StabilityAI image generation failed: {e}")
            raise

    def get_service_info(self) -> Dict[str, Any]:
        """Return information about this service."""
        return {
            "name": "StabilityAI",
            "type": "api",
            "provider": "api.stability.ai",
            "available": self.is_available(),
            "supported_tasks": self.get_supported_tasks(),
            "default_model": self.default_model
        }
