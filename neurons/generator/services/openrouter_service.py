import os
import time
import base64
import io
from typing import Dict, Any, Optional

import requests
import bittensor as bt
from PIL import Image

from .base_service import BaseGenerationService
from ..task_manager import GenerationTask


class OpenRouterService(BaseGenerationService):
    """
    OpenRouter API service for image generation using various models.
    
    This service integrates with OpenRouter.ai to provide access to multiple
    image generation models including Google Gemini Flash Image Preview.
    
    Based on the nano_banana implementation pattern but adapted for the
    new modular service architecture.
    """
    
    def __init__(self, config: Any = None):
        super().__init__(config)
        self.api_key = os.getenv('OPEN_ROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Default model from nano_banana
        self.default_model = "google/gemini-3-pro-image-preview"
        
        # Timeout and retry settings
        self.timeout = 60.0
        self.max_retries = 3
        
        if self.api_key and self.api_key.strip():
            bt.logging.info("OpenRouter service initialized with API key")
        else:
            bt.logging.warning(f"OpenRouter API key not found or empty. Value: {repr(self.api_key)}. Set OPEN_ROUTER_API_KEY environment variable.")
    
    def is_available(self) -> bool:
        """Check if OpenRouter service is available."""
        return self.api_key is not None and self.api_key.strip() != ""
    
    def supports_modality(self, modality: str) -> bool:
        """Check if this service supports the given modality."""
        # OpenRouter supports image generation via various models
        return modality == "image"
    
    def get_supported_tasks(self) -> Dict[str, list]:
        """Return supported tasks by modality."""
        return {
            "image": ["image_generation"]
        }
    
    def get_api_key_requirements(self) -> Dict[str, str]:
        """Return OpenRouter API key requirements."""
        return {
            "OPEN_ROUTER_API_KEY": "OpenRouter API key for multi-model image generation"
        }
    
    async def process(self, task: GenerationTask) -> Optional[Dict[str, Any]]:
        """
        Process a generation task using OpenRouter API.
        
        Args:
            task: The generation task to process
            
        Returns:
            Dict with generation results or None on failure
        """
        if not self.is_available():
            bt.logging.error("OpenRouter service not available")
            raise RuntimeError("OpenRouter service not available")
        
        try:
            # Extract task parameters with enhanced logging
            prompt = task.prompt
            modality = task.modality
            parameters = task.parameters or {}
            
            bt.logging.info(f"OpenRouter processing task: modality={modality}")
            bt.logging.debug(f"Task prompt: {prompt[:100]}...")
            bt.logging.debug(f"Task parameters: {parameters}")
            
            # Validate task
            if not self.supports_modality(modality):
                bt.logging.error(f"OpenRouter service doesn't support modality {modality}")
                raise ValueError(f"OpenRouter service doesn't support modality {modality}")
            
            # Get model from parameters or use default
            model = parameters.get('model', self.default_model)
            
            bt.logging.info(f"Generating image with OpenRouter model: {model}")
            
            # Call OpenRouter API
            api_result = await self._generate_image(prompt, model)
            
            if api_result is None:
                bt.logging.error("OpenRouter API returned None")
                raise RuntimeError("OpenRouter API returned None")
            
            bt.logging.debug(f"OpenRouter API result keys: {list(api_result.keys())}")
            
            # Use raw binary data to preserve C2PA metadata from Google/Gemini
            image_data = api_result.get("raw_binary")
            pil_image = api_result.get("image")
            
            if image_data is None:
                bt.logging.error("No raw binary data in OpenRouter API result")
                raise ValueError("No raw binary data in OpenRouter API result")
            
            bt.logging.info(
                f"OpenRouterService returning: {len(image_data)} bytes, "
                f"magic={image_data[:16].hex()}, same_as_raw={image_data is api_result.get('raw_binary')}"
            )
            
            result = {
                "data": image_data,
                "metadata": {
                    "model": model,
                    "provider": "openrouter",
                    "format": api_result.get("format", "unknown"),
                    "generation_time": api_result.get('gen_duration', 0),
                    "width": pil_image.width if pil_image else 0,
                    "height": pil_image.height if pil_image else 0,
                }
            }
            
            bt.logging.success(f"OpenRouter generation completed in {api_result.get('gen_duration', 0):.2f}s")
            return result
            
        except Exception as e:
            bt.logging.error(f"Error in OpenRouter processing: {e}")
            bt.logging.error(f"OpenRouter processing failed for task {task.task_id}: {type(e).__name__}: {e}")
            raise  # Re-raise instead of returning None
    
    async def _generate_image(self, prompt: str, model: str) -> Optional[Dict[str, Any]]:
        """
        Generate an image using OpenRouter API.
        
        Based on the nano_banana.py implementation but adapted for async operation.
        
        Args:
            prompt: The text prompt for image generation
            model: The model to use for generation
            
        Returns:
            Dict with generation results or None on failure
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "modalities": ["image", "text"]
        }
        
        start_time = time.time()
        
        try:
            # Make API request
            response = requests.post(
                self.base_url, 
                headers=headers, 
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                error_text = response.text if response.text else "Unknown error"
                bt.logging.error(f"OpenRouter API error: {response.status_code} - {error_text}")
                raise Exception(f"OpenRouter API returned {response.status_code}: {error_text}")

            # Safely parse JSON response
            try:
                result = response.json()
                if result is None:
                    bt.logging.error("OpenRouter API returned null JSON")
                    raise ValueError("OpenRouter API returned null JSON")
                if not isinstance(result, dict):
                    bt.logging.error(f"OpenRouter API returned non-dict JSON: {type(result)}")
                    raise ValueError(f"OpenRouter API returned non-dict JSON: {type(result)}")
            except ValueError as e:
                bt.logging.error(f"OpenRouter API returned invalid JSON: {e}")
                raise ValueError(f"OpenRouter API returned invalid JSON: {e}")

            # Parse response (following nano_banana pattern)
            if 'choices' not in result or not result['choices']:
                bt.logging.error("OpenRouter response missing choices")
                bt.logging.debug(f"OpenRouter response: {result}")
                raise ValueError("OpenRouter response missing choices")

            choice = result['choices'][0]
            bt.logging.debug(f"OpenRouter choice keys: {list(choice.keys())}")

            if 'message' not in choice or 'images' not in choice['message']:
                bt.logging.error("OpenRouter response missing images in message")
                bt.logging.debug(f"OpenRouter message keys: {list(choice.get('message', {}).keys())}")
                raise ValueError("OpenRouter response missing images in message")

            message = choice['message']
            bt.logging.debug(f"OpenRouter message keys: {list(message.keys())}")

            images = message['images']
            if not images:
                bt.logging.error("OpenRouter response contains no images")
                raise ValueError("OpenRouter response contains no images")

            bt.logging.debug(f"OpenRouter image[0] keys: {list(images[0].keys())}")

            # Process the first image
            image_url = images[0]['image_url']['url']

            # Handle base64 data URL format: "data:image/<fmt>;base64,<data>"
            if ',' not in image_url:
                bt.logging.error("Invalid image URL format from OpenRouter")
                raise ValueError("Invalid image URL format from OpenRouter")

            # Extract format from data URL (e.g., "data:image/png;base64,...")
            header_part = image_url.split(',')[0]
            image_format = "PNG"
            if "image/" in header_part:
                fmt = header_part.split("image/")[1].split(";")[0].upper()
                if fmt in ["PNG", "JPEG", "JPG", "WEBP"]:
                    image_format = fmt

            bt.logging.debug(f"OpenRouter image data URL header: {header_part}")
            bt.logging.debug(f"OpenRouter detected format: {image_format}")
                
            base64_data = image_url.split(',')[1]
            image_binary = base64.b64decode(base64_data)
            
            bt.logging.info(f"OpenRouter raw binary size: {len(image_binary)} bytes")
            bt.logging.debug(f"OpenRouter binary magic bytes: {image_binary[:16].hex()}")
            
            # Open with PIL just to get dimensions, but preserve raw bytes for C2PA
            pil_image = Image.open(io.BytesIO(image_binary))

            gen_time = time.time() - start_time

            output = {
                "image": pil_image,
                "raw_binary": image_binary,  # Preserve original bytes with C2PA
                "format": image_format,
                "modality": "image",
                "media_type": "synthetic",
                "prompt": prompt,
                "model_name": model,
                "time": time.time(),
                "gen_duration": gen_time,
                "gen_args": {
                    "provider": "openrouter",
                    "model": model,
                    "modalities": ["image", "text"],
                    "api_url": self.base_url,
                },
            }

            return output
            
        except requests.exceptions.Timeout:
            bt.logging.error(f"OpenRouter API timeout after {self.timeout}s")
            raise
        except requests.exceptions.RequestException as e:
            bt.logging.error(f"OpenRouter API request failed: {e}")
            raise
        except Exception as e:
            bt.logging.error(f"OpenRouter image processing failed: {e}")
            raise
    
    def get_service_info(self) -> Dict[str, Any]:
        """Return information about this service."""
        return {
            "name": "OpenRouter",
            "type": "api",
            "provider": "openrouter.ai",
            "available": self.is_available(),
            "supported_tasks": self.get_supported_tasks(),
            "default_model": self.default_model,
            "base_url": self.base_url,
            "config": {
                "timeout": self.timeout,
                "max_retries": self.max_retries,
            }
        }
