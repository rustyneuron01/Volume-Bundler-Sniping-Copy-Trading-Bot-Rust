import os
import io
import tempfile
from typing import Dict, Any, Optional
from pathlib import Path

import torch
import bittensor as bt
import numpy as np
from PIL import Image
from diffusers.utils import export_to_video
from diffusers import StableDiffusionPipeline, AnimateDiffPipeline, DDIMScheduler, MotionAdapter, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
            
from .base_service import BaseGenerationService
from ..task_manager import GenerationTask


class LocalService(BaseGenerationService):
    """
    Local model service for running open source models.
    
    This demonstrates how to implement a local service that:
    1. Loads and runs models locally
    2. Returns binary data instead of URLs
    3. Handles model loading and GPU management
    4. Can be extended with different model types
    """
    
    def __init__(self, config: Any = None, target_modality: Optional[str] = None):
        super().__init__(config)

        self.image_model = None
        self.video_model = None
        self.models_loaded = False
        self.target_modality = target_modality

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. A CUDA-capable device is required.")

        self.device = 'cuda'
        if self.config and hasattr(self.config, 'device') and self.config.device and self.config.device.startswith("cuda"):
            self.device = self.config.device
        
        self._load_models()

    def _load_models(self):
        """Load local models based on target_modality."""
        bt.logging.info("Loading local generation models...")
        
        load_image = self.target_modality is None or self.target_modality == "image"
        load_video = self.target_modality is None or self.target_modality == "video"
        
        if load_image:
            try:
                self._load_image_model()
            except Exception as e:
                bt.logging.warning(f"Failed to load image model: {e}")
        
        if load_video:
            try:
                self._load_video_model()
            except Exception as e:
                bt.logging.warning(f"Failed to load video model: {e}")
        
        self.models_loaded = True
    
    def _load_image_model(self):
        """Load Stable Diffusion model for image generation with local-first loading."""
        try:
            model_id = "runwayml/stable-diffusion-v1-5"
            bt.logging.info(f"Loading Stable Diffusion model: {model_id}")

            try:
                bt.logging.info(f"Attempting to load {model_id} from local cache...")
                self.image_model = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    local_files_only=True,
                )
            except (OSError, ValueError) as e:
                bt.logging.info(f"Model not in local cache, downloading from HuggingFace...")
                self.image_model = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                )

            if self.device == "cuda":
                self.image_model = self.image_model.to("cuda")

            bt.logging.success("✅ Stable Diffusion model loaded")

        except Exception as e:
            bt.logging.error(f"Failed to load Stable Diffusion: {e}")
            raise

    def _load_video_model(self):
        """Load video generation model."""
        try:
            # AnimateDiff-Lightning setup per official docs
            step = 4  # Options: [1, 2, 4, 8]
            self._ad_lightning_step = step
            repo = "ByteDance/AnimateDiff-Lightning"
            ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
            base_model_id = "emilianJR/epiCRealism"

            bt.logging.info(
                f"Loading AnimateDiff-Lightning video model: base={base_model_id}, step={step}"
            )

            adapter = MotionAdapter().to(self.device, torch.float16)
            try:
                bt.logging.info(f"Attempting to load AnimateDiff adapter from local cache...")
                checkpoint_path = hf_hub_download(repo, ckpt, local_files_only=True)
            except (OSError, ValueError) as e:
                bt.logging.info(f"Adapter not in local cache, downloading from HuggingFace...")
                checkpoint_path = hf_hub_download(repo, ckpt)
            adapter.load_state_dict(load_file(checkpoint_path, device=self.device))

            try:
                bt.logging.info(f"Attempting to load {base_model_id} from local cache...")
                pipe = AnimateDiffPipeline.from_pretrained(
                    base_model_id,
                    motion_adapter=adapter,
                    torch_dtype=torch.float16,
                    local_files_only=True,
                ).to(self.device)
            except (OSError, ValueError) as e:
                bt.logging.info(f"Pipeline not in local cache, downloading from HuggingFace...")
                pipe = AnimateDiffPipeline.from_pretrained(
                    base_model_id,
                    motion_adapter=adapter,
                    torch_dtype=torch.float16,
                ).to(self.device)

            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config,
                timestep_spacing="trailing",
                beta_schedule="linear",
            )

            self.video_model = pipe
            bt.logging.success("✅ AnimateDiff-Lightning video model loaded")

        except Exception as e:
            bt.logging.error(f"Failed to load video model: {e}")
            self.video_model = None
    
    def is_available(self) -> bool:
        """Check if local service is available."""
        available = self.models_loaded and (self.image_model is not None or self.video_model is not None)
        if not available:
            bt.logging.debug(f"LocalService not available: models_loaded={self.models_loaded}, "
                            f"image_model={self.image_model is not None}, "
                            f"video_model={self.video_model is not None}")
        return available
    
    def supports_modality(self, modality: str) -> bool:
        """Check if this service supports the given modality."""
        if modality == "image":
            return self.image_model is not None
        elif modality == "video":
            return self.video_model is not None
        return False
    
    def get_supported_tasks(self) -> Dict[str, list]:
        """Return supported tasks by modality."""
        tasks = {}
        
        if self.image_model is not None:
            tasks["image"] = ["image_generation"]
        else:
            tasks["image"] = []
            
        if self.video_model is not None:
            tasks["video"] = ["video_generation"]
        else:
            tasks["video"] = []
            
        return tasks
    
    def get_api_key_requirements(self) -> Dict[str, str]:
        """Return API key requirements for local service."""
        return {
            "HUGGINGFACE_HUB_TOKEN": "Hugging Face token for model downloads (optional but recommended)"
        }
    
    def process(self, task: GenerationTask) -> Dict[str, Any]:
        """Process a task using local models."""
        if task.modality == "image":
            return self._generate_image_local(task)
        elif task.modality == "video":
            return self._generate_video_local(task)
        else:
            raise ValueError(f"Unsupported modality: {task.modality}")
    
    def _generate_image_local(self, task: GenerationTask) -> Dict[str, Any]:
        """Generate an image using local Stable Diffusion."""
        try:
            bt.logging.info(f"Generating image locally: {task.prompt[:50]}...")
            
            # Check if image model is loaded
            if self.image_model is None:
                raise RuntimeError("Image model is not loaded. Cannot generate images.")
            
            # Ensure parameters is not None
            params = task.parameters or {}
            
            # Extract parameters
            width = params.get("width", 512)
            height = params.get("height", 512)
            num_inference_steps = params.get("steps", 20)
            guidance_scale = params.get("guidance_scale", 7.5)
            
            # Generate image
            bt.logging.debug("Running Stable Diffusion inference...")
            image = self.image_model(
                prompt=task.prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
            
            # Convert to bytes
            img_bytes = self._pil_to_bytes(image)
            
            bt.logging.success("Image generated successfully with local model")
            
            return {
                "data": img_bytes,
                "metadata": {
                    "model": "stable-diffusion-v1-5",
                    "width": width,
                    "height": height,
                    "steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "provider": "local"
                }
            }
            
        except Exception as e:
            bt.logging.error(f"Local image generation failed: {e}")
            raise
    
    def _generate_video_local(self, task: GenerationTask) -> Dict[str, Any]:
        """Generate a video using local AnimateDiff model.""" 
        try:
            bt.logging.info(f"Generating video locally: {task.prompt[:50]}...")
            
            if self.video_model is None:
                raise ValueError("Video model not loaded")
            
            # Ensure parameters is not None
            params = task.parameters or {}
            
            height = params.get("height", 512)
            width = params.get("width", 512) 
            num_frames = params.get("num_frames", 16)
            # AnimateDiff-Lightning expects low guidance and specific step count
            guidance_scale = params.get("guidance_scale", 1.0)
            default_steps = getattr(self, "_ad_lightning_step", 4)
            num_inference_steps = params.get("num_inference_steps", default_steps)
            fps = params.get("fps", 15)
            
            negative_prompt = params.get(
                "negative_prompt",
                "low quality, blurry, distorted, watermark"
            )
            
            # Generate video
            bt.logging.debug("Running AnimateDiff video generation...")
            output = self.video_model(
                prompt=task.prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
            
            # Extract frames from output
            frames = output.frames[0]
            bt.logging.info(f"Generated {len(frames)} frames")
            
            # Convert video frames to bytes
            video_bytes = self._frames_to_video_bytes(frames, fps=fps)
            
            bt.logging.success(f"Video generated successfully with AnimateDiff model: {len(video_bytes)} bytes")
            
            return {
                "data": video_bytes,
                "metadata": {
                    "model": "AnimateDiff",
                    "width": width,
                    "height": height,
                    "num_frames": num_frames,
                    "fps": fps,
                    "guidance_scale": guidance_scale,
                    "num_inference_steps": num_inference_steps,
                    "provider": "local"
                }
            }
            
        except Exception as e:
            bt.logging.error(f"Local video generation failed: {e}")
            raise
    
    
    def _frames_to_video_bytes(self, frames, fps: int = 16) -> bytes:
        """Convert video frames to MP4 bytes"""
        if not frames:
            raise ValueError("No frames provided for video export")
        
        try:            
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            try:
                export_to_video(frames, temp_path, fps=fps)
                with open(temp_path, 'rb') as f:
                    video_bytes = f.read()
                
                bt.logging.info(f"Exported video: {len(video_bytes)} bytes")
                return video_bytes
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            bt.logging.error(f"Failed to convert frames to video bytes: {e}")
            raise
    
    def _pil_to_bytes(self, image: Image.Image, format: str = "PNG") -> bytes:
        """Convert PIL Image to bytes."""
        img_buffer = io.BytesIO()
        image.save(img_buffer, format=format)
        return img_buffer.getvalue()
