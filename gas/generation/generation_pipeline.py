import gc
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional, Any, Union, List
import traceback

import bittensor as bt
import numpy as np
import torch
from PIL import Image
import cv2

from gas.types import ModelTask, Modality
from gas.generation.util.image import create_random_mask, is_black_output
from gas.generation.util.prompt import truncate_prompt_if_too_long
from gas.generation.model_registry import ModelRegistry
from gas.generation.models import initialize_model_registry
from gas.generation.util.model import (
    create_pipeline_generator,
    enable_model_optimizations,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


class GenerationPipeline:

    def __init__(
        self,
        model_registry: Optional[ModelRegistry] = None,
        device: str = "cuda",
    ) -> None:
        """
        Args:
            model_name: Name of the generative image/video model
            model_registry: Optional ModelRegistry instance
            device: Device identifier

        Raises:
            ValueError: If an invalid model name or configuration is provided
        """
        self.model_registry = model_registry or initialize_model_registry()
        self.device = device

    def generate_media(
        self,
        prompt: str,
        model_names: Optional[Union[str, List[str]]] = None,
        image_sample: Optional[dict] = None,
        tasks: Optional[Union[str, List[str]]] = None,
    ):
        """
        Generate synthetic data based on a text prompt.

        Args:
            prompt: The text prompt used for generation, or a dictionary with
                the outer key of generation task type, inner key of the image index,
                and value of the prompt
            task: The generation task type ('t2i', 't2v', 'i2i', or None).
            model_name: Optional model name to use for generation.
            image: Optional image, required for image-to-image generation.

        Yields:
            Dict containing generated media and metadata
        """
        model_names = self._validate_model_names(model_names, tasks)

        n_models = len(model_names)

        for model_idx, model_name in enumerate(model_names):
            modality = self.model_registry.get_modality(model_name)

            bt.logging.info(
                f"Starting Generation | Model {model_idx+1}/{n_models}: {model_name}"
            )

            try:
                image = image_sample.get("image") if image_sample is not None else None
                gen_output = self._generate_media_with_model(model_name, prompt, image)
                if is_black_output(modality.value, gen_output):
                    bt.logging.warning(
                        f"Model {model_name} generated a black/empty output from prompt: {prompt}"
                    )
                    continue

                yield gen_output

            except Exception as e:
                bt.logging.error(f"Failed to generate media: {e}")
                bt.logging.error(f"  Model: {model_name}")
                bt.logging.error(f"  Prompt: {prompt}")
                bt.logging.error(traceback.format_exc())

    def _load_model(
        self,
        model_name: Optional[str] = None,
    ) -> None:
        bt.logging.info(f"Loading {model_name}")
        try:
            self.clear_gpu()
            model_config = self.model_registry.get_model_dict(model_name)
            bt.logging.info(
                json.dumps({k: str(v) for k, v in model_config.items()}, indent=2)
            )

            pipeline_cls = model_config["pipeline_cls"]
            pipeline_args = model_config.get("from_pretrained_args", {}).copy()

            # Handle custom loading functions passed as tuples
            for k, v in pipeline_args.items():
                if isinstance(v, tuple) and callable(v[0]):
                    pipeline_args[k] = v[0](**v[1])

            model_id = pipeline_args.pop("model_id", model_name)
            
            hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

            if isinstance(pipeline_cls, dict):
                # Multi-stage pipeline
                MODEL = {}
                for stage_name, stage_cls in pipeline_cls.items():
                    stage_args = pipeline_args.get(stage_name, {})
                    base_model = stage_args.get("base", model_id)
                    stage_args_filtered = {
                        k: v for k, v in stage_args.items() if k != "base"
                    }

                    bt.logging.debug(f"Loading {stage_name} from {base_model}")
                    # Try local-first loading for multi-stage pipelines
                    try:
                        bt.logging.info(f"Attempting to load {stage_name} from local cache...")
                        MODEL[stage_name] = stage_cls.from_pretrained(
                            base_model,
                            **stage_args_filtered,
                            add_watermarker=False,
                            local_files_only=True,
                            token=hf_token,
                        )
                    except (OSError, ValueError) as e:
                        bt.logging.info(f"Stage {stage_name} not in local cache, downloading...")
                        MODEL[stage_name] = stage_cls.from_pretrained(
                            base_model,
                            **stage_args_filtered,
                            add_watermarker=False,
                            token=hf_token,
                        )

                    enable_model_optimizations(
                        model=MODEL[stage_name],
                        device=self.device,
                        enable_cpu_offload=model_config.get(
                            "enable_model_cpu_offload", False
                        ),
                        enable_sequential_cpu_offload=model_config.get(
                            "enable_sequential_cpu_offload", False
                        ),
                        enable_vae_slicing=model_config.get(
                            "vae_enable_slicing", False
                        ),
                        enable_vae_tiling=model_config.get("vae_enable_tiling", False),
                        stage_name=stage_name,
                    )

                    MODEL[stage_name].watermarker = None
            else:
                # Single-stage pipeline with local-first loading
                try:
                    # Try loading from local cache first (no HF API calls)
                    bt.logging.info(f"Attempting to load {model_id} pipeline from local cache...")
                    MODEL = pipeline_cls.from_pretrained(
                        model_id,
                        **pipeline_args,
                        add_watermarker=False,
                        local_files_only=True,
                        token=hf_token,
                    )
                except (OSError, ValueError) as e:
                    # Model not in cache, download from HuggingFace
                    bt.logging.info(f"Pipeline not in local cache, downloading from HuggingFace...")
                    MODEL = pipeline_cls.from_pretrained(
                        model_id,
                        **pipeline_args,
                        add_watermarker=False,
                        token=hf_token,
                    )

                # Load LoRA weights if specified
                if "lora_model_id" in model_config:
                    bt.logging.info(
                        f"Loading LoRA weights from {model_config['lora_model_id']}"
                    )
                    lora_loading_args = model_config.get("lora_loading_args", {})
                    MODEL.load_lora_weights(
                        model_config["lora_model_id"], token=hf_token, **lora_loading_args
                    )

                # Load scheduler if specified
                scheduler_config = model_config.get("scheduler", {})
                if scheduler_config:
                    sched_cls = scheduler_config["cls"]
                    sched_args = scheduler_config.get("from_config_args", {})
                    MODEL.scheduler = sched_cls.from_config(
                        MODEL.scheduler.config, **sched_args
                    )

                enable_model_optimizations(
                    model=MODEL,
                    device=self.device,
                    enable_cpu_offload=model_config.get(
                        "enable_model_cpu_offload", False
                    ),
                    enable_sequential_cpu_offload=model_config.get(
                        "enable_sequential_cpu_offload", False
                    ),
                    enable_vae_slicing=model_config.get("vae_enable_slicing", False),
                    enable_vae_tiling=model_config.get("vae_enable_tiling", False),
                )
                MODEL.watermarker = None

            self.model = MODEL
            bt.logging.info(f"Loaded {model_name}")
            return True

        except Exception as e:
            bt.logging.error(f"Error loading model: {model_name}")
            bt.logging.error(traceback.format_exc())
            return False

    def _generate_media_with_model(self, model_name, prompt, image):
        model_config = self.model_registry.get_model_dict(model_name)
        task = self.model_registry.get_task(model_name)

        if task == "i2i" and image is None:
            raise ValueError(
                "An image must be provided for image-to-image model {model_name}"
            )

        if not self._load_model(model_name):
            raise RuntimeError(f"Failed to load {model_name}")
            return {}

        bt.logging.debug("Preparing generation arguments")
        gen_args = model_config.get("generation_args", {}).copy()

        # prep inptask-specific generation args
        if task == "i2i":
            if image is None:
                raise ValueError("image cannot be None for image-to-image generation")
            image = Image.fromarray(image)
            target_size = (1024, 1024)
            if image.size[0] > target_size[0] or image.size[1] > target_size[1]:
                image = image.resize(target_size, Image.Resampling.LANCZOS)

            gen_args["mask_image"] = create_random_mask(image.size)
            gen_args["image"] = image

        elif task == "i2v":
            if image is None:
                raise ValueError("image cannot be None for image-to-video generation")
            image = Image.fromarray(image)
            # Get target size from gen_args if specified, otherwise use default
            target_size = (gen_args.get("height", 768), gen_args.get("width", 768))
            if image.size[0] > target_size[0] or image.size[1] > target_size[1]:
                image = image.resize(target_size, Image.Resampling.LANCZOS)
            gen_args["image"] = image

        # Prepare generation arguments
        for k, v in gen_args.items():
            if isinstance(v, dict):
                if "min" in v and "max" in v:
                    # For i2v, use minimum values to save memory
                    if task == "i2v":
                        gen_args[k] = v["min"]
                    else:
                        gen_args[k] = np.random.randint(v["min"], v["max"])

                if "options" in v:
                    gen_args[k] = random.choice(v["options"])

        if "resolution" in gen_args:
            gen_args["height"] = gen_args["resolution"][0]
            gen_args["width"] = gen_args["resolution"][1]
            del gen_args["resolution"]

        truncated_prompt = truncate_prompt_if_too_long(prompt, self.model)
        bt.logging.debug(f"Generating media from prompt: {truncated_prompt}")
        bt.logging.debug(f"Generation args: {gen_args}")

        generate_fn = create_pipeline_generator(model_config, self.model)

        torch.cuda.empty_cache()
        gc.collect()

        start_time = time.time()

        bt.logging.debug("Generating media")

        if model_config.get("use_autocast", False):
            pretrained_args = model_config.get("from_pretrained_args", {})
            torch_dtype = pretrained_args.get("torch_dtype", torch.bfloat16)

            with torch.autocast(self.device, torch_dtype, cache_enabled=False):
                gen_output = generate_fn(truncated_prompt, **gen_args)
        else:
            gen_output = generate_fn(truncated_prompt, **gen_args)

        gen_time = time.time() - start_time

        hours = int(gen_time // 3600)
        minutes = int((gen_time % 3600) // 60)
        seconds = int(gen_time % 60)
        bt.logging.info(
            f"Finished generation in {hours:02d}:{minutes:02d}:{seconds:02d}"
        )

        modality = self.model_registry.get_modality(model_name)
        media_type = self.model_registry.get_output_media_type(model_name)

        output = {
            modality.value: gen_output,  # image or video
            "modality": modality,
            "media_type": media_type,
            "prompt": truncated_prompt,
            "model_name": model_name,
            "time": time.time(),
            "gen_duration": gen_time,
            "gen_args": {
                k: v 
                for k, v in gen_args.items()
                if isinstance(v, (str, int, float, bool, list, dict))
            }
        }

        source_image = gen_args.get("image", None)
        if source_image is not None:
            output["source_image"] = source_image

        mask_image = gen_args.get("mask_image", None)
        if mask_image is not None and modality == Modality.IMAGE:
            # Get generated image from gen_output
            generated_img = None
            if hasattr(gen_output, "images") and gen_output.images:
                generated_img = gen_output.images[0]
            elif isinstance(gen_output, Image.Image):
                generated_img = gen_output
            if generated_img is not None:
                if isinstance(generated_img, Image.Image):
                    gen_img_np = np.array(generated_img)
                else:
                    gen_img_np = generated_img
                if isinstance(mask_image, Image.Image):
                    mask_np = np.array(mask_image)
                else:
                    mask_np = mask_image
                # Resize mask to size of generated image
                mask_resized = cv2.resize(
                    mask_np,
                    (gen_img_np.shape[1], gen_img_np.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                # Ensure mask_resized is uint8 and single-channel
                mask_image = Image.fromarray(mask_resized.astype(np.uint8), mode="L")
                output["mask_image"] = mask_image
            else:
                output["mask_image"] = mask_image
        elif mask_image is not None:
            output["mask_image"] = mask_image

        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        return output

    def _validate_model_names(self, model_names, tasks) -> str:
        if model_names is None:
            if tasks is None:
                model_names = self.model_registry.get_interleaved_model_names()
            else:
                tasks = [tasks] if not isinstance(tasks, (list, tuple)) else tasks
                model_names = self.model_registry.get_model_names_by_task(tasks)

        elif isinstance(model_names, str):
            model_names = [model_names]

        invalid_models = [
            name for name in model_names if name not in self.model_registry.model_names
        ]
        if invalid_models:
            raise ValueError(
                f"Invalid model names {invalid_models}. "
                f"Options are {self.model_registry.model_names}"
            )
        return model_names

    def clear_gpu(self):
        if hasattr(self, "model") and self.model is not None:
            bt.logging.trace("Deleting model")
            if isinstance(self.model, dict):
                for stage_name, stage_model in self.model.items():
                    del stage_model
            else:
                del self.model

        gc.collect()
        if torch.cuda.is_available():
            bt.logging.trace("Clearing CUDA cache")
            torch.cuda.empty_cache()

    def shutdown(self):
        self.clear_gpu()
