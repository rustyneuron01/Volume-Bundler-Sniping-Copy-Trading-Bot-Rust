import PIL.Image
import numpy as np
import torch
import bittensor as bt

from diffusers import (
    DiffusionPipeline,
    HunyuanVideoTransformer3DModel,
    MotionAdapter,
    AutoencoderKLWan
)
from huggingface_hub import hf_hub_download
from janus.models import VLChatProcessor
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
from typing import Any, Dict, Optional


def load_autoencoder_kl_wan(
    model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers", 
    subfolder="vae", 
    torch_dtype=torch.float32
):
    """
    Load Wan AutoencoderKL with local-first loading strategy.
    """
    try:
        # Try loading from local cache first (no HF API calls)
        bt.logging.info(f"Attempting to load {model_id} VAE from local cache...")
        return AutoencoderKLWan.from_pretrained(
            model_id, 
            subfolder=subfolder, 
            torch_dtype=torch_dtype,
            local_files_only=True
        )
    except (OSError, ValueError) as e:
        # Model not in cache, download from HuggingFace
        bt.logging.info(f"VAE not in local cache, downloading from HuggingFace...")
        return AutoencoderKLWan.from_pretrained(
            model_id, 
            subfolder=subfolder, 
            torch_dtype=torch_dtype
        )

def load_hunyuanvideo_transformer(
    model_id: str = "tencent/HunyuanVideo",
    subfolder: str = "transformer",
    torch_dtype: torch.dtype = torch.bfloat16,
    revision: str = "refs/pr/18",
):
    """
    Load HunyuanVideo transformer with local-first loading strategy.
    
    Tries to load from local cache first to avoid unnecessary HF API calls.
    Only downloads from HuggingFace if the model is not yet cached locally.
    """
    try:
        # Try loading from local cache first (no HF API calls)
        bt.logging.info(f"Attempting to load {model_id} from local cache...")
        return HunyuanVideoTransformer3DModel.from_pretrained(
            model_id, 
            subfolder=subfolder, 
            torch_dtype=torch_dtype, 
            revision=revision,
            local_files_only=True
        )
    except (OSError, ValueError) as e:
        # Model not in cache, download from HuggingFace
        bt.logging.info(f"Model not in local cache, downloading from HuggingFace...")
        return HunyuanVideoTransformer3DModel.from_pretrained(
            model_id, subfolder=subfolder, torch_dtype=torch_dtype, revision=revision
        )


def load_annimatediff_motion_adapter(step: int = 4) -> MotionAdapter:
    """
    Load a motion adapter model for AnimateDiff with local-first loading strategy.

    Args:
        step: The step size for the motion adapter. Options: [1, 2, 4, 8].
        repo: The HuggingFace repository to download the motion adapter from.
        ckpt: The checkpoint filename
    Returns:
        A loaded MotionAdapter model.

    Raises:
        ValueError: If step is not one of [1, 2, 4, 8].
    """
    if step not in [1, 2, 4, 8]:
        raise ValueError("Step must be one of [1, 2, 4, 8]")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter = MotionAdapter().to(device, torch.float16)

    repo = "ByteDance/AnimateDiff-Lightning"
    ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"

    try:
        bt.logging.info(f"Attempting to load AnimateDiff adapter from local cache...")
        checkpoint_path = hf_hub_download(repo, ckpt, local_files_only=True)
    except (OSError, ValueError) as e:
        bt.logging.info(f"Adapter not in local cache, downloading from HuggingFace...")
        checkpoint_path = hf_hub_download(repo, ckpt)
    
    adapter.load_state_dict(load_file(checkpoint_path, device=device))
    return adapter


class JanusWrapper(DiffusionPipeline):
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor
        self.tokenizer = self.processor.tokenizer
        self.register_modules(
            model=model, processor=processor, tokenizer=self.processor.tokenizer
        )

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        temperature: float = 1.0,
        parallel_size: int = 4,
        cfg_weight: float = 5.0,
        image_token_num_per_image: int = 576,
        img_size: int = 384,
        patch_size: int = 16,
        **kwargs,
    ):
        conversation = [
            {
                "role": "<|User|>",
                "content": prompt,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        sft_format = self.processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + self.processor.image_start_tag

        input_ids = self.processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids).to(self.device)

        tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(
            self.device
        )
        for i in range(parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = self.processor.pad_id

        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)
        generated_tokens = torch.zeros(
            (parallel_size, image_token_num_per_image), dtype=torch.int
        ).to(self.device)
        outputs = None

        for i in range(image_token_num_per_image):
            outputs = self.model.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=outputs.past_key_values if i != 0 else None,
            )
            hidden_states = outputs.last_hidden_state

            logits = self.model.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]

            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat(
                [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
            ).view(-1)
            img_embeds = self.model.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        dec = self.model.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        images = []
        for i in range(parallel_size):
            images.append(PIL.Image.fromarray(dec[i].astype(np.uint8)))

        # Return object with images attribute
        class Output:
            def __init__(self, images):
                self.images = images

        return Output(images)

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        model, processor = load_janus_model(model_path, **kwargs)
        return cls(model=model, processor=processor)

    def to(self, device):
        self.model = self.model.to(device)
        return self


def load_janus_model(model_path: str, **kwargs):
    """
    Load Janus model with local-first loading strategy.
    """
    try:
        bt.logging.info(f"Attempting to load Janus {model_path} from local cache...")
        processor = VLChatProcessor.from_pretrained(model_path, local_files_only=True)
        janus_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": kwargs.get("torch_dtype", torch.bfloat16),
            "local_files_only": True,
        }

        model = AutoModelForCausalLM.from_pretrained(model_path, **janus_kwargs).eval()

    except (OSError, ValueError) as e:
        bt.logging.info(f"Janus model not in local cache, downloading from HuggingFace...")
        processor = VLChatProcessor.from_pretrained(model_path)
        janus_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": kwargs.get("torch_dtype", torch.bfloat16),
        }
        model = AutoModelForCausalLM.from_pretrained(model_path, **janus_kwargs).eval()

    return model, processor


def create_pipeline_generator(model_config: Dict[str, Any], model: Any) -> callable:
    """
    Creates a generator function based on pipeline configuration.

    Args:
        model_config: Model configuration dictionary
        model: Loaded model instance(s)

    Returns:
        Callable that handles the generation process for the model
    """
    if isinstance(model_config.get("pipeline_stages"), list):

        def generate(prompt: str, **kwargs):
            output = None
            prompt_embeds = None
            negative_embeds = None

            for stage in model_config["pipeline_stages"]:
                stage_args = {**kwargs}  # Copy base args

                # Add stage-specific args
                if stage.get("input_key") and output is not None:
                    stage_args[stage["input_key"]] = output

                # Add any stage-specific generation args
                if stage.get("args"):
                    stage_args.update(stage["args"])

                # Handle prompt embeddings
                if stage.get("use_prompt_embeds") and prompt_embeds is not None:
                    stage_args["prompt_embeds"] = prompt_embeds
                    stage_args["negative_prompt_embeds"] = negative_embeds
                    stage_args.pop("prompt", None)
                elif stage.get("save_prompt_embeds"):
                    # Get embeddings directly from encode_prompt
                    prompt_embeds, negative_embeds = model[stage["name"]].encode_prompt(
                        prompt=prompt,
                        device=model[stage["name"]].device,
                        num_images_per_prompt=stage_args.get(
                            "num_images_per_prompt", 1
                        ),
                    )
                    stage_args["prompt_embeds"] = prompt_embeds
                    stage_args["negative_prompt_embeds"] = negative_embeds
                    stage_args.pop("prompt", None)
                else:
                    stage_args["prompt"] = prompt

                # Run stage
                result = model[stage["name"]](**stage_args)

                # Extract output based on stage config
                output = getattr(result, stage.get("output_attr", "images"))

                # Clear memory if configured
                if model_config.get("clear_memory_on_stage_end"):
                    import gc
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

            return result

        return generate

    # Default single-stage pipeline
    return lambda prompt, **kwargs: model(prompt=prompt, **kwargs)


def enable_model_optimizations(
    model: Any,
    device: str,
    enable_cpu_offload: bool = False,
    enable_sequential_cpu_offload: bool = False,
    enable_vae_slicing: bool = False,
    enable_vae_tiling: bool = False,
    disable_progress_bar: bool = True,
    stage_name: Optional[str] = None,
) -> None:
    """
    Enables various model optimizations for better memory usage and performance.

    Args:
        model: The model to optimize
        device: Device to move model to ('cuda', 'cpu', etc)
        enable_cpu_offload: Whether to enable model CPU offloading
        enable_sequential_cpu_offload: Whether to enable sequential CPU offloading
        enable_vae_slicing: Whether to enable VAE slicing
        enable_vae_tiling: Whether to enable VAE tiling
        disable_progress_bar: Whether to disable the progress bar
        stage_name: Optional name of pipeline stage for logging
    """
    model_name = f"{stage_name} " if stage_name else ""

    if disable_progress_bar:
        bt.logging.debug(f"Disabling progress bar for {model_name}model")
        model.set_progress_bar_config(disable=True)

    # Handle CPU offloading
    if enable_cpu_offload:
        bt.logging.debug(f"Enabling CPU offload for {model_name}model")
        model.enable_model_cpu_offload(device=device)
    elif enable_sequential_cpu_offload:
        bt.logging.debug(f"Enabling sequential CPU offload for {model_name}model")
        model.enable_sequential_cpu_offload()
    else:
        # Only move to device if not using CPU offload
        bt.logging.debug(f"Moving {model_name}model to {device}")
        model.to(device)

    # Handle VAE optimizations if not using CPU offload
    if not enable_cpu_offload:
        # Store the model's dtype for ensuring VAE consistency
        model_dtype = None
        if hasattr(model, 'dtype'):
            model_dtype = model.dtype
        elif hasattr(model, 'unet') and hasattr(model.unet, 'dtype'):
            model_dtype = model.unet.dtype
        
        if enable_vae_slicing:
            bt.logging.debug(f"Enabling VAE slicing for {model_name}model")
            try:
                model.vae.enable_slicing()
            except Exception:
                try:
                    model.enable_vae_slicing()
                except Exception as e:
                    bt.logging.warning(
                        f"Failed to enable VAE slicing for {model_name}model: {e}"
                    )

        if enable_vae_tiling:
            bt.logging.debug(f"Enabling VAE tiling for {model_name}model")
            try:
                model.vae.enable_tiling()
            except Exception:
                try:
                    model.enable_vae_tiling()
                except Exception as e:
                    bt.logging.warning(
                        f"Failed to enable VAE tiling for {model_name}model: {e}"
                    )

        # Ensure VAE stays in the correct dtype after optimizations
        # This fixes "expected scalar type Half but found Float" errors
        if model_dtype is not None and hasattr(model, 'vae'):
            try:
                if model.vae.dtype != model_dtype:
                    bt.logging.debug(f"Casting VAE from {model.vae.dtype} to {model_dtype} for dtype consistency")
                    model.vae.to(dtype=model_dtype)
            except Exception as e:
                bt.logging.debug(f"Could not cast VAE dtype: {e}")
