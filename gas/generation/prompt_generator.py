import re
import gc
import json
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import contextlib

import bittensor as bt
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    pipeline,
)
import numpy as np

from gas.types import Modality


IMAGE_ANNOTATION_MODEL: str = "Qwen/Qwen2.5-VL-3B-Instruct"
TEXT_MODERATION_MODEL: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"


class PromptGenerator:
    """
    A class for generating and moderating image annotations using transformer models.

    This class provides functionality to generate descriptive captions for images
    using BLIP2 models and optionally moderate the generated text using a separate
    language model.
    """

    def __init__(
        self,
        vlm_name: str = IMAGE_ANNOTATION_MODEL,
        llm_name: str = TEXT_MODERATION_MODEL,
        device: str = "cuda",
    ) -> None:
        """
        Initialize the ImageAnnotationGenerator with specific models and device settings.

        Args:
            model_name: The name of the BLIP model for generating image captions.
            text_moderation_model_name: The name of the model used for moderating
                text descriptions.
            device: The device to use.
            apply_moderation: Flag to determine whether text moderation should be
                applied to captions.
        """
        self.vlm_name = vlm_name
        self.llm_name = llm_name
        self.vlm_processor = None
        self.vlm = None
        self.llm = None
        self.device = device

    def load_vlm(self) -> None:
        """
        Load the vision-language model for image annotation with local-first loading.
        Uses Qwen3-VL-2B-Instruct with flash_attention_2 for speed optimization.
        """
        bt.logging.debug(f"Loading caption generation model {self.vlm_name}")

        # Determine attention implementation based on availability
        attn_impl = "flash_attention_2"
        try:
            import flash_attn
        except ImportError:
            bt.logging.warning("flash_attn not available, using eager attention")
            attn_impl = "eager"

        try:
            bt.logging.info(f"Attempting to load {self.vlm_name} from local cache...")
            self.vlm_processor = AutoProcessor.from_pretrained(
                self.vlm_name, local_files_only=True
            )
            self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.vlm_name,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
                device_map="auto",
                local_files_only=True,
            )
        except (OSError, ValueError, TypeError) as e:
            bt.logging.info(f"Model not in local cache, downloading from HuggingFace...")
            self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_name)
            self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.vlm_name,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
                device_map="auto",
            )

        bt.logging.info(f"Loaded image annotation model {self.vlm_name}")

    def load_llm(self) -> None:
        """
        Load the language model for text moderation with local-first loading.
        """
        bt.logging.debug(f"Loading caption moderation model {self.llm_name}")
        m = re.match(r"cuda:(\d+)", self.device)
        gpu_id = int(m.group(1)) if m else 0

        try:
            bt.logging.info(f"Attempting to load {self.llm_name} from local cache...")
            llm = AutoModelForCausalLM.from_pretrained(
                self.llm_name, torch_dtype=torch.bfloat16, device_map={"": gpu_id},
                local_files_only=True
            )
            tokenizer = AutoTokenizer.from_pretrained(self.llm_name, local_files_only=True)
        except (OSError, ValueError) as e:
            bt.logging.info(f"Model not in local cache, downloading from HuggingFace...")
            llm = AutoModelForCausalLM.from_pretrained(
                self.llm_name, torch_dtype=torch.bfloat16, device_map={"": gpu_id}
            )
            tokenizer = AutoTokenizer.from_pretrained(self.llm_name)

        self.llm = pipeline("text-generation", model=llm, tokenizer=tokenizer)
        bt.logging.info(f"Loaded caption moderation model {self.llm_name}")

    def load_models(self) -> None:
        """
        Load the necessary models for image annotation and text moderation onto
        the specified device.
        """
        if self.vlm is None:
            self.load_vlm()
        else:
            bt.logging.warning(f"vlm already loaded")

        if self.llm is None:
            self.load_llm()
        else:
            bt.logging.warning(f"llm already loaded")

    def clear_gpu(self) -> None:
        """
        Clear GPU memory by moving models back to CPU and deleting them,
        followed by collecting garbage.
        """
        bt.logging.debug("Clearing GPU memory after prompt generation")
        if self.vlm:
            del self.vlm
            self.vlm = None

        if self.llm:
            del self.llm
            self.llm = None

        gc.collect()
        torch.cuda.empty_cache()

    def generate_prompt_from_image(
        self, image: Image.Image, intended_modality: str = None, max_new_tokens: int = 256,
        target_model: str = None
    ) -> str:
        """
        Generate a detailed description for a given image using Qwen3-VL.

        Args:
            image: The image for which the description is to be generated.
            intended_modality: The generation modality (IMAGE or VIDEO). If VIDEO,
                motion descriptions will be added.
            max_new_tokens: The maximum number of tokens to generate.
            target_model: Optional target model name for model-specific optimization.

        Returns:
            A generated description of the image.
        """
        if self.vlm is None or self.vlm_processor is None:
            self.load_vlm()

        # Select a random caption template for variety
        caption_templates = [
            "Describe this image in comprehensive detail. Include the main subject, setting, "
            "lighting conditions, color palette, mood, composition, and any notable visual elements. "
            "Make the description suitable for AI image generation.",
            
            "Analyze this image thoroughly. Describe: 1) The primary subject and their/its appearance, "
            "2) The environment and background, 3) The lighting and atmosphere, 4) The artistic style "
            "or photographic qualities, 5) Colors and textures present.",
            
            "Provide a detailed prompt that could recreate this image. Focus on the subject, "
            "scene composition, lighting quality, color tones, mood, and any distinctive visual "
            "characteristics that define this image.",
            
            "What is depicted in this image? Give a rich, detailed description covering the subject, "
            "background, lighting, colors, style, and overall atmosphere. Be specific and descriptive.",
            
            "Create a comprehensive visual description of this image. Include details about the main "
            "elements, spatial composition, lighting direction and quality, color scheme, textures, "
            "and the emotional tone conveyed.",
        ]
        
        selected_template = random.choice(caption_templates)
        
        # Build the message for Qwen3-VL chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": selected_template},
                ],
            }
        ]

        # Process with chat template
        text = self.vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.vlm_processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
            #min_pixels=256 * 28 * 28,
            #max_pixels=1280 * 28 * 28,
        )
        inputs = inputs.to(self.vlm.device)

        # Generate description
        generated_ids = self.vlm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        
        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        description = self.vlm_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
            )[0].strip()

        bt.logging.trace(f"Generated description: {description[:200]}...")

        # Clean up the description
        if not description.endswith("."):
            description += "."

        # Moderate the description
        moderated_description = self.moderate(description)

        # Enhance for video if needed
        if intended_modality == Modality.VIDEO:
            return self.enhance(moderated_description)
        
        return moderated_description

    def moderate(self, description: str, max_new_tokens: int = 80) -> str:
        """
        Use the text moderation pipeline to make the description more concise
        and neutral.

        Args:
            description: The text description to be moderated.
            max_new_tokens: Maximum number of new tokens to generate in the
                moderated text.

        Returns:
            The moderated description text, or the original description if
            moderation fails.
        """
        if self.llm is None:
            self.load_llm()

        messages = [
            {
                "role": "system",
                "content": (
                    "[INST]You always concisely rephrase given descriptions, "
                    "eliminate redundancy, and remove all specific references to "
                    "individuals by name. You do not respond with anything other "
                    "than the revised description.[/INST]"
                ),
            },
            {"role": "user", "content": description},
        ]
        try:
            moderated_text = self.llm(
                messages,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.llm.tokenizer.eos_token_id,
                return_full_text=False,
            )
            return moderated_text[0]["generated_text"]

        except Exception as e:
            bt.logging.error(f"An error occurred during moderation: {e}", exc_info=True)
            return description

    def enhance(
        self,
        description: str,
        max_new_tokens: int = 120,
        motion_intensity: str = "moderate",
        target_model: str = None,
    ) -> str:
        """
        Enhance a static image description to make it suitable for video generation
        by adding dynamic elements, camera movements, and motion descriptors.

        Args:
            description: The static image description to enhance.
            max_new_tokens: Maximum number of new tokens to generate.
            motion_intensity: Level of motion ("subtle", "moderate", "dynamic").
            target_model: Optional target model for model-specific enhancements.

        Returns:
            An enhanced description suitable for video generation.
        """
        if self.llm is None:
            self.load_llm()

        # Comprehensive motion vocabulary
        camera_movements = {
            "subtle": [
                "slow pan", "gentle dolly", "subtle zoom", "slight tilt",
                "floating camera", "steady glide", "minimal drift",
            ],
            "moderate": [
                "smooth pan", "tracking shot", "dolly zoom", "crane movement",
                "orbiting camera", "push in", "pull out", "follow shot",
            ],
            "dynamic": [
                "sweeping pan", "dramatic crane", "fast tracking", "whip pan",
                "dynamic orbit", "rapid zoom", "handheld movement", "action follow",
            ],
        }
        
        motion_descriptors = {
            "subtle": [
                "gentle movement", "slight motion", "delicate sway", "soft drift",
                "barely perceptible motion", "calm and steady", "minimal animation",
            ],
            "moderate": [
                "smooth motion", "natural movement", "flowing action", "steady pace",
                "continuous motion", "rhythmic movement", "balanced dynamics",
            ],
            "dynamic": [
                "energetic motion", "bold movement", "dramatic action", "intense pace",
                "powerful dynamics", "rapid motion", "vibrant action",
            ],
        }
        
        temporal_pacing = {
            "subtle": ["real-time", "natural pace", "leisurely tempo"],
            "moderate": ["steady rhythm", "consistent pace", "natural timing"],
            "dynamic": ["quick cuts feel", "rapid pacing", "energetic tempo", "slow motion contrast"],
        }
        
        subject_motions = {
            "subtle": [
                "breathing gently", "eyes blinking", "hair swaying slightly",
                "leaves rustling", "water rippling", "fabric flowing softly",
            ],
            "moderate": [
                "walking naturally", "gesturing", "turning slowly",
                "wind blowing", "clouds drifting", "water flowing",
            ],
            "dynamic": [
                "running", "jumping", "dancing energetically",
                "explosion", "rapid transformation", "intense action",
            ],
        }
        
        environmental_motions = {
            "subtle": [
                "dust particles floating", "light rays shifting", "shadows moving slowly",
                "grass swaying gently", "smoke wisping", "mist drifting",
            ],
            "moderate": [
                "clouds moving", "trees swaying", "water flowing steadily",
                "rain falling", "snow drifting", "birds flying past",
            ],
            "dynamic": [
                "storm brewing", "waves crashing", "fire blazing",
                "lightning flashing", "leaves swirling", "debris flying",
            ],
        }
        
        # Select motion elements based on intensity
        intensity = motion_intensity if motion_intensity in camera_movements else "moderate"
        
        selected_camera = random.choice(camera_movements[intensity])
        selected_motion = random.choice(motion_descriptors[intensity])
        selected_tempo = random.choice(temporal_pacing[intensity])
        selected_env = random.choice(environmental_motions[intensity])

        # Build the enhancement prompt with specific vocabulary
        enhancement_context = f"""Camera movement suggestion: {selected_camera}
Motion quality: {selected_motion}
Temporal feel: {selected_tempo}
Environmental motion: {selected_env}"""

        messages = [
            {
                "role": "system",
                "content": (
                    "[INST]You are an expert at converting image descriptions into compelling video prompts. "
                    "Transform the given description into a dynamic video scene using these guidelines:\n\n"
                    f"MOTION ELEMENTS TO INCORPORATE:\n{enhancement_context}\n\n"
                    "RULES:\n"
                    "1. ANALYZE the scene to identify what could naturally move\n"
                    "2. ADD appropriate camera motion that enhances the scene\n"
                    "3. INCLUDE environmental motion (wind, water, light changes) where fitting\n"
                    "4. For STATIC subjects (portraits, still life): keep subject stable, add subtle environmental motion\n"
                    "5. For ACTION subjects: emphasize and extend the motion naturally\n"
                    "6. Use descriptive motion verbs: gliding, drifting, flowing, sweeping, etc.\n"
                    "7. Describe the motion's QUALITY: smooth, gentle, dramatic, energetic\n"
                    "8. Keep the description concise (2-3 sentences) but vivid\n\n"
                    "Only respond with the enhanced video description.[/INST]"
                ),
            },
            {"role": "user", "content": description},
        ]

        try:
            enhanced_text = self.llm(
                messages,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.llm.tokenizer.eos_token_id,
                return_full_text=False,
                temperature=0.7,
            )
            result = enhanced_text[0]["generated_text"].strip()
            
            # Ensure the result ends properly
            if not result.endswith((".", "!", "?")):
                result += "."
            
            return result

        except Exception as e:
            bt.logging.error(f"An error occurred during motion enhancement: {e}")
            # Fallback: add basic motion to original description
            fallback_motion = f"{description} {selected_camera}, {selected_env}."
            return fallback_motion

    def sanitize(self, prompt: str, max_new_tokens: int = 80) -> str:
        """
        Use the LLM to make the prompt more SFW (less NSFW).
        """

        if self.llm is None:
            self.load_llm()

        messages = [
            {
                "role": "system",
                "content": (
                    "[INST]You are an expert at making prompts safe for work (SFW). "
                    "Rephrase the following prompt to remove or neutralize any NSFW, sexual, or explicit content. "
                    "Keep the prompt as close as possible to the original intent, but ensure it is SFW. "
                    "Only respond with the sanitized prompt.[/INST]"
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            sanitized = self.llm(
                messages,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.llm.tokenizer.eos_token_id,
                return_full_text=False,
            )
            return sanitized[0]["generated_text"]
        except Exception as e:
            bt.logging.error(f"An error occurred during prompt sanitization: {e}")
            return prompt

    def generate_negative_prompt(
        self,
        content_type: str = "general",
        target_model: str = None,
    ) -> str:
        """
        Generate an appropriate negative prompt based on content type and target model.

        Args:
            content_type: Type of content ("portrait", "landscape", "video", "general")
            target_model: Optional target model name for model-specific negatives.

        Returns:
            A negative prompt string appropriate for the content and model.
        """
        # Base negative prompts that apply to most models
        base_negatives = [
            "blurry", "out of focus", "low quality", "low resolution",
            "pixelated", "jpeg artifacts", "compression artifacts",
            "distorted", "deformed", "disfigured", "mutated",
            "watermark", "text", "logo", "signature", "copyright",
            "cropped", "cut off", "partial", "incomplete",
            "overexposed", "underexposed", "bad lighting",
        ]
        
        # Content-specific negatives
        portrait_negatives = [
            "bad anatomy", "wrong anatomy", "extra limbs", "missing limbs",
            "mutated hands", "extra fingers", "missing fingers", "fused fingers",
            "deformed face", "ugly face", "asymmetric face",
            "crossed eyes", "dead eyes", "uncanny valley",
            "bad proportions", "long neck", "long body",
        ]
        
        landscape_negatives = [
            "oversaturated", "unrealistic colors", "bad composition",
            "cluttered", "messy", "chaotic", "unbalanced",
            "artificial looking", "fake looking", "obvious cgi",
        ]
        
        video_negatives = [
            "flickering", "temporal inconsistency", "frame jumping",
            "motion blur artifacts", "ghosting", "tearing",
            "stuttering", "jerky motion", "unnatural movement",
            "static", "frozen", "no motion",
        ]
        
        # Build the negative prompt
        negatives = base_negatives.copy()
        
        if content_type == "portrait":
            negatives.extend(random.sample(portrait_negatives, min(8, len(portrait_negatives))))
        elif content_type == "landscape":
            negatives.extend(random.sample(landscape_negatives, min(5, len(landscape_negatives))))
        elif content_type == "video":
            negatives.extend(random.sample(video_negatives, min(6, len(video_negatives))))
        
        # Model-specific adjustments
        if target_model:
            model_lower = target_model.lower()
            
            # SDXL benefits from more specific quality negatives
            if "sdxl" in model_lower or "xl" in model_lower:
                negatives.extend(["worst quality", "normal quality", "bad quality"])
            
            # Anime models have specific issues
            if "animagine" in model_lower or "anime" in model_lower:
                negatives.extend(["bad hands", "extra digits", "fewer digits"])
            
            # Realistic models should avoid cartoon-like outputs
            if "realvis" in model_lower or "realistic" in model_lower:
                negatives.extend(["cartoon", "anime", "illustration", "painting", "drawing"])
        
        # Shuffle and join
        random.shuffle(negatives)
        return ", ".join(negatives)

    def generate_search_query(self, max_tokens: int = 30) -> str:
        """
        Generate a random Google search query for image retrieval.
        Uses 20+ categories with weighted sampling for diverse, realistic queries.
        """
        if self.llm is None:
            self.load_llm()

        # Comprehensive search categories with subjects, modifiers, and contexts
        search_categories = {
            "nature_landscapes": {
                "subjects": [
                    "mountain landscape", "ocean sunset", "forest path", "desert dunes",
                    "waterfall", "lake reflection", "canyon view", "rolling hills",
                    "volcanic landscape", "glacier", "northern lights", "rainbow",
                    "misty valley", "rocky coastline", "tropical beach", "alpine meadow",
                ],
                "modifiers": ["at sunrise", "at sunset", "in autumn", "in winter",
                              "after rain", "during storm", "in spring bloom", "at night"],
                "locations": ["Patagonia", "Iceland", "Swiss Alps", "New Zealand",
                              "Norway fjords", "Japanese countryside", "Scottish Highlands"],
            },
            "wildlife": {
                "subjects": [
                    "lion hunting", "eagle in flight", "whale breaching", "bear fishing",
                    "wolf pack", "elephant herd", "tiger stalking", "gorilla family",
                    "penguin colony", "flamingo flock", "owl hunting", "fox in snow",
                    "deer in forest", "cheetah running", "dolphin jumping", "butterfly migration",
                ],
                "modifiers": ["close up", "in natural habitat", "at dawn", "wildlife photography",
                              "national geographic style", "action shot", "portrait"],
            },
            "urban_city": {
                "subjects": [
                    "city skyline", "street scene", "neon signs", "subway station",
                    "rooftop view", "busy intersection", "alleyway", "skyscraper",
                    "bridge at night", "urban park", "street market", "cafe terrace",
                    "graffiti wall", "old town street", "modern architecture",
                ],
                "modifiers": ["at night", "rain reflection", "long exposure", "aerial view",
                              "golden hour", "blue hour", "black and white"],
                "locations": ["Tokyo", "New York", "Paris", "London", "Hong Kong",
                              "Dubai", "Singapore", "Barcelona", "Amsterdam"],
            },
            "portraits_people": {
                "subjects": [
                    "professional headshot", "candid street portrait", "elderly person",
                    "child playing", "musician performing", "athlete training",
                    "artist working", "chef cooking", "farmer working", "fisherman",
                    "dancer", "craftsman", "scientist", "doctor", "teacher",
                ],
                "modifiers": ["natural light", "studio lighting", "environmental portrait",
                              "black and white", "close up", "full body", "action shot"],
            },
            "food_cuisine": {
                "subjects": [
                    "gourmet dish", "street food", "traditional cuisine", "dessert plating",
                    "fresh ingredients", "cooking process", "restaurant interior",
                    "food market", "bakery display", "sushi preparation", "pizza making",
                    "wine and cheese", "coffee art", "breakfast spread", "bbq grill",
                ],
                "modifiers": ["overhead shot", "close up", "rustic style", "modern plating",
                              "natural light", "dark moody", "bright and airy"],
                "locations": ["Italian", "Japanese", "Mexican", "Indian", "French",
                              "Thai", "Mediterranean", "Korean", "Vietnamese"],
            },
            "architecture": {
                "subjects": [
                    "modern building", "historic cathedral", "ancient temple", "castle",
                    "bridge", "stadium", "museum interior", "library", "train station",
                    "airport terminal", "office building", "residential home", "palace",
                    "mosque", "pagoda", "ruins", "lighthouse", "windmill",
                ],
                "modifiers": ["exterior", "interior", "detail shot", "wide angle",
                              "symmetrical", "at night", "golden hour", "aerial view"],
                "styles": ["Gothic", "Art Deco", "Brutalist", "Victorian", "Modern",
                           "Traditional", "Minimalist", "Baroque", "Classical"],
            },
            "technology": {
                "subjects": [
                    "smartphone", "laptop setup", "gaming setup", "server room",
                    "robot", "drone", "electric car", "smart home", "VR headset",
                    "3D printer", "circuit board", "satellite", "solar panels",
                    "wind turbine", "laboratory equipment", "medical device",
                ],
                "modifiers": ["product photography", "in use", "close up", "futuristic",
                              "minimalist", "studio shot", "lifestyle"],
            },
            "sports_action": {
                "subjects": [
                    "soccer goal", "basketball dunk", "surfing wave", "skiing",
                    "rock climbing", "marathon runner", "swimming", "cycling",
                    "skateboarding", "martial arts", "gymnastics", "tennis serve",
                    "golf swing", "boxing", "yoga pose", "crossfit workout",
                ],
                "modifiers": ["action shot", "motion blur", "freeze frame", "dramatic lighting",
                              "silhouette", "wide angle", "close up intensity"],
            },
            "art_creative": {
                "subjects": [
                    "oil painting", "sculpture", "street art", "gallery installation",
                    "pottery making", "glassblowing", "textile art", "photography exhibition",
                    "digital art display", "mural", "mosaic", "calligraphy",
                    "woodcarving", "jewelry making", "tattoo art",
                ],
                "modifiers": ["detail shot", "artist at work", "gallery setting",
                              "studio", "process", "finished piece", "exhibition"],
            },
            "fashion_style": {
                "subjects": [
                    "fashion model", "street style", "runway show", "vintage fashion",
                    "haute couture", "casual outfit", "formal wear", "accessories",
                    "shoes", "handbag", "jewelry", "sunglasses", "watch",
                    "hat", "sustainable fashion", "designer clothes",
                ],
                "modifiers": ["editorial", "lookbook", "behind the scenes", "detail shot",
                              "full outfit", "close up", "studio", "outdoor"],
            },
            "vehicles_transport": {
                "subjects": [
                    "classic car", "sports car", "motorcycle", "airplane",
                    "train", "yacht", "bicycle", "helicopter", "truck",
                    "bus", "tram", "ferry", "hot air balloon", "sailboat",
                ],
                "modifiers": ["in motion", "parked", "detail shot", "interior",
                              "aerial view", "sunset", "reflection", "vintage"],
            },
            "interior_design": {
                "subjects": [
                    "living room", "bedroom", "kitchen", "bathroom", "office",
                    "dining room", "home library", "balcony", "garden room",
                    "studio apartment", "loft", "hotel lobby", "restaurant interior",
                ],
                "modifiers": ["modern", "minimalist", "cozy", "luxury", "scandinavian",
                              "industrial", "bohemian", "traditional", "eclectic"],
            },
            "events_celebrations": {
                "subjects": [
                    "wedding ceremony", "birthday party", "graduation", "concert",
                    "festival", "parade", "fireworks", "carnival", "sporting event",
                    "conference", "art opening", "food festival", "cultural celebration",
                ],
                "modifiers": ["candid", "group photo", "detail shot", "atmosphere",
                              "emotional moment", "celebration", "documentary style"],
            },
            "science_education": {
                "subjects": [
                    "laboratory", "microscope view", "space telescope", "chemistry experiment",
                    "biology specimen", "physics demonstration", "astronomy",
                    "classroom", "library study", "research facility", "museum exhibit",
                ],
                "modifiers": ["scientific", "educational", "documentary", "close up",
                              "in action", "detailed", "professional"],
            },
            "weather_atmospheric": {
                "subjects": [
                    "thunderstorm", "lightning strike", "rainbow", "fog",
                    "snow falling", "rain", "clouds", "sunset sky", "starry night",
                    "aurora borealis", "tornado", "hurricane", "frost patterns",
                ],
                "modifiers": ["dramatic", "long exposure", "time lapse", "moody",
                              "atmospheric", "epic", "beautiful"],
            },
            "macro_closeup": {
                "subjects": [
                    "water droplet", "insect", "flower petal", "eye", "texture",
                    "crystal", "feather", "leaf veins", "snowflake", "soap bubble",
                    "spider web", "butterfly wing", "bee on flower", "moss",
                ],
                "modifiers": ["extreme close up", "macro photography", "detailed",
                              "abstract", "colorful", "black and white"],
            },
            "abstract_conceptual": {
                "subjects": [
                    "light trails", "smoke art", "water splash", "paint splatter",
                    "geometric patterns", "reflections", "shadows", "silhouettes",
                    "double exposure", "motion blur art", "kaleidoscope", "fractals",
                ],
                "modifiers": ["artistic", "creative", "experimental", "colorful",
                              "minimalist", "surreal", "dreamlike"],
            },
            "documentary_journalism": {
                "subjects": [
                    "protest", "humanitarian", "environmental", "social issue",
                    "daily life", "working conditions", "community", "tradition",
                    "migration", "urban development", "rural life", "industry",
                ],
                "modifiers": ["photojournalism", "documentary", "candid", "raw",
                              "powerful", "emotional", "storytelling"],
            },
            "pets_domestic": {
                "subjects": [
                    "dog portrait", "cat playing", "puppy", "kitten", "bird",
                    "rabbit", "hamster", "fish aquarium", "horse", "parrot",
                    "guinea pig", "turtle", "pet photography",
                ],
                "modifiers": ["cute", "funny", "action shot", "portrait", "outdoor",
                              "studio", "with owner", "sleeping", "playing"],
            },
            "seasons_holidays": {
                "subjects": [
                    "christmas decorations", "halloween", "easter", "thanksgiving",
                    "new year celebration", "valentine", "autumn leaves",
                    "spring flowers", "summer beach", "winter snow", "harvest",
                ],
                "modifiers": ["festive", "traditional", "cozy", "celebration",
                              "decorations", "family", "seasonal"],
            },
        }
        
        # Weight categories (some more common than others)
        category_weights = {
            "nature_landscapes": 1.5,
            "portraits_people": 1.5,
            "urban_city": 1.2,
            "food_cuisine": 1.0,
            "wildlife": 1.0,
            "architecture": 0.8,
            "technology": 0.8,
            "sports_action": 0.8,
            "art_creative": 0.7,
            "fashion_style": 0.7,
            "vehicles_transport": 0.6,
            "interior_design": 0.6,
            "events_celebrations": 0.6,
            "science_education": 0.5,
            "weather_atmospheric": 0.5,
            "macro_closeup": 0.5,
            "abstract_conceptual": 0.4,
            "documentary_journalism": 0.4,
            "pets_domestic": 0.8,
            "seasons_holidays": 0.5,
        }
        
        # Weighted random selection
        categories = list(search_categories.keys())
        weights = [category_weights.get(c, 1.0) for c in categories]
        selected_category = random.choices(categories, weights=weights, k=1)[0]
        category_data = search_categories[selected_category]
        
        # Build the search query components
        subject = random.choice(category_data["subjects"])
        
        # Optionally add modifier
        query_parts = [subject]
        if "modifiers" in category_data and random.random() < 0.6:
            query_parts.append(random.choice(category_data["modifiers"]))
        
        # Optionally add location/style
        for key in ["locations", "styles"]:
            if key in category_data and random.random() < 0.3:
                query_parts.append(random.choice(category_data[key]))
        
        base_query = " ".join(query_parts)

        messages = [
            {
                "role": "system",
                "content": (
                    f"[INST]Generate ONE creative and realistic Google image search query based on: {base_query}. "
                    "Make it sound natural and specific, as if a real person was searching for reference images. "
                    "You may add relevant details like specific styles, locations, or descriptors. "
                    "Only respond with the search query.[/INST]"
                ),
            },
            {"role": "user", "content": "Generate a search query"},
        ]

        try:
            query = self.llm(
                messages,
                max_new_tokens=max_tokens,
                pad_token_id=self.llm.tokenizer.eos_token_id,
                return_full_text=False,
            )
            return query[0]["generated_text"].strip()
        except Exception as e:
            bt.logging.error(f"An error occurred generating search query: {e}")
            return base_query  # Fallback to the constructed query