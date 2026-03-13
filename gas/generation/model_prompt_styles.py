"""
Model-specific prompt optimization configurations.

This module provides per-model prompt formatting rules and best practices
to optimize generation quality for different AI image/video models.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import random

from gas.generation.prompt_modifiers import PromptModifiers, MODIFIERS


@dataclass
class ModelPromptConfig:
    """Configuration for model-specific prompt optimization."""
    
    # Prompt format style
    format: str = "natural_language"  # "natural_language", "tag_based", "hybrid", "action_focused"
    
    # Whether to include quality enhancement tags
    quality_tags: bool = False
    
    # Whether the model supports/benefits from composition instructions
    supports_composition: bool = True
    
    # Whether the model uses negative prompts
    negative_prompt: bool = False
    
    # Base negative prompt if applicable
    negative_base: str = ""
    
    # Optimal token length range (min, max)
    optimal_length: Tuple[int, int] = (50, 150)
    
    # Video-specific: emphasize motion descriptions
    motion_emphasis: bool = False
    
    # Video-specific: supports camera motion instructions
    camera_motion: bool = False
    
    # Video-specific: supports temporal descriptors
    temporal_descriptors: bool = False
    
    # Prefix to add to all prompts (if any)
    prefix: str = ""
    
    # Suffix to add to all prompts (if any)
    suffix: str = ""
    
    # Modifier intensity for this model
    modifier_intensity: str = "moderate"  # "minimal", "moderate", "rich"
    
    # Categories of modifiers that work well with this model
    preferred_modifier_categories: List[str] = field(default_factory=lambda: [
        "style", "lighting", "mood", "color_palette"
    ])


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

MODEL_STYLES: Dict[str, ModelPromptConfig] = {
    # -------------------------------------------------------------------------
    # FLUX Models
    # -------------------------------------------------------------------------
    "flux": ModelPromptConfig(
        format="natural_language",
        quality_tags=False,  # FLUX doesn't need quality tags
        supports_composition=True,  # Good at following spatial instructions
        negative_prompt=False,  # FLUX doesn't use negative prompts
        optimal_length=(50, 200),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "lighting", "mood", "atmosphere", "composition"],
    ),
    
    "flux.1-dev": ModelPromptConfig(
        format="natural_language",
        quality_tags=False,
        supports_composition=True,
        negative_prompt=False,
        optimal_length=(50, 200),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "lighting", "mood", "atmosphere", "composition"],
    ),
    
    "flux.1-schnell": ModelPromptConfig(
        format="natural_language",
        quality_tags=False,
        supports_composition=True,
        negative_prompt=False,
        optimal_length=(30, 100),  # Shorter prompts work better for schnell
        modifier_intensity="minimal",
        preferred_modifier_categories=["style", "lighting", "mood"],
    ),
    
    # -------------------------------------------------------------------------
    # Stable Diffusion XL Models
    # -------------------------------------------------------------------------
    "sdxl": ModelPromptConfig(
        format="hybrid",  # Natural language + tags work well
        quality_tags=True,  # Benefits from quality tags
        supports_composition=True,
        negative_prompt=True,
        negative_base="low quality, blurry, bad anatomy, watermark, text, logo, signature, cropped, worst quality, jpeg artifacts",
        optimal_length=(30, 77),  # CLIP token limit consideration
        modifier_intensity="rich",
        preferred_modifier_categories=["style", "lighting", "camera", "quality_tags", "mood", "color_palette"],
    ),
    
    "stable-diffusion-xl": ModelPromptConfig(
        format="hybrid",
        quality_tags=True,
        supports_composition=True,
        negative_prompt=True,
        negative_base="low quality, blurry, bad anatomy, watermark, text, logo, signature, cropped, worst quality, jpeg artifacts",
        optimal_length=(30, 77),
        modifier_intensity="rich",
        preferred_modifier_categories=["style", "lighting", "camera", "quality_tags", "mood", "color_palette"],
    ),
    
    "realvis": ModelPromptConfig(
        format="hybrid",
        quality_tags=True,
        supports_composition=True,
        negative_prompt=True,
        negative_base="cartoon, anime, illustration, painting, drawing, low quality, blurry, bad anatomy",
        optimal_length=(30, 77),
        modifier_intensity="rich",
        preferred_modifier_categories=["technical", "camera", "lighting", "quality_tags"],
    ),
    
    # -------------------------------------------------------------------------
    # Stable Diffusion 1.x/2.x Models
    # -------------------------------------------------------------------------
    "stable-diffusion": ModelPromptConfig(
        format="tag_based",
        quality_tags=True,
        supports_composition=False,  # Less capable at composition
        negative_prompt=True,
        negative_base="low quality, blurry, bad anatomy, watermark, text",
        optimal_length=(20, 77),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "lighting", "quality_tags"],
    ),
    
    "openjourney": ModelPromptConfig(
        format="tag_based",
        quality_tags=True,
        supports_composition=False,
        negative_prompt=True,
        prefix="mdjrny-v4 style",  # Model-specific trigger
        negative_base="low quality, blurry, bad anatomy",
        optimal_length=(20, 60),
        modifier_intensity="moderate",
        preferred_modifier_categories=["art_movement", "style", "mood"],
    ),
    
    "animagine": ModelPromptConfig(
        format="tag_based",
        quality_tags=True,
        supports_composition=False,
        negative_prompt=True,
        negative_base="low quality, worst quality, bad anatomy, bad hands",
        optimal_length=(20, 77),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "mood", "color_palette"],
    ),
    
    # -------------------------------------------------------------------------
    # HiDream
    # -------------------------------------------------------------------------
    "hidream": ModelPromptConfig(
        format="natural_language",
        quality_tags=False,
        supports_composition=True,
        negative_prompt=False,
        optimal_length=(50, 150),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "lighting", "mood", "atmosphere"],
    ),
    
    # -------------------------------------------------------------------------
    # CogView
    # -------------------------------------------------------------------------
    "cogview": ModelPromptConfig(
        format="natural_language",
        quality_tags=False,
        supports_composition=True,
        negative_prompt=False,
        optimal_length=(50, 150),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "lighting", "mood"],
    ),
    
    # -------------------------------------------------------------------------
    # Chroma
    # -------------------------------------------------------------------------
    "chroma": ModelPromptConfig(
        format="natural_language",
        quality_tags=False,
        supports_composition=True,
        negative_prompt=False,
        optimal_length=(50, 150),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "lighting", "mood", "color_palette"],
    ),
    
    # -------------------------------------------------------------------------
    # DeepFloyd IF
    # -------------------------------------------------------------------------
    "deepfloyd": ModelPromptConfig(
        format="natural_language",
        quality_tags=True,
        supports_composition=True,
        negative_prompt=True,
        negative_base="blurry, low quality, bad anatomy, watermark",
        optimal_length=(50, 150),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "lighting", "quality_tags"],
    ),
    
    # -------------------------------------------------------------------------
    # Janus
    # -------------------------------------------------------------------------
    "janus": ModelPromptConfig(
        format="natural_language",
        quality_tags=False,
        supports_composition=True,
        negative_prompt=False,
        optimal_length=(50, 150),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "mood", "atmosphere"],
    ),
    
    # -------------------------------------------------------------------------
    # Video Models
    # -------------------------------------------------------------------------
    "cogvideo": ModelPromptConfig(
        format="action_focused",
        quality_tags=False,
        supports_composition=True,
        negative_prompt=False,
        motion_emphasis=True,
        camera_motion=True,
        temporal_descriptors=True,
        optimal_length=(50, 150),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "mood", "atmosphere", "lighting"],
    ),
    
    "cogvideox": ModelPromptConfig(
        format="action_focused",
        quality_tags=False,
        supports_composition=True,
        negative_prompt=False,
        motion_emphasis=True,
        camera_motion=True,
        temporal_descriptors=True,
        optimal_length=(50, 150),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "mood", "atmosphere", "lighting"],
    ),
    
    "hunyuan": ModelPromptConfig(
        format="natural_language",
        quality_tags=False,
        supports_composition=True,
        negative_prompt=False,
        motion_emphasis=True,
        camera_motion=True,
        temporal_descriptors=True,
        optimal_length=(50, 200),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "mood", "atmosphere", "lighting"],
    ),
    
    "hunyuanvideo": ModelPromptConfig(
        format="natural_language",
        quality_tags=False,
        supports_composition=True,
        negative_prompt=False,
        motion_emphasis=True,
        camera_motion=True,
        temporal_descriptors=True,
        optimal_length=(50, 200),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "mood", "atmosphere", "lighting"],
    ),
    
    "mochi": ModelPromptConfig(
        format="natural_language",
        quality_tags=False,
        supports_composition=True,
        negative_prompt=False,
        motion_emphasis=True,
        camera_motion=True,
        temporal_descriptors=True,
        optimal_length=(50, 150),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "mood", "atmosphere"],
    ),
    
    "animatediff": ModelPromptConfig(
        format="hybrid",
        quality_tags=True,
        supports_composition=False,
        negative_prompt=True,
        motion_emphasis=True,
        camera_motion=False,  # Limited camera control
        temporal_descriptors=True,
        negative_base="low quality, blurry, bad anatomy, static, no motion",
        optimal_length=(30, 77),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "mood", "quality_tags"],
    ),
    
    "wan": ModelPromptConfig(
        format="natural_language",
        quality_tags=False,
        supports_composition=True,
        negative_prompt=False,
        motion_emphasis=True,
        camera_motion=True,
        temporal_descriptors=True,
        optimal_length=(50, 200),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "mood", "atmosphere", "lighting"],
    ),
    
    "text-to-video-ms": ModelPromptConfig(
        format="natural_language",
        quality_tags=False,
        supports_composition=False,
        negative_prompt=False,
        motion_emphasis=True,
        camera_motion=False,
        temporal_descriptors=True,
        optimal_length=(30, 100),
        modifier_intensity="minimal",
        preferred_modifier_categories=["style", "mood"],
    ),
    
    # -------------------------------------------------------------------------
    # Inpainting Models
    # -------------------------------------------------------------------------
    "inpainting": ModelPromptConfig(
        format="hybrid",
        quality_tags=True,
        supports_composition=True,
        negative_prompt=True,
        negative_base="blurry, low quality, bad blending, visible seam, artifact",
        optimal_length=(30, 77),
        modifier_intensity="minimal",
        preferred_modifier_categories=["style", "lighting"],
    ),
    
    "dreamshaper": ModelPromptConfig(
        format="hybrid",
        quality_tags=True,
        supports_composition=True,
        negative_prompt=True,
        negative_base="blurry, low quality, bad anatomy, watermark",
        optimal_length=(30, 77),
        modifier_intensity="moderate",
        preferred_modifier_categories=["style", "lighting", "mood", "quality_tags"],
    ),
}

# Default configuration for unknown models
DEFAULT_CONFIG = ModelPromptConfig(
    format="natural_language",
    quality_tags=False,
    supports_composition=True,
    negative_prompt=False,
    optimal_length=(50, 150),
    modifier_intensity="moderate",
    preferred_modifier_categories=["style", "lighting", "mood"],
)


# =============================================================================
# PROMPT OPTIMIZER CLASS
# =============================================================================

class ModelPromptOptimizer:
    """
    Optimizes prompts based on target model characteristics.
    """
    
    def __init__(self, modifiers: Optional[PromptModifiers] = None):
        """
        Initialize the optimizer.
        
        Args:
            modifiers: Optional PromptModifiers instance for enhancement.
        """
        self.modifiers = modifiers or PromptModifiers()
        self.model_configs = MODEL_STYLES
    
    def get_model_config(self, model_name: str) -> ModelPromptConfig:
        """
        Get the configuration for a specific model.
        
        Args:
            model_name: The model name or path.
            
        Returns:
            ModelPromptConfig for the model.
        """
        # Normalize model name
        model_lower = model_name.lower()
        
        # Try exact match first
        if model_lower in self.model_configs:
            return self.model_configs[model_lower]
        
        # Try partial matches
        for key, config in self.model_configs.items():
            if key in model_lower or model_lower in key:
                return config
        
        # Check for known patterns
        if "flux" in model_lower:
            return self.model_configs.get("flux", DEFAULT_CONFIG)
        elif "sdxl" in model_lower or "xl" in model_lower:
            return self.model_configs.get("sdxl", DEFAULT_CONFIG)
        elif "cogvideo" in model_lower:
            return self.model_configs.get("cogvideo", DEFAULT_CONFIG)
        elif "hunyuan" in model_lower:
            return self.model_configs.get("hunyuan", DEFAULT_CONFIG)
        elif "wan" in model_lower:
            return self.model_configs.get("wan", DEFAULT_CONFIG)
        elif "mochi" in model_lower:
            return self.model_configs.get("mochi", DEFAULT_CONFIG)
        elif "hidream" in model_lower:
            return self.model_configs.get("hidream", DEFAULT_CONFIG)
        elif "animatediff" in model_lower:
            return self.model_configs.get("animatediff", DEFAULT_CONFIG)
        elif "inpaint" in model_lower:
            return self.model_configs.get("inpainting", DEFAULT_CONFIG)
        
        return DEFAULT_CONFIG
    
    def optimize_prompt(
        self,
        prompt: str,
        model_name: str,
        add_modifiers: bool = True,
        content_type: str = "general",
    ) -> Dict[str, Any]:
        """
        Optimize a prompt for a specific model.
        
        Args:
            prompt: The base prompt to optimize.
            model_name: Target model name.
            add_modifiers: Whether to add enhancement modifiers.
            content_type: Type of content ("portrait", "landscape", "action", "general")
            
        Returns:
            Dictionary containing optimized prompt and optional negative prompt.
        """
        config = self.get_model_config(model_name)
        
        # Start with the base prompt
        optimized = prompt.strip()
        
        # Add prefix if specified
        if config.prefix:
            optimized = f"{config.prefix} {optimized}"
        
        # Add modifiers if requested
        if add_modifiers:
            # Sample from preferred categories
            modifier_parts = []
            for category in config.preferred_modifier_categories:
                if random.random() < 0.5:  # 50% chance per category
                    modifier = self.modifiers.sample_modifier(category)
                    if modifier:
                        modifier_parts.append(modifier)
            
            if modifier_parts:
                if config.format == "tag_based":
                    # Tag-based format: append as comma-separated tags
                    optimized = f"{optimized}, {', '.join(modifier_parts)}"
                else:
                    # Natural language: integrate more smoothly
                    optimized = f"{optimized}, {', '.join(modifier_parts)}"
        
        # Add quality tags if the model benefits from them
        if config.quality_tags:
            quality_samples = random.sample(
                MODIFIERS.get("quality_tags", []),
                min(2, len(MODIFIERS.get("quality_tags", [])))
            )
            optimized = f"{optimized}, {', '.join(quality_samples)}"
        
        # Add suffix if specified
        if config.suffix:
            optimized = f"{optimized} {config.suffix}"
        
        # Build result
        result = {
            "prompt": optimized,
            "model_config": config,
        }
        
        # Add negative prompt if model uses it
        if config.negative_prompt:
            negative = config.negative_base
            if content_type == "portrait":
                negative += ", " + ", ".join(random.sample(
                    MODIFIERS.get("negative_portrait", []),
                    min(3, len(MODIFIERS.get("negative_portrait", [])))
                ))
            elif content_type == "landscape":
                negative += ", " + ", ".join(random.sample(
                    MODIFIERS.get("negative_landscape", []),
                    min(3, len(MODIFIERS.get("negative_landscape", [])))
                ))
            result["negative_prompt"] = negative
        
        return result
    
    def format_for_video(
        self,
        prompt: str,
        model_name: str,
        motion_level: str = "moderate",
    ) -> str:
        """
        Format a prompt for video generation with appropriate motion descriptors.
        
        Args:
            prompt: The base prompt.
            model_name: Target video model name.
            motion_level: Level of motion ("subtle", "moderate", "dynamic")
            
        Returns:
            Prompt formatted for video generation.
        """
        config = self.get_model_config(model_name)
        
        if not config.motion_emphasis:
            return prompt
        
        # Motion intensity descriptors
        motion_descriptors = {
            "subtle": ["gentle", "slight", "minimal", "soft", "delicate"],
            "moderate": ["smooth", "steady", "natural", "flowing", "continuous"],
            "dynamic": ["energetic", "bold", "dramatic", "intense", "powerful"],
        }
        
        # Camera movements (if supported)
        camera_movements = [
            "slow pan", "gentle dolly", "smooth tracking shot",
            "subtle zoom", "steady crane movement", "floating camera",
        ]
        
        motion_word = random.choice(motion_descriptors.get(motion_level, motion_descriptors["moderate"]))
        
        enhanced = prompt
        
        if config.camera_motion:
            camera = random.choice(camera_movements)
            enhanced = f"{enhanced}, {camera}"
        
        if config.temporal_descriptors:
            enhanced = f"{motion_word} motion, {enhanced}"
        
        return enhanced


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_model_config(model_name: str) -> ModelPromptConfig:
    """Get configuration for a model by name."""
    optimizer = ModelPromptOptimizer()
    return optimizer.get_model_config(model_name)


def optimize_prompt_for_model(
    prompt: str,
    model_name: str,
    add_modifiers: bool = True,
) -> Dict[str, Any]:
    """Optimize a prompt for a specific model."""
    optimizer = ModelPromptOptimizer()
    return optimizer.optimize_prompt(prompt, model_name, add_modifiers)


def is_video_model(model_name: str) -> bool:
    """Check if a model is configured for video generation."""
    config = get_model_config(model_name)
    return config.motion_emphasis or config.camera_motion

