"""
Prompt modifier system for enhancing AI image/video generation prompts.

This module provides comprehensive modifier categories that can be sampled
and applied to prompts to create more diverse and descriptive outputs.
"""

import random
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass


# =============================================================================
# MODIFIER DICTIONARIES
# =============================================================================

MODIFIERS: Dict[str, List[str]] = {
    # -------------------------------------------------------------------------
    # Visual Style
    # -------------------------------------------------------------------------
    "style": [
        "cinematic", "photorealistic", "hyperrealistic", "artistic", "documentary",
        "editorial", "fine art", "commercial", "candid", "posed", "spontaneous",
        "stylized", "raw", "polished", "gritty", "clean", "minimalist", "maximalist",
        "abstract", "conceptual", "surreal", "dreamlike", "ethereal", "dramatic",
    ],
    
    "art_movement": [
        "impressionist", "expressionist", "surrealist", "minimalist", "art deco",
        "art nouveau", "baroque", "renaissance", "modernist", "post-modern",
        "pop art", "cubist", "abstract expressionist", "romanticism", "realism",
        "neoclassical", "gothic", "futurism", "constructivism", "dadaism",
    ],
    
    "film_style": [
        "film noir", "neo-noir", "Wes Anderson aesthetic", "Kubrick style",
        "Spielberg cinematic", "Tarantino style", "Christopher Nolan aesthetic",
        "indie film", "blockbuster", "arthouse", "documentary style",
        "French New Wave", "German Expressionism", "Italian Neorealism",
        "Hollywood golden age", "1980s aesthetic", "1990s aesthetic",
        "retro film", "vintage cinema", "modern blockbuster",
    ],
    
    "render_style": [
        "3D render", "CGI", "digital art", "digital painting", "matte painting",
        "concept art", "illustration", "vector art", "pixel art", "voxel art",
        "low poly", "high poly", "photogrammetry", "ray traced", "path traced",
        "Unreal Engine", "Octane render", "V-Ray render", "Blender render",
    ],
    
    # -------------------------------------------------------------------------
    # Technical / Camera
    # -------------------------------------------------------------------------
    "lighting": [
        "golden hour", "blue hour", "magic hour", "midday sun", "overcast",
        "studio lighting", "natural light", "window light", "backlighting",
        "rim lighting", "Rembrandt lighting", "butterfly lighting", "split lighting",
        "loop lighting", "broad lighting", "short lighting", "dramatic lighting",
        "soft lighting", "hard lighting", "diffused lighting", "volumetric lighting",
        "god rays", "chiaroscuro", "low-key lighting", "high-key lighting",
        "neon lighting", "practical lighting", "ambient lighting", "fill light",
    ],
    
    "camera": [
        "35mm lens", "50mm lens", "85mm portrait lens", "24mm wide angle",
        "14mm ultra wide", "135mm telephoto", "200mm telephoto", "macro lens",
        "fisheye lens", "tilt-shift lens", "anamorphic lens", "vintage lens",
        "Leica lens", "Zeiss lens", "Canon L series", "Nikon lens",
        "prime lens", "zoom lens", "cinema lens", "smartphone camera",
    ],
    
    "aperture": [
        "f/1.2 shallow depth", "f/1.4 bokeh", "f/1.8 portrait depth",
        "f/2.8 balanced", "f/4 moderate depth", "f/5.6 landscape",
        "f/8 sharp throughout", "f/11 maximum sharpness", "f/16 deep focus",
        "shallow depth of field", "deep depth of field", "selective focus",
        "bokeh balls", "creamy bokeh", "swirly bokeh", "busy bokeh",
    ],
    
    "technical": [
        "8K resolution", "4K UHD", "high resolution", "ultra detailed",
        "sharp focus", "tack sharp", "crystal clear", "RAW quality",
        "HDR", "high dynamic range", "long exposure", "short exposure",
        "high speed capture", "freeze motion", "motion blur intentional",
        "low ISO clean", "high ISO grain", "film grain", "noise free",
        "professionally retouched", "unedited", "straight out of camera",
    ],
    
    "camera_brand": [
        "shot on Hasselblad", "shot on Phase One", "shot on Canon EOS R5",
        "shot on Sony A7R V", "shot on Nikon Z9", "shot on Fujifilm GFX",
        "shot on Leica M11", "shot on RED camera", "shot on ARRI Alexa",
        "shot on Blackmagic", "medium format", "full frame", "APS-C sensor",
    ],
    
    # -------------------------------------------------------------------------
    # Composition
    # -------------------------------------------------------------------------
    "composition": [
        "rule of thirds", "centered composition", "symmetrical", "asymmetrical",
        "leading lines", "natural framing", "frame within frame", "negative space",
        "golden ratio", "golden spiral", "diagonal composition", "triangular composition",
        "S-curve", "C-curve", "L-shaped composition", "radial composition",
        "layered composition", "foreground interest", "balanced composition",
        "dynamic composition", "static composition", "minimalist composition",
    ],
    
    "perspective": [
        "eye level", "bird's eye view", "worm's eye view", "aerial view",
        "Dutch angle", "tilted angle", "straight on", "three-quarter view",
        "profile view", "isometric view", "forced perspective", "low angle",
        "high angle", "overhead shot", "ground level", "elevated perspective",
    ],
    
    "framing": [
        "extreme close-up", "close-up", "medium close-up", "medium shot",
        "medium long shot", "full shot", "long shot", "extreme long shot",
        "wide shot", "establishing shot", "over-the-shoulder", "point of view",
        "two-shot", "group shot", "insert shot", "cutaway", "master shot",
    ],
    
    # -------------------------------------------------------------------------
    # Atmosphere
    # -------------------------------------------------------------------------
    "color_palette": [
        "warm tones", "cool tones", "neutral tones", "earth tones",
        "monochromatic", "complementary colors", "analogous colors", "triadic colors",
        "vibrant colors", "muted colors", "pastel colors", "saturated colors",
        "desaturated", "high contrast colors", "low contrast", "color graded",
        "teal and orange", "cyan and magenta", "golden tones", "silver tones",
        "sepia tones", "cross-processed", "bleach bypass", "vintage color",
    ],
    
    "mood": [
        "serene", "peaceful", "tranquil", "calm", "relaxing",
        "dramatic", "intense", "powerful", "bold", "striking",
        "mysterious", "enigmatic", "intriguing", "suspenseful", "eerie",
        "joyful", "happy", "cheerful", "uplifting", "celebratory",
        "melancholic", "somber", "pensive", "reflective", "nostalgic",
        "romantic", "intimate", "tender", "passionate", "sensual",
        "energetic", "dynamic", "vibrant", "lively", "exciting",
    ],
    
    "atmosphere": [
        "foggy", "misty", "hazy", "smoky", "dusty",
        "clear", "crisp", "fresh", "clean air",
        "humid", "tropical", "arid", "dry",
        "ethereal", "dreamy", "magical", "enchanting",
        "gritty", "urban decay", "industrial", "raw",
        "cozy", "warm atmosphere", "inviting", "homey",
    ],
    
    "time_period": [
        "modern", "contemporary", "futuristic", "sci-fi",
        "vintage 1920s", "vintage 1950s", "vintage 1970s", "vintage 1980s",
        "retro", "nostalgic", "timeless", "classic",
        "Victorian era", "Edwardian era", "Art Deco era", "Mid-century modern",
        "Ancient", "Medieval", "Renaissance period", "Industrial revolution",
    ],
    
    # -------------------------------------------------------------------------
    # Environment
    # -------------------------------------------------------------------------
    "weather": [
        "sunny", "partly cloudy", "overcast", "cloudy",
        "rainy", "light rain", "heavy rain", "drizzle",
        "stormy", "thunderstorm", "lightning", "dramatic clouds",
        "snowy", "light snow", "blizzard", "frost",
        "foggy morning", "misty", "humid", "dry heat",
        "windy", "calm", "breezy",
    ],
    
    "time_of_day": [
        "dawn", "early morning", "sunrise", "golden hour morning",
        "mid-morning", "late morning", "midday", "noon",
        "early afternoon", "late afternoon", "golden hour evening",
        "sunset", "dusk", "twilight", "blue hour",
        "evening", "night", "midnight", "pre-dawn",
    ],
    
    "season": [
        "spring", "early spring", "late spring", "spring bloom",
        "summer", "early summer", "midsummer", "late summer",
        "autumn", "early fall", "peak fall colors", "late autumn",
        "winter", "early winter", "midwinter", "late winter",
    ],
    
    # -------------------------------------------------------------------------
    # Quality Descriptors (for models that benefit from them)
    # -------------------------------------------------------------------------
    "quality_tags": [
        "masterpiece", "best quality", "high quality", "ultra quality",
        "highly detailed", "intricate details", "fine details",
        "professional photography", "award-winning", "featured on 500px",
        "trending on artstation", "8K wallpaper", "stunning",
    ],
    
    # -------------------------------------------------------------------------
    # Negative Prompt Components
    # -------------------------------------------------------------------------
    "negative_base": [
        "blurry", "out of focus", "soft focus unintentional", "motion blur unintentional",
        "low quality", "low resolution", "pixelated", "jpeg artifacts", "compression artifacts",
        "distorted", "deformed", "disfigured", "mutated",
        "watermark", "text", "logo", "signature", "copyright",
        "cropped", "cut off", "partial", "incomplete",
        "overexposed", "underexposed", "bad lighting", "flat lighting",
    ],
    
    "negative_portrait": [
        "bad anatomy", "wrong anatomy", "extra limbs", "missing limbs",
        "mutated hands", "extra fingers", "missing fingers", "fused fingers",
        "deformed face", "ugly face", "asymmetric face",
        "crossed eyes", "dead eyes", "uncanny valley",
        "bad proportions", "long neck", "long body",
    ],
    
    "negative_landscape": [
        "oversaturated", "unrealistic colors", "bad composition",
        "cluttered", "messy", "chaotic", "unbalanced",
        "artificial looking", "fake", "cgi obvious",
    ],
    
    "negative_video": [
        "flickering", "temporal inconsistency", "frame jumping",
        "motion blur artifacts", "ghosting", "tearing",
        "stuttering", "jerky motion", "unnatural movement",
    ],
}


# =============================================================================
# PROMPT MODIFIER CLASS
# =============================================================================

@dataclass
class ModifierSelection:
    """Represents a selection of modifiers for a prompt."""
    style: Optional[str] = None
    lighting: Optional[str] = None
    camera: Optional[str] = None
    composition: Optional[str] = None
    mood: Optional[str] = None
    color_palette: Optional[str] = None
    atmosphere: Optional[str] = None
    time_of_day: Optional[str] = None
    quality_tags: Optional[List[str]] = None
    negative_prompt: Optional[str] = None
    
    def to_suffix(self) -> str:
        """Convert selected modifiers to a prompt suffix string."""
        parts = []
        for field in ['style', 'lighting', 'camera', 'composition', 'mood', 
                      'color_palette', 'atmosphere', 'time_of_day']:
            value = getattr(self, field)
            if value:
                parts.append(value)
        if self.quality_tags:
            parts.extend(self.quality_tags)
        return ", ".join(parts)


class PromptModifiers:
    """
    A class for sampling and applying prompt modifiers to enhance generation prompts.
    """
    
    def __init__(self, modifiers: Optional[Dict[str, List[str]]] = None):
        """
        Initialize the PromptModifiers with modifier dictionaries.
        
        Args:
            modifiers: Optional custom modifier dictionary. Uses defaults if not provided.
        """
        self.modifiers = modifiers or MODIFIERS
    
    def sample_modifier(self, category: str) -> Optional[str]:
        """
        Sample a single modifier from a category.
        
        Args:
            category: The modifier category to sample from.
            
        Returns:
            A randomly selected modifier string, or None if category doesn't exist.
        """
        if category not in self.modifiers:
            return None
        return random.choice(self.modifiers[category])
    
    def sample_modifiers(
        self,
        categories: List[str],
        num_per_category: int = 1,
    ) -> Dict[str, List[str]]:
        """
        Sample modifiers from multiple categories.
        
        Args:
            categories: List of category names to sample from.
            num_per_category: Number of modifiers to sample per category.
            
        Returns:
            Dictionary mapping category names to lists of sampled modifiers.
        """
        result = {}
        for category in categories:
            if category in self.modifiers:
                available = self.modifiers[category]
                num_to_sample = min(num_per_category, len(available))
                result[category] = random.sample(available, num_to_sample)
        return result
    
    def generate_modifier_selection(
        self,
        content_type: str = "general",
        include_quality_tags: bool = False,
        intensity: str = "moderate",  # "minimal", "moderate", "rich"
    ) -> ModifierSelection:
        """
        Generate a balanced selection of modifiers for a prompt.
        
        Args:
            content_type: Type of content ("portrait", "landscape", "action", "general")
            include_quality_tags: Whether to include quality enhancement tags
            intensity: How many modifiers to include ("minimal", "moderate", "rich")
            
        Returns:
            ModifierSelection with sampled modifiers.
        """
        selection = ModifierSelection()
        
        # Define which categories to sample based on intensity
        if intensity == "minimal":
            categories_to_sample = ["style", "lighting"]
            sample_probability = 0.5
        elif intensity == "moderate":
            categories_to_sample = ["style", "lighting", "mood", "color_palette"]
            sample_probability = 0.6
        else:  # rich
            categories_to_sample = [
                "style", "lighting", "camera", "composition", 
                "mood", "color_palette", "atmosphere", "time_of_day"
            ]
            sample_probability = 0.7
        
        # Sample each category with probability
        for category in categories_to_sample:
            if random.random() < sample_probability:
                value = self.sample_modifier(category)
                if hasattr(selection, category):
                    setattr(selection, category, value)
        
        # Add quality tags if requested
        if include_quality_tags:
            num_tags = random.randint(2, 4)
            selection.quality_tags = random.sample(
                self.modifiers.get("quality_tags", []),
                min(num_tags, len(self.modifiers.get("quality_tags", [])))
            )
        
        return selection
    
    def generate_negative_prompt(
        self,
        content_type: str = "general",
        include_content_specific: bool = True,
    ) -> str:
        """
        Generate an appropriate negative prompt.
        
        Args:
            content_type: Type of content ("portrait", "landscape", "video", "general")
            include_content_specific: Whether to include content-specific negatives
            
        Returns:
            A negative prompt string.
        """
        # Start with base negatives
        negatives = random.sample(
            self.modifiers.get("negative_base", []),
            min(8, len(self.modifiers.get("negative_base", [])))
        )
        
        # Add content-specific negatives
        if include_content_specific:
            if content_type == "portrait":
                negatives.extend(random.sample(
                    self.modifiers.get("negative_portrait", []),
                    min(5, len(self.modifiers.get("negative_portrait", [])))
                ))
            elif content_type == "landscape":
                negatives.extend(random.sample(
                    self.modifiers.get("negative_landscape", []),
                    min(4, len(self.modifiers.get("negative_landscape", [])))
                ))
            elif content_type == "video":
                negatives.extend(random.sample(
                    self.modifiers.get("negative_video", []),
                    min(4, len(self.modifiers.get("negative_video", [])))
                ))
        
        return ", ".join(negatives)
    
    def enhance_prompt(
        self,
        base_prompt: str,
        content_type: str = "general",
        include_quality_tags: bool = False,
        intensity: str = "moderate",
    ) -> str:
        """
        Enhance a base prompt with sampled modifiers.
        
        Args:
            base_prompt: The original prompt to enhance.
            content_type: Type of content for appropriate modifier selection.
            include_quality_tags: Whether to add quality tags.
            intensity: Modifier intensity level.
            
        Returns:
            Enhanced prompt string.
        """
        selection = self.generate_modifier_selection(
            content_type=content_type,
            include_quality_tags=include_quality_tags,
            intensity=intensity,
        )
        
        suffix = selection.to_suffix()
        if suffix:
            return f"{base_prompt}, {suffix}"
        return base_prompt
    
    def get_all_categories(self) -> List[str]:
        """Get list of all available modifier categories."""
        return list(self.modifiers.keys())
    
    def get_category_size(self, category: str) -> int:
        """Get the number of modifiers in a category."""
        return len(self.modifiers.get(category, []))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def sample_style_modifier() -> str:
    """Sample a random style modifier."""
    return random.choice(MODIFIERS["style"])


def sample_lighting_modifier() -> str:
    """Sample a random lighting modifier."""
    return random.choice(MODIFIERS["lighting"])


def sample_camera_modifier() -> str:
    """Sample a random camera modifier."""
    return random.choice(MODIFIERS["camera"])


def sample_mood_modifier() -> str:
    """Sample a random mood modifier."""
    return random.choice(MODIFIERS["mood"])


def get_random_modifiers(
    num_modifiers: int = 3,
    categories: Optional[List[str]] = None,
) -> List[str]:
    """
    Get a list of random modifiers from various categories.
    
    Args:
        num_modifiers: Number of modifiers to return.
        categories: Optional list of categories to sample from.
        
    Returns:
        List of modifier strings.
    """
    if categories is None:
        categories = ["style", "lighting", "mood", "color_palette", "atmosphere"]
    
    modifiers_pool = []
    for cat in categories:
        if cat in MODIFIERS:
            modifiers_pool.extend(MODIFIERS[cat])
    
    return random.sample(modifiers_pool, min(num_modifiers, len(modifiers_pool)))

