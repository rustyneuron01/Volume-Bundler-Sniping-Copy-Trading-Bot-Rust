import os
from typing import Dict, List, Optional, Any
import bittensor as bt

from .base_service import BaseGenerationService
from .openai_service import OpenAIService
from .openrouter_service import OpenRouterService
from .stabilityai_service import StabilityAIService


SERVICE_MAP = {
    "openai": OpenAIService,
    "openrouter": OpenRouterService,
    "stabilityai": StabilityAIService,
    #"local": LocalService,
}


class ServiceRegistry:
    """
    Registry for managing generation services.

    Set per-modality service via env vars:
      IMAGE_SERVICE=openai|openrouter|stabilityai|none
      VIDEO_SERVICE=openai|openrouter|stabilityai|none

    Services:
      - openai: DALL-E 3 (requires OPENAI_API_KEY) - produces C2PA-signed content
      - openrouter: Google Gemini via OpenRouter (requires OPEN_ROUTER_API_KEY) - produces C2PA-signed content
      - stabilityai: Stability AI (requires STABILITY_API_KEY) - produces C2PA-signed content
      - none: Disable this modality (requests will be rejected)

    If not set, falls back to loading all available services.
    All services must produce C2PA-signed content for validator verification.
    """

    def __init__(self, config: Any = None):
        self.config = config
        self.services: Dict[str, BaseGenerationService] = {}  # modality -> service
        self._all_services: List[BaseGenerationService] = []  # fallback list
        self._initialize_services()

    def _initialize_services(self):
        """Initialize services based on IMAGE_SERVICE and VIDEO_SERVICE env vars."""
        image_service = os.getenv("IMAGE_SERVICE", "").lower().strip()
        video_service = os.getenv("VIDEO_SERVICE", "").lower().strip()

        if image_service or video_service:
            self._init_modality_services(image_service, video_service)
        else:
            self._init_all_services()

    def _init_modality_services(self, image_service: str, video_service: str):
        """Initialize specific services for each modality."""
        initialized = set()

        if image_service == "none":
            bt.logging.info("IMAGE_SERVICE=none, no image service will be loaded")
        elif image_service:
            service = self._create_service(image_service, "image")
            if service:
                self.services["image"] = service
                initialized.add(image_service)

        if video_service == "none":
            bt.logging.info("VIDEO_SERVICE=none, no video service will be loaded")
        elif video_service:
            if video_service in initialized:
                # Reuse the same service instance for video
                self.services["video"] = self.services.get("image") or self._create_service(video_service, "video")
            else:
                service = self._create_service(video_service, "video")
                if service:
                    self.services["video"] = service

    def _create_service(self, service_name: str, modality: str) -> Optional[BaseGenerationService]:
        """Create and validate a service instance."""
        if service_name not in SERVICE_MAP:
            bt.logging.error(f"Unknown service: {service_name}. Valid options: {list(SERVICE_MAP.keys())}")
            return None
        bt.logging.info(f"Initializing {service_name} for {modality}")
        service_class = SERVICE_MAP[service_name]
        try:
            service = service_class(self.config)
            if service.is_available():
                bt.logging.success(f"✅ {service.name} ready for {modality}")
                return service
            else:
                bt.logging.error(f"❌ {service.name} configured for {modality} but not available (check API keys)")
        except Exception as e:
            bt.logging.error(f"Failed to initialize {service_name}: {e}")
        return None

    def _init_all_services(self):
        """Initialize all available services (fallback behavior)."""
        bt.logging.info("No IMAGE_SERVICE/VIDEO_SERVICE set, initializing all available services...")
        for name, service_class in SERVICE_MAP.items():
            try:
                service = service_class(self.config)
                if service.is_available():
                    self._all_services.append(service)
                    bt.logging.info(f"✅ {service.name} is available")
                else:
                    bt.logging.info(f"❌ {service.name} is not available")
            except Exception as e:
                bt.logging.warning(f"Failed to initialize {name}: {e}")
        bt.logging.info(f"Initialized {len(self._all_services)} generation services")

    def get_service(self, modality: str) -> Optional[BaseGenerationService]:
        """Get the service for a modality."""
        # Check explicit modality mapping first
        if modality in self.services:
            service = self.services[modality]
            bt.logging.debug(f"Using {service.name} for {modality}")
            return service

        # Fallback to scanning all services
        for service in self._all_services:
            if service.supports_modality(modality):
                bt.logging.debug(f"Using {service.name} for {modality}")
                return service

        bt.logging.warning(f"No service available for modality={modality}")
        return None

    def get_available_services(self) -> List[Dict[str, Any]]:
        """Get information about all available services."""
        seen = set()
        result = []

        for service in list(self.services.values()) + self._all_services:
            if service.name not in seen:
                seen.add(service.name)
                result.append(service.get_info())
        return result

    def get_all_api_key_requirements(self) -> Dict[str, str]:
        """Get API key requirements from all services."""
        all_requirements = {
            "IMAGE_SERVICE": "Service for images: openai, openrouter, stabilityai, or none",
            "VIDEO_SERVICE": "Service for videos: openai, openrouter, stabilityai, or none",
        }

        for name, service_class in SERVICE_MAP.items():
            try:
                temp_service = service_class()
                all_requirements.update(temp_service.get_api_key_requirements())
            except Exception as e:
                bt.logging.warning(f"Failed to get API key requirements from {name}: {e}")

        return all_requirements

    def reload_services(self):
        """Reload all services (useful for configuration changes)."""
        self.services.clear()
        self._all_services.clear()
        self._initialize_services()
