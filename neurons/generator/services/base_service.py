from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import bittensor as bt

from ..task_manager import GenerationTask


class BaseGenerationService(ABC):
    """
    Abstract base class for generation services.
    
    This defines the interface that all generation services must implement,
    whether they use 3rd party APIs or local models.
    """
    
    def __init__(self, config: Any = None):
        self.config = config
        self.name = self.__class__.__name__
        bt.logging.info(f"Initializing {self.name}")
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this service is available (e.g., API keys set, models loaded)."""
        pass
    
    @abstractmethod
    def supports_modality(self, modality: str) -> bool:
        """Check if this service supports the given modality (e.g., image, video)."""
        pass
    
    @abstractmethod
    def process(self, task: GenerationTask) -> Dict[str, Any]:
        """
        Process a generation task and return the result.
        
        Args:
            task: The generation task to process
            
        Returns:
            Dict containing:
            - 'data': Optional[bytes] - Binary result data (if generated locally)
            - 'url': Optional[str] - Direct URL to result (if from 3rd party)
            - 'metadata': Optional[Dict] - Any additional metadata
            
        Raises:
            Exception: If processing fails
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this service."""
        return {
            "name": self.name,
            "available": self.is_available(),
            "supported_tasks": self.get_supported_tasks()
        }
    
    @abstractmethod
    def get_supported_tasks(self) -> Dict[str, list]:
        """Return dict of supported task types by modality."""
        pass
    
    @abstractmethod
    def get_api_key_requirements(self) -> Dict[str, str]:
        """
        Return dict of required environment variables and their descriptions.
        
        Returns:
            Dict with env var names as keys and human-readable descriptions as values.
            Example: {
                'OPENAI_API_KEY': 'OpenAI API key for DALL-E image generation',
                'OPENAI_ORG_ID': 'OpenAI organization ID (optional)'
            }
        """
        pass
