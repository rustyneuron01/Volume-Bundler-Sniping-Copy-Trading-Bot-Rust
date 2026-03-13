from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, auto
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Union
import json


class NeuronType(Enum):
    VALIDATOR = "VALIDATOR"
    MINER = "MINER"


class MinerType(Enum):
    DISCRIMINATOR = "DISCRIMINATOR"
    GENERATOR = "GENERATOR"


class DiscriminatorType(Enum):
    DETECTOR = "DETECTOR"
    SEGMENTER = "SEGMENTER"


class FileType(Enum):
    PARQUET = auto()
    ZIP = auto()
    VIDEO = auto()
    IMAGE = auto()


class Modality(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class MediaType(str, Enum):
    REAL = "real"
    SYNTHETIC = "synthetic"
    SEMISYNTHETIC = "semisynthetic"

    @property
    def int_value(self):
        """Get the integer value for this media type"""
        mapping = {
            MediaType.REAL: 0,
            MediaType.SYNTHETIC: 1,
            MediaType.SEMISYNTHETIC: 2,
        }
        return mapping[self]


class SourceType(str, Enum):
    """Canonical source types for media provenance."""

    SCRAPER = "scraper"
    DATASET = "dataset"
    GENERATED = "generated"
    MINER = "miner"


SOURCE_TYPE_TO_NAME: Dict[SourceType, str] = {
    SourceType.GENERATED: "model_name",
    SourceType.DATASET: "dataset_name",
    SourceType.SCRAPER: "download_url",
    SourceType.MINER: "hotkey",
}


SOURCE_TYPE_TO_DB_NAME_FIELD: Dict[SourceType, str] = {
    SourceType.GENERATED: "model_name",
    SourceType.DATASET: "dataset_name",
    SourceType.SCRAPER: "scraper_name",
    SourceType.MINER: "hotkey",
}


@dataclass
class DatasetConfig:
    """For datasets used by the Validator"""
    path: str  # HuggingFace path
    modality: Modality
    media_type: MediaType
    file_format: str = ""
    source_format: str = ""
    priority: int = 1  # Optional: priority for sampling (higher is more frequent)
    enabled: bool = True

    def __post_init__(self):
        """Validate and set defaults"""
        if not self.source_format:
            if self.modality == Modality.IMAGE:
                self.source_format = "parquet"
            elif self.modality == Modality.VIDEO:
                self.source_format = "zip"

        if isinstance(self.modality, str):
            self.modality = Modality(self.modality.lower())

        if isinstance(self.media_type, str):
            self.media_type = MediaType(self.media_type.lower())


class ModelTask(str, Enum):
    "Tasks supported by validator models"
    TEXT_TO_IMAGE = "t2i"
    TEXT_TO_VIDEO = "t2v"
    IMAGE_TO_IMAGE = "i2i"
    IMAGE_TO_VIDEO = "i2v"


class ModelConfig:
    """
    Configuration for a generative AI model.

    Attributes:
        path: The Hugging Face model path or identifier
        task: The primary task of the model (T2I, T2V, I2I)
        media_type: Type of output (synthetic or semisynthetic)
        pipeline_cls: Pipeline class used to load the model
        pretrained_args: Arguments for the from_pretrained method
        generation_args: Default arguments for generation
        tags: List of tags for categorizing the model
        use_autocast: Whether to use autocast during generation
        scheduler: Optional scheduler configuration
        scheduler_cls: Optional scheduler class
        scheduler_args: Optional scheduler args
    """

    def __init__(
        self,
        path: str,
        task: ModelTask,
        pipeline_cls: Union[Any, Dict[str, Any]],
        media_type: Optional[MediaType] = None,
        pretrained_args: Dict[str, Any] = None,
        generation_args: Dict[str, Any] = None,
        tags: List[str] = None,
        use_autocast: bool = True,
        enable_model_cpu_offload: bool = False,
        enable_sequential_cpu_offload: bool = False,
        vae_enable_slicing: bool = False,
        vae_enable_tiling: bool = False,
        scheduler: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None,
        pipeline_stages: List[Dict[str, Any]] = None,
        clear_memory_on_stage_end: bool = False,
        lora_model_id: str = None,
        lora_loading_args: Dict[str, Any] = None,
    ):
        self.path = path
        self.task = task
        self.pipeline_cls = pipeline_cls
        self.media_type = media_type

        if self.media_type is None:
            self.media_type = (
                MediaType.SEMISYNTHETIC
                if task == ModelTask.IMAGE_TO_IMAGE
                else MediaType.SYNTHETIC
            )

        self.pretrained_args = pretrained_args or {}
        self.generation_args = generation_args or {}
        self.tags = tags or []
        self.use_autocast = use_autocast
        self.enable_model_cpu_offload = enable_model_cpu_offload
        self.enable_sequential_cpu_offload = enable_sequential_cpu_offload
        self.vae_enable_slicing = vae_enable_slicing
        self.vae_enable_tiling = vae_enable_tiling
        self.scheduler = scheduler
        self.save_args = save_args or {}
        self.pipeline_stages = pipeline_stages
        self.clear_memory_on_stage_end = clear_memory_on_stage_end
        self.lora_model_id = lora_model_id
        self.lora_loading_args = lora_loading_args

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary format"""
        return {
            "pipeline_cls": self.pipeline_cls,
            "from_pretrained_args": self.pretrained_args,
            "generation_args": self.generation_args,
            "use_autocast": self.use_autocast,
            "enable_model_cpu_offload": self.enable_model_cpu_offload,
            "enable_sequential_cpu_offload": self.enable_sequential_cpu_offload,
            "vae_enable_slicing": self.vae_enable_slicing,
            "vae_enable_tiling": self.vae_enable_tiling,
            "scheduler": self.scheduler,
            "save_args": self.save_args,
            "pipeline_stages": self.pipeline_stages,
            "clear_memory_on_stage_end": self.clear_memory_on_stage_end,
        }


# =============================================================================
# MODEL MANAGEMENT TYPES
# =============================================================================

@dataclass
class DiscriminatorModelId:
    """Unique identifier for a discriminator model"""

    key: str  # Remote storage key
    hash: str  # Model dir hash

    def __post_init__(self):
        """Validate and truncate commit and hash to 16 chars"""
        if self.hash and len(self.hash) > 16:
            self.hash = self.hash[:16]

    def to_compressed_str(self) -> str:
        """Convert to compressed string for chain storage"""
        data = {
            "key": self.key,
            "hash": self.hash,
        }
        return json.dumps(data, separators=(",", ":"))

    @classmethod
    def from_compressed_str(cls, compressed_str: str) -> "DiscriminatorModelId":
        """Create DiscriminatorModelId from compressed string"""
        data = json.loads(compressed_str)
        return cls(
            key=data["key"],
            hash=data["hash"],
        )

    def __eq__(self, other):
        if not isinstance(other, DiscriminatorModelId):
            return False
        return (
            self.key == other.key
            and self.hash == other.hash
        )


@dataclass
class DiscriminatorModelMetadata:
    """Metadata for a discriminator model stored on chain"""

    id: DiscriminatorModelId
    block: int  # Block number when metadata was stored

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "id": {
                "key": self.id.key,
                "hash": self.id.hash,
            },
            "block": self.block,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DiscriminatorModelMetadata":
        """Create DiscriminatorModelMetadata from dictionary"""
        model_id = DiscriminatorModelId(
            key=data["id"]["key"],
            hash=data["id"]["hash"],
        )
        return cls(id=model_id, block=data["block"])


class ValidatorConfig(BaseModel):
    skip_weight_set: Optional[bool] = False
    set_weights_on_start: Optional[bool] = False
    max_concurrent_organics: Optional[int] = 2
