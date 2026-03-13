"""
Dataset definitions from gasbench for the validator cache system 
"""

from typing import List
from gas.types import DatasetConfig, MediaType, Modality
from gasbench.dataset.config import load_benchmark_datasets_from_yaml
 


def _from_benchmark_config(bc) -> DatasetConfig:
    """
    Map gasbench BenchmarkDatasetConfig -> bitmind-subnet DatasetConfig
    """
    return DatasetConfig(
        path=bc.path,
        modality=bc.modality,          # coerced to Modality in DatasetConfig.__post_init__
        media_type=bc.media_type,      # coerced to MediaType in DatasetConfig.__post_init__
        source_format=(bc.source_format or ""),
    )


def _load_yaml_datasets(modality: str) -> List[DatasetConfig]:
    """
    Load datasets from gasbench's bundled benchmark_datasets.yaml and map them.
    Filters out unsupported sources for current downloader (e.g., gasstation/*, non-HF sources).
    """
    data = load_benchmark_datasets_from_yaml()
    bench_list = data.get(modality, []) or []

    mapped: List[DatasetConfig] = []
    for bc in bench_list:
        # Exclude gasstation paths and non-HuggingFace sources (bitmind-subnet downloader only supports HF)
        try:
            source = getattr(bc, "source", "huggingface")
            path = getattr(bc, "path", "") or ""
            if source != "huggingface":
                continue
            if path.startswith("gasstation/"):
                continue
            mapped.append(_from_benchmark_config(bc))
        except Exception:
            continue

    return mapped


def get_image_datasets() -> List[DatasetConfig]:
    return _load_yaml_datasets("image")


def get_video_datasets() -> List[DatasetConfig]:
    return _load_yaml_datasets("video")


def load_all_datasets() -> List[DatasetConfig]:
    return get_image_datasets() + get_video_datasets()
