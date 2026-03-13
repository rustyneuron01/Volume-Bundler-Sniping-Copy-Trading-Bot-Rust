import asyncio
import base64
import hashlib
import json
import random
import os
import traceback
import tempfile
import shutil
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Dict, Any, Generator, Tuple
from zipfile import ZipFile
import tarfile
from contextlib import closing

import bittensor as bt
import numpy as np
import pyarrow.parquet as pq
from PIL import Image


import requests
from gas.types import Modality, MediaType, DatasetConfig


IMAGE_FILE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
VIDEO_FILE_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v"}


def download_and_extract(
    dataset: DatasetConfig,
    images_per_parquet: int = 100,
    videos_per_zip: int = 50,
    parquet_per_dataset: int = 5,
    zips_per_dataset: int = 2,
    temp_dir: Optional[str] = None,
) -> Generator[Tuple[bytes, Dict[str, Any]], None, None]:
    """
    Download datasets and yield extracted media as a generator.

    This approach uses temp files to avoid storing large zip/parquet files
    locally while ensuring diverse media availability.

    Args:
        datasets: List of DatasetConfig objects
        exclude_tags: Tags to exclude from download

    Yields:
        Tuples of (media_bytes, metadata_dict)
    """
    try:
        # temp dir for zip/parquet
        temp_dir_root = None
        if temp_dir is not None:
            try:
                Path(temp_dir).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            temp_dir_root = Path(tempfile.mkdtemp(dir=temp_dir))
        else:
            temp_dir_root = Path(tempfile.mkdtemp())

        try:
            filenames = _list_remote_dataset_files(dataset.path, dataset.source_format)
            if not filenames:
                bt.logging.warning(f"No files found for {dataset.path}")
                return None, None

            remote_paths = _get_download_urls(dataset.path, filenames)

            src_fmt = str(getattr(dataset, "source_format", "")).lower().lstrip(".")
            if dataset.modality == Modality.IMAGE:
                # For direct image datasets, download as many files as we would have extracted
                # from parquets (images_per_parquet * parquet_per_dataset).
                image_exts_nodot = {ext.lstrip('.') for ext in IMAGE_FILE_EXTENSIONS}
                is_direct_image_format = src_fmt in image_exts_nodot
                n_files = (images_per_parquet * parquet_per_dataset) if is_direct_image_format else parquet_per_dataset
            else:
                # For direct mp4 datasets, download as many files as we would have extracted 
                # from archives (zips_per_dataset * videos_per_zip).
                if src_fmt == "mp4":
                    n_files = zips_per_dataset * videos_per_zip
                elif src_fmt == "parquet":
                    n_files = parquet_per_dataset
                else:
                    n_files = zips_per_dataset

            to_download = _select_files_to_download(remote_paths, n_files)

            bt.logging.debug(
                f"Downloading {len(to_download)} files from {dataset.path}"
            )
            downloaded_files = download_files(to_download, temp_dir_root)

            for source_file in downloaded_files:
                num_items = images_per_parquet if dataset.modality == Modality.IMAGE else videos_per_zip
                for media_obj, metadata in yield_media_from_source(source_file, dataset, num_items):
                    yield media_obj, metadata

        finally:
            if temp_dir_root.exists():
                shutil.rmtree(temp_dir_root)

    except Exception as e:
        bt.logging.error(f"Error processing {dataset.path}: {e}")


def yield_media_from_source(
    source_path: Path,
    dataset: DatasetConfig,
    num_items: int,
) -> Generator[Tuple[Any, Dict[str, Any]], None, None]:
    """
    Unified media extractor for parquet, zip, and tar sources.

    Returns only the common metadata fields across all previous extractors:
    dataset, dataset_path, dataset_priority, modality, media_type
    """
    try:
        filename = str(source_path.name).lower()

        if _is_parquet_file(filename):
            yield from _process_parquet(source_path, dataset, num_items)
            return

        if _is_zip_file(filename) or _is_tar_file(filename):
            yield from _process_zip_or_tar(source_path, dataset, num_items)
            return

        if any(filename.endswith(ext) for ext in (IMAGE_FILE_EXTENSIONS | VIDEO_FILE_EXTENSIONS)):
            yield from _process_raw(source_path, dataset)
            return

        bt.logging.warning(f"Unsupported source format for {source_path}")
        return
    except Exception as e:
        bt.logging.warning(f"Error in yield_media_from_source for {source_path}: {e}")
        return


def _select_files_to_download(urls: List[str], count: int) -> List[str]:
    """Select random files to download"""
    return random.sample(urls, min(count, len(urls)))


def _list_remote_dataset_files(
    dataset_path: str, source_format: str = ".parquet"
) -> List[str]:
    """List available files in a dataset, filtered by source_format.

    Supports single extensions (e.g., .parquet, .zip) and tar variants (.tar, .tar.gz, .tgz).
    """
    if not source_format.startswith("."):
        source_format = "." + source_format

    if source_format == ".tar":
        extensions = [".tar", ".tar.gz", ".tgz"]
        return list_hf_files(repo_id=dataset_path, extension=extensions)

    return list_hf_files(repo_id=dataset_path, extension=source_format)


def _get_download_urls(dataset_path: str, filenames: List[str]) -> List[str]:
    """Get Hugging Face download URLs for data files"""
    return [
        f"https://huggingface.co/datasets/{dataset_path}/resolve/main/{f}"
        for f in filenames
    ]


def _common_metadata(dataset: DatasetConfig, source_path: Path) -> Dict[str, Any]:
    return {
        "dataset": source_path.parent.name,
        "dataset_path": dataset.path,
        "dataset_priority": dataset.priority,
        "modality": dataset.modality,
        "media_type": dataset.media_type,
    }


def _process_parquet(source_path: Path, dataset: DatasetConfig, num_items: int):
    table = pq.read_table(source_path)
    df = table.to_pandas()
    sample_df = df.sample(n=min(num_items, len(df)))

    if dataset.modality == Modality.IMAGE:
        media_col = (
            next((c for c in sample_df.columns if c.lower() == "image"), None)
            or next((c for c in sample_df.columns if "image" in c.lower() and "_id" not in c.lower() and "width" not in c.lower() and "height" not in c.lower()), None)
            or next((c for c in sample_df.columns if "image" in c.lower()), None)
        )
    else:
        candidates = ["video", "bytes", "content", "data"]
        media_col = (
            next((c for c in sample_df.columns if c.lower() in candidates), None)
            or next((c for c in sample_df.columns if any(k in c.lower() for k in candidates) and "_id" not in c.lower()), None)
            or next((c for c in sample_df.columns if any(k in c.lower() for k in candidates)), None)
        )

    if not media_col:
        bt.logging.warning(f"No media column found in {source_path} for modality {dataset.modality}")
        return

    bt.logging.debug(f"Selected media column '{media_col}' from {source_path}")

    for _, row in sample_df.iterrows():
        try:
            media_data = row[media_col]
            
            if media_data is None or isinstance(media_data, (int, float)):
                continue
            
            if isinstance(media_data, dict):
                key = next(
                    (
                        k for k in media_data 
                        if any(s in k.lower() 
                        for s in ["bytes", "image", "video", "data", "content"])
                    ), None
                )
                media_data = media_data[key]

            if dataset.modality == Modality.IMAGE:
                try:
                    img = Image.open(BytesIO(media_data))
                except Exception:
                    media_data = base64.b64decode(media_data)
                    img = Image.open(BytesIO(media_data))
                yield img, _common_metadata(dataset, source_path)
            else:
                if media_data is None or isinstance(media_data, (int, float)):
                    continue
                if not isinstance(media_data, (bytes, bytearray)):
                    media_data = base64.b64decode(media_data)
                yield bytes(media_data), _common_metadata(dataset, source_path)
        except Exception as e:
            bt.logging.warning(f"Failed to extract row from {source_path}: {e}")
            continue


def _process_zip_or_tar(source_path: Path, dataset: DatasetConfig, num_items: int):
    filename = str(source_path.name).lower()
    is_zip = _is_zip_file(filename)
    try:
        cm = (ZipFile(source_path) if is_zip else tarfile.open(source_path, mode="r:*"))
        with cm as archive:
            valid_exts = IMAGE_FILE_EXTENSIONS if dataset.modality == Modality.IMAGE else VIDEO_FILE_EXTENSIONS

            if is_zip:
                list_entries = archive.namelist()
                get_name = lambda e: e
                def open_entry(e):
                    return archive.open(e)
            else:
                list_entries = [m for m in archive.getmembers() if m.isreg()]
                get_name = lambda m: m.name
                def open_entry(m):
                    return archive.extractfile(m)

            candidates = [
                e for e in list_entries
                if any(get_name(e).lower().endswith(ext) for ext in valid_exts)
                and "MACOSX" not in get_name(e)
            ]
            if not candidates:
                bt.logging.warning(f"No matching files found in {source_path}")
                return

            selected = random.sample(candidates, min(num_items, len(candidates)))

            for entry in selected:
                try:
                    src = open_entry(entry)
                    if src is None:
                        continue
                    with closing(src):
                        data_bytes = src.read()

                    if dataset.modality == Modality.IMAGE:
                        try:
                            media_obj = Image.open(BytesIO(data_bytes))
                        except Exception:
                            bt.logging.warning(f"Failed to open image {get_name(entry)} from {source_path}")
                            continue
                    else:
                        media_obj = data_bytes

                    yield media_obj, _common_metadata(dataset, source_path)
                except Exception as e:
                    bt.logging.warning(f"Error extracting {get_name(entry)} from {source_path}: {e}")
                    continue
    except Exception as e:
        bt.logging.warning(f"Error processing archive file {source_path}: {e}")
        return


def _process_raw(source_path: Path, dataset: DatasetConfig):
    filename = str(source_path.name).lower()
    try:
        data_bytes = source_path.read_bytes()
        if dataset.modality == Modality.IMAGE and any(filename.endswith(ext) for ext in IMAGE_FILE_EXTENSIONS):
            media_obj = Image.open(BytesIO(data_bytes))
        elif dataset.modality == Modality.VIDEO and any(filename.endswith(ext) for ext in VIDEO_FILE_EXTENSIONS):
            media_obj = data_bytes
        else:
            bt.logging.warning(f"Direct file {source_path} does not match modality {dataset.modality}")
            return

        yield media_obj, _common_metadata(dataset, source_path)
    except Exception as e:
        bt.logging.warning(f"Error reading direct file {source_path}: {e}")
        return


def list_hf_files(repo_id, repo_type="dataset", extension=None):
    """List files from a Hugging Face repository.

    Args:
        repo_id: Repository ID
        repo_type: Type of repository ('dataset', 'model', etc.)
        extension: Filter files by extension

    Returns:
        List of files in the repository
    """
    files = []
    try:
        import huggingface_hub as hf_hub

        files = list(hf_hub.list_repo_files(repo_id=repo_id, repo_type=repo_type))
        if extension:
            if isinstance(extension, (list, tuple, set)):
                exts = tuple(extension)
                files = [f for f in files if f.endswith(exts)]
            else:
                files = [f for f in files if f.endswith(extension)]
    except Exception as e:
        bt.logging.error(f"Failed to list files of type {extension} in {repo_id}: {e}")
    return files


def _is_zip_file(filename_lower: str) -> bool:
    """Return True if filename looks like a zip archive."""
    return filename_lower.endswith(".zip")


def _is_tar_file(filename_lower: str) -> bool:
    """Return True if filename looks like a tar archive (.tar, .tar.gz, .tgz)."""
    return filename_lower.endswith(".tar") or filename_lower.endswith(".tar.gz") or filename_lower.endswith(".tgz")


def _is_parquet_file(filename_lower: str) -> bool:
    """Return True if filename looks like a parquet file."""
    return filename_lower.endswith(".parquet")


def download_files(
    urls: List[str], output_dir: Path, chunk_size: int = 8192
) -> List[Path]:
    """Download multiple files synchronously.

    Args:
        urls: List of URLs to download
        output_dir: Directory to save the files
        chunk_size: Size of chunks to download at a time

    Returns:
        List of successfully downloaded file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files = []

    for url in urls:
        try:
            downloaded_file = download_single_file(url, output_dir, chunk_size)
            if downloaded_file:
                downloaded_files.append(downloaded_file)
        except Exception as e:
            bt.logging.error(f"Error downloading {url}: {e}")

    return downloaded_files


def download_single_file(
    url: str, output_dir: Path, chunk_size: int
) -> Optional[Path]:
    """Download a single file synchronously.

    Args:
        url: URL to download
        output_dir: Directory to save the file
        chunk_size: Size of chunks to download at a time

    Returns:
        Path to the downloaded file, or None if failed
    """
    try:
        bt.logging.info(f"Downloading {url}")

        response = requests.get(url, stream=True, timeout=3600)
        if response.status_code != 200:
            bt.logging.error(f"Failed to download {url}: Status {response.status_code}")
            return None

        filename = os.path.basename(url)
        filepath = output_dir / filename

        bt.logging.trace(f"Writing to {filepath}")

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)

        return filepath

    except Exception as e:
        bt.logging.error(f"Error downloading {url}: {str(e)}")
        bt.logging.error(traceback.format_exc())
        return None
