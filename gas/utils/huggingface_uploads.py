import os
import time
import json
import tempfile
import tarfile
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone

import bittensor as bt
import pandas as pd
from huggingface_hub import HfApi, create_repo, CommitOperationAdd
from datasets import Dataset, Features, Value, Image as HFImage
from PIL import Image

from gas.utils.model_zips import calculate_sha256


def get_current_time_split() -> str:
    """
    Uses ISO week format for subset/config names: 2025W03, 2025W04, etc.

    This creates weekly dataset subsets/configs, not splits.
    Users load them like: load_dataset('repo', '2025W36')

    Note: HF split names must match ^\w+(\.\w+)*$ (no hyphens allowed)

    Returns:
        str: Weekly subset/config name (e.g., "2025W03")
    """
    now = datetime.now(timezone.utc)
    year = now.year
    iso_week = now.isocalendar()[1]  # (1-53)
    return f"{year}W{iso_week:02d}"


def upload_images_to_hf(
    media_entries: List[Any],
    hf_token: str,
    dataset_repo: str,
    images_per_archive: int,
    validator_hotkey: str = None,
    validator_uid: int = None,
) -> List[str]:
    """Upload images and metadata to hf"""
    return upload_media_to_hf(
        media_entries,
        hf_token,
        dataset_repo,
        "image",
        images_per_archive=images_per_archive,
        validator_hotkey=validator_hotkey,
        validator_uid=validator_uid,
    )


def upload_videos_to_hf(
    media_entries: List[Any],
    hf_token: str,
    dataset_repo: str,
    videos_per_archive: int,
    validator_hotkey: str = None,
    validator_uid: int = None,
) -> List[str]:
    """Upload video .tar.gz files and metadata to hf"""
    return upload_media_to_hf(
        media_entries,
        hf_token,
        dataset_repo,
        "video",
        videos_per_archive=videos_per_archive,
        validator_hotkey=validator_hotkey,
        validator_uid=validator_uid,
    )


def upload_media_to_hf(
    media_entries: List[Any],
    hf_token: str,
    dataset_repo: str,
    modality: str,
    **kwargs
) -> List[str]:
    """
    Core hf dataset upload logic

    Returns:
        List of successfully uploaded media IDs
    """
    try:
        hf_api = HfApi(token=hf_token)

        bt.logging.info(f"Preparing to upload {len(media_entries)} {modality} files to {dataset_repo}")
        try:
            create_repo(
                repo_id=dataset_repo,
                repo_type="dataset",
                exist_ok=True,
                token=hf_token
            )
        except Exception as e:
            if (
                "429" in str(e)
                or "rate" in str(e).lower()
                or "too many requests" in str(e).lower()
            ):
                bt.logging.warning(
                    "Hit HuggingFace rate limit during repo creation. Stopping upload and will retry next cycle."
                )
                return []
            raise

        validator_hotkey = kwargs.get("validator_hotkey")
        validator_uid = kwargs.get("validator_uid")
        upload_files, metadata_entries, successfully_processed_ids = (
            _process_media_entries(media_entries, modality, validator_hotkey, validator_uid)
        )

        if not upload_files:
            bt.logging.warning(f"No valid {modality} files to upload in batch")
            return []

        if modality == "image":
            images_per_archive = kwargs.get("images_per_archive", 500)
            result = _prepare_image_operations(upload_files, metadata_entries, images_per_archive)
            if not result or result[0] is None:
                return []
            dataset, config_name, archive_uploads = result
        elif modality == "video":
            result = _prepare_video_dataset_and_archives(
                upload_files,
                metadata_entries,
                kwargs.get("videos_per_archive", 200),
                kwargs.get("validator_hotkey"),
            )
            if not result or result[0] is None:
                return []
            dataset, config_name, archive_uploads = result
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        try:
            create_repo(dataset_repo, repo_type="dataset", exist_ok=True, token=hf_token)
        except:
            pass

        archive_operations = []
        archive_temp_files = []
        for archive_path, _ in archive_uploads:
            archive_name = Path(archive_path).name
            archive_filename = f"archives/{config_name}/{archive_name}"
            archive_operations.append(
                CommitOperationAdd(path_in_repo=archive_filename, path_or_fileobj=str(archive_path))
            )
            archive_temp_files.append(archive_path)

        parquet_operations, parquet_temp_files = _prepare_shard_operations(
            dataset, config_name, kwargs.get("validator_hotkey")
        )

        all_operations = parquet_operations + archive_operations
        all_temp_files = parquet_temp_files + archive_temp_files

        if all_operations:
            commit_info = hf_api.create_commit(
                repo_id=dataset_repo,
                repo_type="dataset",
                operations=all_operations,
                commit_message=f"Add {modality} parquet shard and {len(archive_uploads)} archives to {config_name}",
            )
            bt.logging.info(
                f"Successfully uploaded {len(archive_uploads)} {modality} archives and "
                f"1 parquet shard ({len(dataset)} entries) to {dataset_repo} "
                f"config {config_name} (commit: {commit_info.oid[:8]})"
            )

            for temp_file in all_temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass

        return successfully_processed_ids

    except Exception as e:
        if (
            "429" in str(e)
            or "rate" in str(e).lower()
            or "too many requests" in str(e).lower()
        ):
            bt.logging.warning(
                f"Hit HuggingFace rate limit during {modality} upload. "
                "Stopping upload and will retry next cycle."
            )
            return []
        bt.logging.error(f"Failed to upload {modality} files: {e}")
        return []


def _process_media_entries(
    media_entries: List[Any], modality: str, validator_hotkey: str = None, validator_uid: int = None
) -> Tuple[List[Tuple], List[Dict], List[str]]:
    """Process media entries into upload files, metadata, and success IDs.
    
    Args:
        media_entries: List of media entry objects
        modality: "image" or "video"
        validator_hotkey: Validator hotkey running the upload
        validator_uid: Validator UID on the network
    """
    upload_files = []
    metadata_entries = []
    successfully_processed_ids = []

    for media_entry in media_entries:
        media_path = Path(media_entry.file_path)
        if not media_path.exists():
            bt.logging.warning(f"{modality.title()} file not found: {media_path}")
            continue

        file_extension = media_path.suffix
        timestamp = int(media_entry.created_at) if hasattr(media_entry, 'created_at') and media_entry.created_at else int(time.time())
        filename = f"{timestamp}_{media_entry.id[:8]}{file_extension}"
        repo_path = f"{modality}s/{filename}"

        upload_files.append((media_path, repo_path))

        with open(media_path, "rb") as f:
            content = f.read()
            media_hash = calculate_sha256(content)[:16]

        prompt_content = getattr(media_entry, "prompt_content", "") or ""
        
        # Extract resolution (tuple or JSON string)
        resolution_str = "null"
        if hasattr(media_entry, 'resolution') and media_entry.resolution:
            if isinstance(media_entry.resolution, (list, tuple)):
                resolution_str = json.dumps(list(media_entry.resolution))
            else:
                resolution_str = str(media_entry.resolution)

        # Determine generator type and info based on source_type
        source_type_value = media_entry.source_type.value if hasattr(media_entry.source_type, 'value') else str(media_entry.source_type)
        is_miner_generated = source_type_value == "miner"
        
        if is_miner_generated:
            generator_type = "miner"
            generator_hotkey = getattr(media_entry, "hotkey", None) or "unknown"
            generator_uid = getattr(media_entry, "uid", None) or 0
        else:
            generator_type = "validator"
            generator_hotkey = validator_hotkey or "unknown"
            generator_uid = validator_uid if validator_uid is not None else 0

        metadata_entries.append(
            {
                "filename": filename,
                "media_id": media_entry.id,
                "media_hash": media_hash,
                "validator_hotkey": validator_hotkey or "unknown",
                "validator_uid": validator_uid if validator_uid is not None else 0,
                "generator_hotkey": generator_hotkey,
                "generator_uid": generator_uid,
                "generator_type": generator_type,
                "model_name": media_entry.model_name or "unknown",
                "c2pa_issuer": getattr(media_entry, "c2pa_issuer", None) or "",
                "prompt_id": media_entry.prompt_id or "",
                "prompt_content": prompt_content,
                "modality": media_entry.modality.value,
                "media_type": media_entry.media_type.value,
                "source_type": source_type_value,
                "format": getattr(media_entry, "format", file_extension.lstrip('.')).upper(),
                "resolution": resolution_str,
                "file_size": getattr(media_entry, "file_size", len(content)),
                "verified": getattr(media_entry, "verified", False) or False,
                "timestamp": timestamp,
                "upload_timestamp": int(time.time()),
                "week_partition": get_current_time_split(),
            }
        )

        successfully_processed_ids.append(media_entry.id)

    return upload_files, metadata_entries, successfully_processed_ids


def _prepare_image_operations(
    upload_files: List[Tuple], metadata_entries: List[Dict], images_per_archive: int = 500
) -> tuple:
    """Create image tar archives and metadata - unified schema with videos"""
    time_split = get_current_time_split()
    bt.logging.info(f"Creating image archives for config: {time_split}")
    
    archive_uploads = create_image_archives(
        upload_files, metadata_entries, images_per_archive
    )
    
    if not archive_uploads:
        bt.logging.warning("No image archives created")
        return None, None, []
    
    all_metadata = []
    for archive_path, archive_metadata_list in archive_uploads:
        all_metadata.extend(archive_metadata_list)
    
    dataset_dict = {
        "media_id": [],
        "media_hash": [],
        "archive_filename": [],
        "file_path_in_archive": [],
        "validator_hotkey": [],
        "validator_uid": [],
        "generator_hotkey": [],
        "generator_uid": [],
        "generator_type": [],
        "model_name": [],
        "c2pa_issuer": [],
        "prompt_id": [],
        "prompt_content": [],
        "modality": [],
        "media_type": [],
        "source_type": [],
        "format": [],
        "resolution": [],
        "file_size": [],
        "verified": [],
        "timestamp": [],
        "upload_timestamp": [],
        "week_partition": [],
    }
    
    for metadata in all_metadata:
        dataset_dict["media_id"].append(metadata["media_id"])
        dataset_dict["media_hash"].append(metadata["media_hash"])
        dataset_dict["archive_filename"].append(metadata["archive_filename"])
        dataset_dict["file_path_in_archive"].append(metadata["file_path_in_archive"])
        dataset_dict["validator_hotkey"].append(metadata["validator_hotkey"])
        dataset_dict["validator_uid"].append(metadata["validator_uid"])
        dataset_dict["generator_hotkey"].append(metadata["generator_hotkey"])
        dataset_dict["generator_uid"].append(metadata["generator_uid"])
        dataset_dict["generator_type"].append(metadata["generator_type"])
        dataset_dict["model_name"].append(metadata["model_name"])
        dataset_dict["c2pa_issuer"].append(metadata["c2pa_issuer"])
        dataset_dict["prompt_id"].append(metadata["prompt_id"])
        dataset_dict["prompt_content"].append(metadata["prompt_content"])
        dataset_dict["modality"].append(metadata["modality"])
        dataset_dict["media_type"].append(metadata["media_type"])
        dataset_dict["source_type"].append(metadata["source_type"])
        dataset_dict["format"].append(metadata["format"])
        dataset_dict["resolution"].append(metadata["resolution"])
        dataset_dict["file_size"].append(metadata["file_size"])
        dataset_dict["verified"].append(metadata["verified"])
        dataset_dict["timestamp"].append(metadata["timestamp"])
        dataset_dict["upload_timestamp"].append(metadata["upload_timestamp"])
        dataset_dict["week_partition"].append(metadata["week_partition"])
    
    if not dataset_dict["media_id"]:
        bt.logging.warning("No valid image metadata to upload")
        return None, None, []
    
    features = Features({
        "media_id": Value("string"),
        "media_hash": Value("string"),
        "archive_filename": Value("string"),
        "file_path_in_archive": Value("string"),
        "validator_hotkey": Value("string"),
        "validator_uid": Value("int64"),
        "generator_hotkey": Value("string"),
        "generator_uid": Value("int64"),
        "generator_type": Value("string"),
        "model_name": Value("string"),
        "c2pa_issuer": Value("string"),
        "prompt_id": Value("string"),
        "prompt_content": Value("string"),
        "modality": Value("string"),
        "media_type": Value("string"),
        "source_type": Value("string"),
        "format": Value("string"),
        "resolution": Value("string"),
        "file_size": Value("int64"),
        "verified": Value("bool"),
        "timestamp": Value("int64"),
        "upload_timestamp": Value("int64"),
        "week_partition": Value("string"),
    })
    
    dataset = Dataset.from_dict(dataset_dict, features=features)
    bt.logging.info(f"Created metadata with {len(dataset)} image entries")
    
    return (dataset, time_split, archive_uploads)


def _prepare_shard_operations(
    dataset: Dataset, config_name: str, validator_hotkey: str = None, shard_suffix: str = ""
) -> tuple:
    """Create parquet shard operations"""
    timestamp = int(time.time())
    split_name = f"data_{config_name}"

    if validator_hotkey:
        hotkey_prefix = validator_hotkey[:8] if len(validator_hotkey) >= 8 else validator_hotkey
        shard_filename = f"{split_name}/shard-{hotkey_prefix}-{timestamp}{shard_suffix}.parquet"
    else:
        shard_filename = f"{split_name}/shard-{timestamp}{shard_suffix}.parquet"

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
        temp_path = tmp_file.name

    dataset.to_parquet(temp_path)
    operation = CommitOperationAdd(path_in_repo=shard_filename, path_or_fileobj=temp_path)

    bt.logging.info(f"Created parquet shard: {shard_filename} with {len(dataset)} entries")
    return [operation], [temp_path]


def _prepare_video_dataset_and_archives(
    upload_files: List[Tuple],
    metadata_entries: List[Dict],
    videos_per_archive: int,
    validator_hotkey: str = None,
) -> tuple:
    """Create video tar archives and metadata - unified schema with images"""
    time_split = get_current_time_split()
    bt.logging.info(f"Creating video archives for config: {time_split}")
    
    archive_uploads = create_video_archives(
        upload_files, metadata_entries, videos_per_archive, validator_hotkey
    )

    if not archive_uploads:
        bt.logging.warning("No video archives created")
        return None, None, []
    
    all_metadata = []
    for archive_path, archive_metadata_list in archive_uploads:
        all_metadata.extend(archive_metadata_list)
    
    dataset_dict = {
        "media_id": [],
        "media_hash": [],
        "archive_filename": [],
        "file_path_in_archive": [],
        "validator_hotkey": [],
        "validator_uid": [],
        "generator_hotkey": [],
        "generator_uid": [],
        "generator_type": [],
        "model_name": [],
        "c2pa_issuer": [],
        "prompt_id": [],
        "prompt_content": [],
        "modality": [],
        "media_type": [],
        "source_type": [],
        "format": [],
        "resolution": [],
        "file_size": [],
        "verified": [],
        "timestamp": [],
        "upload_timestamp": [],
        "week_partition": [],
    }
    
    for metadata in all_metadata:
        dataset_dict["media_id"].append(metadata["media_id"])
        dataset_dict["media_hash"].append(metadata["media_hash"])
        dataset_dict["archive_filename"].append(metadata["archive_filename"])
        dataset_dict["file_path_in_archive"].append(metadata["file_path_in_archive"])
        dataset_dict["validator_hotkey"].append(metadata["validator_hotkey"])
        dataset_dict["validator_uid"].append(metadata["validator_uid"])
        dataset_dict["generator_hotkey"].append(metadata["generator_hotkey"])
        dataset_dict["generator_uid"].append(metadata["generator_uid"])
        dataset_dict["generator_type"].append(metadata["generator_type"])
        dataset_dict["model_name"].append(metadata["model_name"])
        dataset_dict["c2pa_issuer"].append(metadata["c2pa_issuer"])
        dataset_dict["prompt_id"].append(metadata["prompt_id"])
        dataset_dict["prompt_content"].append(metadata["prompt_content"])
        dataset_dict["modality"].append(metadata["modality"])
        dataset_dict["media_type"].append(metadata["media_type"])
        dataset_dict["source_type"].append(metadata["source_type"])
        dataset_dict["format"].append(metadata["format"])
        dataset_dict["resolution"].append(metadata["resolution"])
        dataset_dict["file_size"].append(metadata["file_size"])
        dataset_dict["verified"].append(metadata["verified"])
        dataset_dict["timestamp"].append(metadata["timestamp"])
        dataset_dict["upload_timestamp"].append(metadata["upload_timestamp"])
        dataset_dict["week_partition"].append(metadata["week_partition"])
    
    if not dataset_dict["media_id"]:
        bt.logging.warning("No valid video metadata to upload")
        return None, None, []
    
    features = Features({
        "media_id": Value("string"),
        "media_hash": Value("string"),
        "archive_filename": Value("string"),
        "file_path_in_archive": Value("string"),
        "validator_hotkey": Value("string"),
        "validator_uid": Value("int64"),
        "generator_hotkey": Value("string"),
        "generator_uid": Value("int64"),
        "generator_type": Value("string"),
        "model_name": Value("string"),
        "c2pa_issuer": Value("string"),
        "prompt_id": Value("string"),
        "prompt_content": Value("string"),
        "modality": Value("string"),
        "media_type": Value("string"),
        "source_type": Value("string"),
        "format": Value("string"),
        "resolution": Value("string"),
        "file_size": Value("int64"),
        "verified": Value("bool"),
        "timestamp": Value("int64"),
        "upload_timestamp": Value("int64"),
        "week_partition": Value("string"),
    })
    
    dataset = Dataset.from_dict(dataset_dict, features=features)
    bt.logging.info(f"Created metadata with {len(dataset)} video entries")
    
    return (dataset, time_split, archive_uploads)


def create_image_archives(
    upload_files: List[Tuple[str, str]],
    metadata_entries: List[Dict[str, Any]],
    images_per_archive: int,
    validator_hotkey: str = None,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Create tar.gz archives containing images with their metadata"""
    if not upload_files:
        bt.logging.warning("No upload files provided for image archiving")
        return []

    if images_per_archive is None or images_per_archive <= 0:
        images_per_archive = 500

    if len(upload_files) != len(metadata_entries):
        raise ValueError(f"Mismatch between upload_files ({len(upload_files)}) and metadata_entries ({len(metadata_entries)})")

    archives = []
    temp_files_to_cleanup = []
    base_timestamp = int(time.time())

    bt.logging.info(f"Creating image archives with {images_per_archive} images per archive")

    try:
        for i in range(0, len(upload_files), images_per_archive):
            batch_files = upload_files[i : i + images_per_archive]
            batch_metadata = metadata_entries[i : i + images_per_archive]

            batch_index = i // images_per_archive
            timestamp = base_timestamp + batch_index

            if validator_hotkey:
                hotkey_prefix = validator_hotkey[:8] if len(validator_hotkey) >= 8 else validator_hotkey
                archive_filename = f"images_{hotkey_prefix}_{timestamp}.tar.gz"
            else:
                archive_filename = f"images_{timestamp}.tar.gz"

            temp_dir = tempfile.gettempdir()
            archive_path = os.path.join(temp_dir, archive_filename)
            temp_files_to_cleanup.append(archive_path)

            with tarfile.open(archive_path, "w:gz") as tar:
                archive_metadata_list = []

                for (local_path, repo_path), metadata in zip(batch_files, batch_metadata):
                    local_path = Path(local_path)
                    if not local_path.exists():
                        bt.logging.warning(f"Skipping missing image file: {local_path}")
                        continue

                    tar.add(str(local_path), arcname=repo_path)

                    archive_metadata = metadata.copy()
                    archive_metadata["archive_filename"] = Path(archive_path).name
                    archive_metadata["file_path_in_archive"] = repo_path
                    archive_metadata_list.append(archive_metadata)

                if archive_metadata_list:
                    archives.append((archive_path, archive_metadata_list))
                    temp_files_to_cleanup.remove(archive_path)
                    bt.logging.info(f"Created archive {Path(archive_path).name} with {len(archive_metadata_list)} images")

    except Exception as e:
        bt.logging.error(f"Error creating image archives: {e}")
        for temp_file in temp_files_to_cleanup:
            try:
                os.unlink(temp_file)
            except:
                pass
        raise

    if not archives:
        bt.logging.warning("No image archives created")
    else:
        bt.logging.info(f"Successfully created {len(archives)} image archives")

    return archives


def create_video_archives(
    upload_files: List[Tuple[str, str]],
    metadata_entries: List[Dict[str, Any]],
    videos_per_archive: int,
    validator_hotkey: str = None,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Create tar.gz archives containing videos with their metadata"""
    if not upload_files:
        bt.logging.warning("No upload files provided for video archiving")
        return []

    if videos_per_archive is None or videos_per_archive <= 0:
        videos_per_archive = 200

    if len(upload_files) != len(metadata_entries):
        raise ValueError(f"Mismatch between upload_files ({len(upload_files)}) and metadata_entries ({len(metadata_entries)})")

    archives = []
    temp_files_to_cleanup = []
    base_timestamp = int(time.time())

    bt.logging.info(f"Creating video archives with {videos_per_archive} videos per archive")

    try:
        for i in range(0, len(upload_files), videos_per_archive):
            batch_files = upload_files[i : i + videos_per_archive]
            batch_metadata = metadata_entries[i : i + videos_per_archive]

            batch_index = i // videos_per_archive
            timestamp = base_timestamp + batch_index

            if validator_hotkey:
                hotkey_prefix = validator_hotkey[:8] if len(validator_hotkey) >= 8 else validator_hotkey
                archive_filename = f"videos_{hotkey_prefix}_{timestamp}.tar.gz"
            else:
                archive_filename = f"videos_{timestamp}.tar.gz"

            temp_dir = tempfile.gettempdir()
            archive_path = os.path.join(temp_dir, archive_filename)
            temp_files_to_cleanup.append(archive_path)

            with tarfile.open(archive_path, "w:gz") as tar:
                archive_metadata_list = []

                for (local_path, repo_path), metadata in zip(batch_files, batch_metadata):
                    local_path = Path(local_path)
                    if not local_path.exists():
                        bt.logging.warning(f"Skipping missing video file: {local_path}")
                        continue

                    tar.add(str(local_path), arcname=repo_path)

                    archive_metadata = metadata.copy()
                    archive_metadata["archive_filename"] = Path(archive_path).name
                    archive_metadata["file_path_in_archive"] = repo_path
                    archive_metadata_list.append(archive_metadata)

                if archive_metadata_list:
                    archives.append((archive_path, archive_metadata_list))
                    temp_files_to_cleanup.remove(archive_path)
                    bt.logging.info(f"Created archive {Path(archive_path).name} with {len(archive_metadata_list)} videos")

    except Exception as e:
        bt.logging.error(f"Error creating video archives: {e}")
        for temp_file in temp_files_to_cleanup:
            try:
                os.unlink(temp_file)
            except:
                pass
        raise

    if not archives:
        bt.logging.warning("No video archives created")
    else:
        bt.logging.info(f"Successfully created {len(archives)} video archives")

    return archives
