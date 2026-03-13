import json
import random
import time
import uuid
from typing import Dict, Any, Optional, List
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from diffusers.utils import export_to_video

from gas.cache.util.video import sample_frames
from gas.cache.util import format_to_extension
from gas.cache.types import Media, MediaEntry
from gas.types import Modality, MediaType


class MediaStorage:
    """
    Filesystem storage system for media operations.
    Uses a flat directory structure: modality/media_type (e.g., images/synthetic, videos/real).
    All metadata (model names, datasets, etc.) is stored in the content database.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the media storage system.

        Args:
            base_dir: Base directory for storage (defaults to ~/.cache/sn34)
        """
        if base_dir is None:
            base_dir = Path("~/.cache/sn34").expanduser()

        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / "images"
        self.videos_dir = self.base_dir / "videos"
        for modality_dir in [self.images_dir, self.videos_dir]:
            for media_type in [MediaType.REAL.value, MediaType.SYNTHETIC.value, MediaType.SEMISYNTHETIC.value]:
                (modality_dir / media_type).mkdir(parents=True, exist_ok=True)

    def get_storage_path(self, modality: Modality) -> Path:
        """
        Args:
            modality: The modality (Modality.IMAGE or Modality.VIDEO)

        Returns:
            Path to the storage directory
        """
        if modality == Modality.IMAGE:
            return self.images_dir
        elif modality == Modality.VIDEO:
            return self.videos_dir
        else:
            raise ValueError(f"Unsupported modality: {modality}")

    def get_cache_path(self, modality: Modality, media_type: MediaType) -> Path:
        """
        Get the cache path for a specific modality and media type.
        This is used by scrapers to determine where to save files.

        Args:
            modality: The modality (Modality.IMAGE or Modality.VIDEO)
            media_type: The media type (MediaType.REAL, MediaType.SYNTHETIC, or MediaType.SEMISYNTHETIC)

        Returns:
            Path to the cache directory for this modality/type combination
        """
        storage_dir = self.get_storage_path(modality)
        return storage_dir / media_type.value

    def write_media(self, media_data: "Media") -> tuple[Optional[str], Optional[str]]:
        """
        Write media content to the filesystem.

        Args:
            media_data: Media object containing all necessary information

        Returns:
            Tuple of (save_path, mask_path) where both can be None if failed
        """
        try:
            # Step 1: Prepare file path
            storage_dir = self.get_storage_path(media_data.modality)
            output_dir = storage_dir / media_data.media_type.value

            timestamp = int(time.time())
            random_uuid = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
            base_filename = f"{timestamp}_{random_uuid}"

            extension = format_to_extension(media_data.format)
            save_path = output_dir / f"{base_filename}{extension}"

            # Step 2: Write the media content
            if media_data.modality == Modality.IMAGE:
                self._write_image_content(media_data, save_path)
            elif media_data.modality == Modality.VIDEO:
                self._write_video_content(media_data, save_path)
            else:
                raise ValueError(f"Unsupported modality: {media_data.modality}")

            # Step 3: Write mask if present
            mask_path = None
            if media_data.mask_content is not None:
                mask_path = output_dir / f"{base_filename}_mask.npy"
                mask_arr = (media_data.mask_content > 0).astype(np.uint8)
                np.save(str(mask_path), mask_arr)

            return str(save_path), str(mask_path) if mask_path else None

        except Exception as e:
            print(f"Error writing media: {e}")
            return None, None

    def _write_image_content(self, media_data: "Media", save_path: Path) -> None:
        """Write image content to file."""
        if hasattr(media_data.media_content, "save"):
            # PIL Image
            media_data.media_content.save(str(save_path), format=media_data.format)
        elif hasattr(media_data.media_content, "images"):
            # Diffusers pipeline output
            media_data.media_content.images[0].save(str(save_path), format=media_data.format)
        elif isinstance(media_data.media_content, bytes):
            # Raw image bytes (from miners) - write directly to file
            with open(save_path, 'wb') as f:
                f.write(media_data.media_content)
        else:
            # numpy array
            cv2.imwrite(str(save_path), media_data.media_content)

    def _write_video_content(self, media_data: "Media", save_path: Path) -> None:
        """Write video content to file."""
        if isinstance(media_data.media_content, bytes):
            # Raw video bytes (from datasets) - write directly to file
            with open(save_path, 'wb') as f:
                f.write(media_data.media_content)
        elif hasattr(media_data.media_content, "frames"):
            # Video frames from generators - use diffusers export_to_video
            export_to_video(media_data.media_content.frames[0], str(save_path), fps=30)
        else:
            # Video frames array (numpy array) - use diffusers export_to_video
            export_to_video(media_data.media_content, str(save_path), fps=30)



    def read_image(
        self,
        image_path: Path,
        as_float32: bool = False,
        channels_first: bool = False,
        as_rgb: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Read an image from the filesystem.

        Args:
            image_path: Path to the image file
            as_float32: Whether to return image as float32 (0-1) instead of uint8 (0-255)
            channels_first: Whether to return image with channels first (CHW) instead of channels last (HWC)
            as_rgb: Whether to return image in RGB format (True) or BGR format (False)

        Returns:
            Image as numpy array, or None if failed
        """
        try:
            if not image_path.exists():
                print(f"Image file not found: {image_path}")
                return None

            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Failed to load image: {image_path}")
                return None

            if as_rgb:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if as_float32:
                image = image.astype(np.float32) / 255.0

            if channels_first:
                image = np.transpose(image, (2, 0, 1))

            return image

        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            return None

    def read_mask(self, mask_path: Path) -> Optional[np.ndarray]:
        """
        Args:
            mask_path: Path to the mask file

        Returns:
            Mask as numpy array, or None if failed
        """
        try:
            if not mask_path.exists():
                print(f"Mask file not found: {mask_path}")
                return None

            return np.load(mask_path)

        except Exception as e:
            print(f"Error reading mask {mask_path}: {e}")
            return None

    def read_video_frames(
        self,
        video_path: Path,
        min_duration: float = 1.0,
        max_duration: float = 6.0,
        max_frames: int = 24,
    ) -> Optional[Dict[str, Any]]:
        """
        Args:
            video_path: Path to the video file
            min_duration: Minimum duration of video segment to extract in seconds
            max_duration: Maximum duration of video segment to extract in seconds
            max_frames: Maximum number of frames to extract

        Returns:
            Dictionary with video data, or None if failed
        """
        try:
            if not video_path.exists():
                print(f"Video file not found: {video_path}")
                return None

            return sample_frames(
                video_path=video_path,
                min_duration=min_duration,
                max_duration=max_duration,
                max_frames=max_frames,
            )

        except Exception as e:
            print(f"Error reading video {video_path}: {e}")
            return None

    def list_media_files(
        self,
        modality: Modality,
        file_extensions: Optional[List[str]] = None,
    ) -> List[Path]:
        """
        Args:
            modality: The modality ("image" or "video")
            file_extensions: List of file extensions to filter by

        Returns:
            List of file paths
        """
        storage_dir = self.get_storage_path(modality)

        if not storage_dir.exists():
            return []

        files = []
        for file_path in storage_dir.rglob("*"):
            if file_path.is_file():
                if (
                    file_extensions is None
                    or file_path.suffix.lower() in file_extensions
                ):
                    files.append(file_path)

        return files

    def delete_media_file(self, file_path: Path) -> bool:
        """
        Args:
            file_path: Path to the media file to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            if file_path.exists():
                file_path.unlink()

            if file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                mask_path = file_path.parent / (file_path.stem + "_mask.npy")
                if mask_path.exists():
                    mask_path.unlink()

            return True

        except Exception as e:
            print(f"Error deleting media file {file_path}: {e}")
            return False

    def retrieve_media(
        self,
        media_entries: List["MediaEntry"],
        modality: Modality,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Sample media content from filesystem based on database entries.

        Args:
            media_entries: List of MediaEntry objects from database
            modality: The modality (Modality.IMAGE or Modality.VIDEO)
            **kwargs: Additional arguments for reading media

        Returns:
            Dictionary with sampled items
        """
        if not media_entries:
            return {"count": 0, "items": []}

        sampled_items = []

        for media_entry in media_entries:
            file_path = Path(media_entry.file_path)
            if not file_path.exists():
                print(f"Media file not found: {file_path}")
                continue

            try:
                if modality == Modality.IMAGE:
                    image = self.read_image(
                        file_path,
                        as_float32=kwargs.get("as_float32", False),
                        channels_first=kwargs.get("channels_first", False),
                        as_rgb=kwargs.get("as_rgb", True),
                    )

                    if image is None:
                        continue

                    mask = None
                    mask_path = None
                    if kwargs.get("require_mask", False):
                        mask_path = file_path.with_suffix("_mask.npy")
                        mask = self.read_mask(mask_path)
                        if mask is None:
                            print(f"Mask not found for {file_path}")
                            continue

                    item = {
                        "image": image,
                        "path": str(file_path),
                        "dataset": media_entry.model_name or "unknown",
                    }

                    if mask is not None:
                        item["mask"] = mask
                        item["mask_path"] = str(mask_path)

                    sampled_items.append(item)

                elif modality == Modality.VIDEO:
                    video_result = self.read_video_frames(
                        file_path,
                        min_duration=kwargs.get("min_duration", 1.0),
                        max_duration=kwargs.get("max_duration", 6.0),
                        max_frames=kwargs.get("max_frames", 24),
                    )

                    if video_result is None or video_result.get("video") is None:
                        continue

                    item = {
                        "video": video_result["video"],
                        "path": str(file_path),
                        "dataset": media_entry.model_name or "unknown",
                        "duration": video_result.get("segment", {}).get("duration", 0),
                        "fps": video_result.get("segment", {}).get("fps", 30),
                    }

                    sampled_items.append(item)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        return {"count": len(sampled_items), "items": sampled_items}
