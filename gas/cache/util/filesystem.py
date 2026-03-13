import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from zipfile import BadZipFile, ZipFile

import bittensor as bt
import pyarrow.parquet as pq

from gas.types import FileType, MediaType, Modality


def get_most_recent_update_time(directory: Path) -> float:
    """Get the most recent modification time of any file in directory."""
    try:
        mtimes = [f.stat().st_mtime for f in directory.iterdir()]
        return max(mtimes) if mtimes else 0
    except Exception as e:
        bt.logging.error(f"Error getting modification times: {e}")
        return 0


def is_source_complete(path: Union[str, Path]) -> Callable[[Path], bool]:
    """Checks integrity of parquet or zip file"""

    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return is_parquet_complete(path)
    elif path.suffix.lower() == ".zip":
        return is_zip_complete(path)
    else:
        return None


def is_zip_complete(zip_path: Union[str, Path], testzip=False) -> bool:
    try:
        with ZipFile(zip_path) as zf:
            if testzip:
                zf.testzip()
            else:
                zf.namelist()
            return True
    except (BadZipFile, Exception) as e:
        bt.logging.error(f"Zip file {zip_path} is invalid: {e}")
        return False


def is_parquet_complete(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            pq.read_metadata(f)
        return True
    except Exception as e:
        bt.logging.error(f"Parquet file {path} is incomplete or corrupted: {e}")
        return False


def get_dir_size(
    path: Union[str, Path], exclude_dirs: Optional[List[str]] = None
) -> Tuple[int, int]:
    if exclude_dirs is None:
        exclude_dirs = []

    total_size = 0
    file_count = 0
    path_obj = Path(path)

    try:
        for item in path_obj.iterdir():
            if item.is_dir() and item.name in exclude_dirs:
                continue
            elif item.is_file():
                try:
                    total_size += item.stat().st_size
                    file_count += 1
                except (OSError, PermissionError):
                    pass
            elif item.is_dir():
                subdir_size, subdir_count = get_dir_size(item, exclude_dirs)
                total_size += subdir_size
                file_count += subdir_count
    except (PermissionError, OSError) as e:
        print(f"Error accessing {path}: {e}", file=sys.stderr)

    return total_size, file_count


def scale_size(size: float, from_unit: str = "B", to_unit: str = "GB") -> float:
    if size == 0:
        return 0.0

    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    from_unit, to_unit = from_unit.upper(), to_unit.upper()
    if from_unit not in units or to_unit not in units:
        raise ValueError(f"Units must be one of: {', '.join(units)}")

    from_index = units.index(from_unit)
    to_index = units.index(to_unit)
    scale_factor = from_index - to_index

    if scale_factor > 0:
        return size * (1024**scale_factor)
    elif scale_factor < 0:
        return size / (1024 ** abs(scale_factor))
    return size


def format_size(
    size: float, from_unit: str = "B", to_unit: Optional[str] = None
) -> str:
    if size == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    from_unit = from_unit.upper()

    if from_unit not in units:
        raise ValueError(f"From unit must be one of: {', '.join(units)}")

    if to_unit is None:
        current_size = scale_size(size, from_unit, "B")
        unit_index = 0

        while current_size >= 1024 and unit_index < len(units) - 1:
            current_size /= 1024
            unit_index += 1

        return f"{current_size:.2f} {units[unit_index]}"
    else:
        to_unit = to_unit.upper()
        if to_unit not in units:
            raise ValueError(f"To unit must be one of: {', '.join(units)}")
        scaled_size = scale_size(size, from_unit, to_unit)
        return f"{scaled_size:.2f} {to_unit}"


def analyze_directory(
    root_path: Union[str, Path],
    exclude_dirs: Optional[List[str]] = None,
    min_file_count: int = 1,
    log_func=None,
) -> Dict[str, Any]:
    if exclude_dirs is None:
        exclude_dirs = []

    path_obj = Path(root_path)
    result = {
        "name": path_obj.name or str(path_obj),
        "path": str(path_obj),
        "subdirs": [],
        "excluded_dirs": [],
    }

    size, count = get_dir_size(path_obj, exclude_dirs)
    result["size"] = size
    result["count"] = count

    try:
        subdirs = [d for d in path_obj.iterdir() if d.is_dir()]

        for subdir in sorted(subdirs):
            if subdir.name in exclude_dirs:
                _, excluded_count = get_dir_size(subdir, [])
                if excluded_count < min_file_count:
                    continue

                excluded_data = analyze_directory(subdir, [], min_file_count, log_func)
                excluded_data["excluded"] = True
                result["excluded_dirs"].append(excluded_data)
            else:
                subdir_data = analyze_directory(
                    subdir, exclude_dirs, min_file_count, log_func
                )
                if subdir_data["count"] < min_file_count:
                    continue

                result["subdirs"].append(subdir_data)
    except (PermissionError, OSError) as e:
        error_msg = f"Error accessing {path_obj}: {e}"
        if log_func:
            log_func(error_msg)
        else:
            print(error_msg, file=sys.stderr)

    return result


def print_directory_tree(
    tree_data: Dict[str, Any],
    indent: str = "",
    is_last: bool = True,
    prefix: str = "",
    log_func=None,
) -> None:
    if (
        tree_data["count"] == 0
        and not tree_data["subdirs"]
        and not tree_data["excluded_dirs"]
    ):
        return

    if is_last:
        branch = "└── "
        next_indent = indent + "    "
    else:
        branch = "├── "
        next_indent = indent + "│   "

    name = tree_data["name"]
    count = tree_data["count"]
    size = scale_size(tree_data["size"])

    tree_line = f"{indent}{prefix}{branch}[{name}] - {count} files, {size}"
    if log_func:
        log_func(tree_line)
    else:
        print(tree_line)

    num_subdirs = len(tree_data["subdirs"])

    for i, subdir in enumerate(tree_data["subdirs"]):
        is_subdir_last = (i == num_subdirs - 1) and not tree_data["excluded_dirs"]
        print_directory_tree(subdir, next_indent, is_subdir_last, "", log_func)

    for i, excluded in enumerate(tree_data["excluded_dirs"]):
        is_excluded_last = i == len(tree_data["excluded_dirs"]) - 1
        print_directory_tree(
            excluded, next_indent, is_excluded_last, "(SOURCE) ", log_func
        )


def is_file_older_than(file_path: Union[str, Path], seconds: float = 1.0) -> bool:
    """Check if a file's last modification time is older than specified seconds."""
    try:
        mtime = Path(file_path).stat().st_mtime
        return (time.time() - mtime) >= seconds
    except (FileNotFoundError, PermissionError):
        return False


def has_stable_size(file_path: Union[str, Path], wait_time: float = 0.1) -> bool:
    """Check if a file's size is stable (not changing)."""
    path = Path(file_path)
    try:
        size1 = path.stat().st_size
        time.sleep(wait_time)
        size2 = path.stat().st_size
        return size1 == size2
    except (FileNotFoundError, PermissionError):
        return False


def is_file_locked(file_path: Union[str, Path]) -> bool:
    """Check if a file is locked (being written to by another process)."""
    try:
        with open(file_path, "rb+") as _:
            pass
        return False
    except (PermissionError, OSError):
        return True


def is_file_ready(
    file_path: Union[str, Path],
    min_age_seconds: float = 1.0,
    check_size_stability: bool = False,
    check_file_lock: bool = True,
    stability_wait_time: float = 0.1,
) -> bool:
    """
    Determine if a file is ready for processing (not being downloaded/written to).

    Args:
        file_path: Path to the file to check
        min_age_seconds: Minimum age in seconds since last modification
        check_size_stability: Whether to check if file size is stable
        check_file_lock: Whether to check if file is locked by another process
        stability_wait_time: Time to wait when checking size stability

    Returns:
        bool: True if the file appears ready for processing
    """
    file_path = Path(file_path) if isinstance(file_path, str) else file_path

    if not file_path.exists() or not file_path.is_file():
        return False

    if not is_file_older_than(file_path, min_age_seconds):
        return False

    if check_size_stability and not has_stable_size(file_path, stability_wait_time):
        return False

    if check_file_lock and is_file_locked(file_path):
        return False

    return True


def filter_ready_files(
    file_list: List[Union[str, Path]], **kwargs
) -> List[Union[str, Path]]:
    """
    Filter a list of files to only include those that are ready for processing.

    Args:
        file_list: List of file paths
        **kwargs: Additional arguments to pass to is_file_ready()

    Returns:
        list: Filtered list containing only ready files
    """
    return [f for f in file_list if is_file_ready(f, **kwargs)]


async def wait_for_downloads_to_complete(
    files: List[Path], min_age_seconds: float = 2.0, timeout_seconds: int = 180
) -> bool:
    if not files:
        return True

    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        ready_files = filter_ready_files(
            file_list=files, min_age_seconds=min_age_seconds
        )
        if len(ready_files) == len(files):
            return True
        #  yield to event loop
        await asyncio.sleep(5)

    bt.logging.error(f"Timeout waiting for {files} after {timeout_seconds} seconds")
    return False


def format_to_extension(format_str: str) -> str:
    """
    Convert a format string to its corresponding file extension.

    Args:
        format_str: Format string (e.g., "PNG", "JPEG", "MP4")

    Returns:
        File extension with leading dot (e.g., ".png", ".jpg", ".mp4")
    """
    format_to_extension_map = {
        "PNG": ".png",
        "JPEG": ".jpg",
        "JPG": ".jpg",
        "MP4": ".mp4",
        "AVI": ".avi",
        "MOV": ".mov",
        "WEBM": ".webm",
    }

    return format_to_extension_map.get(format_str.upper(), ".png")


def extract_media_info(
    file_path: Union[str, Path], modality: Modality
) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
    """
    Extract resolution and file size from a media file.

    Args:
        file_path: Path to the media file
        modality: Modality.IMAGE or Modality.VIDEO

    Returns:
        Tuple of (resolution, file_size) where both can be None if failed
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return None, None

        # Get file size
        file_size = path.stat().st_size

        # Get resolution
        resolution = None
        if modality == Modality.IMAGE:
            try:
                from PIL import Image

                with Image.open(path) as img:
                    resolution = img.size
            except Exception as e:
                bt.logging.warning(
                    f"Could not extract image resolution from {file_path}: {e}"
                )

        elif modality == Modality.VIDEO:
            try:
                import cv2

                cap = cv2.VideoCapture(str(path))
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    resolution = (width, height)
                    cap.release()
                else:
                    bt.logging.warning(
                        f"Could not open video file to get resolution: {file_path}"
                    )
            except Exception as e:
                bt.logging.warning(
                    f"Could not extract video resolution from {file_path}: {e}"
                )

        return resolution, file_size

    except Exception as e:
        bt.logging.warning(f"Error extracting media info from {file_path}: {e}")
        return None, None


def get_format_from_content(media_content, modality: Modality = Modality.IMAGE) -> str:
    """
    Determine the appropriate format for media content.

    Args:
        media_content: The media content (PIL Image, video frames, etc.)
        modality: Modality.IMAGE or Modality.VIDEO

    Returns:
        Format string (e.g., "PNG", "JPEG", "MP4")
    """
    # Try to get format from content if it has format info
    if hasattr(media_content, "format") and media_content.format:
        return media_content.format.upper()

    # For images, determine format based on mode
    if modality == Modality.IMAGE and hasattr(media_content, "mode"):
        if media_content.mode == "RGBA":
            return "PNG"  # PNG for images with alpha channel
        elif media_content.mode == "RGB":
            return "JPEG"  # JPEG for RGB images
        else:
            return "PNG"  # Default for other modes

    # Default formats
    if modality == Modality.IMAGE:
        return "PNG"
    elif modality == Modality.VIDEO:
        return "MP4"
    else:
        return "PNG"
