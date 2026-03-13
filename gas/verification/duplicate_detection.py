"""
Perceptual hashing utilities for duplicate detection in miner-submitted media.

Uses imagehash library for images and frame-based hashing for videos.
Supports configurable similarity thresholds via Hamming distance.
Includes crop-resistant hashing to detect cropped duplicates.
"""

import io
from pathlib import Path
from typing import Optional, List, Tuple, Union

import bittensor as bt
import numpy as np
from PIL import Image

try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False
    bt.logging.warning("imagehash not installed. Duplicate detection will be disabled.")


# Default Hamming distance threshold for near-duplicate detection
# Lower values = stricter matching (0 = exact match only)
# Typical values: 5-10 for near-duplicates, 0-2 for exact/near-exact
DEFAULT_HAMMING_THRESHOLD = 8

# Threshold for crop-resistant hash segment matching
# Number of matching segments required to consider as duplicate
DEFAULT_CROP_RESISTANT_MATCH_THRESHOLD = 2


def compute_image_hash(
    image_data: Union[bytes, str, Path, Image.Image],
    hash_size: int = 16,
    include_crop_resistant: bool = True,
) -> Optional[str]:
    """
    Compute perceptual hash (pHash) for an image, optionally with crop-resistant hash.

    Args:
        image_data: Image as bytes, file path, or PIL Image
        hash_size: Size of the hash (default 16 for 256-bit hash)
        include_crop_resistant: Whether to include crop-resistant hash segments

    Returns:
        Hex string representation of the hash (format: "phash" or "phash|crop_hash"),
        or None if failed
    """
    if not IMAGEHASH_AVAILABLE:
        return None

    try:
        # Load image from various input types
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, (str, Path)):
            image = Image.open(image_data)
        elif isinstance(image_data, Image.Image):
            image = image_data
        else:
            bt.logging.warning(f"Unsupported image type: {type(image_data)}")
            return None

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Compute perceptual hash (pHash)
        phash = imagehash.phash(image, hash_size=hash_size)
        result = str(phash)

        # Compute crop-resistant hash if requested
        if include_crop_resistant:
            try:
                crop_hash = compute_crop_resistant_hash(image)
                if crop_hash:
                    result = f"{result}|{crop_hash}"
            except Exception as e:
                bt.logging.debug(f"Crop-resistant hash failed, using pHash only: {e}")

        return result

    except Exception as e:
        bt.logging.warning(f"Failed to compute image hash: {e}")
        return None


def compute_crop_resistant_hash(
    image_data: Union[bytes, str, Path, Image.Image],
    hash_func: str = "phash",
    min_segment_size: int = 64,
    segmentation_method: int = 1,
) -> Optional[str]:
    """
    Compute crop-resistant hash for an image using imagehash's crop_resistant_hash.
    
    This creates multiple segment hashes that can detect duplicates even when
    the image has been cropped.

    Args:
        image_data: Image as bytes, file path, or PIL Image
        hash_func: Hash function to use for segments ("phash", "dhash", "ahash")
        min_segment_size: Minimum size for image segments
        segmentation_method: Segmentation method (1 = default)

    Returns:
        Semicolon-separated string of segment hashes, or None if failed
    """
    if not IMAGEHASH_AVAILABLE:
        return None

    try:
        # Load image if needed
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, (str, Path)):
            image = Image.open(image_data)
        elif isinstance(image_data, Image.Image):
            image = image_data
        else:
            return None

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Select hash function
        if hash_func == "phash":
            hfunc = imagehash.phash
        elif hash_func == "dhash":
            hfunc = imagehash.dhash
        elif hash_func == "ahash":
            hfunc = imagehash.average_hash
        else:
            hfunc = imagehash.phash

        # Compute crop-resistant hash (returns a set of ImageHash objects)
        crop_hashes = imagehash.crop_resistant_hash(
            image,
            hash_func=hfunc,
            limit_segments=4,
            min_segment_size=min_segment_size,
            segmentation_method=segmentation_method,
        )

        if not crop_hashes:
            return None

        # Convert to semicolon-separated string of hex hashes
        hash_strings = [str(h) for h in crop_hashes]
        return ";".join(sorted(hash_strings))

    except Exception as e:
        bt.logging.debug(f"Failed to compute crop-resistant hash: {e}")
        return None


def compute_video_hash(
    video_path: Union[str, Path],
    num_frames: int = 4,
    hash_size: int = 16,
    include_crop_resistant: bool = True,
) -> Optional[str]:
    """
    Compute perceptual hash for a video by hashing key frames and combining.
    Includes crop-resistant hashes for each frame.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample uniformly
        hash_size: Size of each frame hash
        include_crop_resistant: Whether to include crop-resistant hashes

    Returns:
        Combined hex string of frame hashes with optional crop-resistant segments,
        or None if failed
    """
    if not IMAGEHASH_AVAILABLE:
        return None

    try:
        import cv2

        video_path = Path(video_path)
        if not video_path.exists():
            bt.logging.warning(f"Video file not found: {video_path}")
            return None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            bt.logging.warning(f"Could not open video: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            bt.logging.warning(f"Video has no frames: {video_path}")
            cap.release()
            return None

        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
        frame_hashes = []
        all_crop_segments = []

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret and frame is not None:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)

                # Compute pHash for this frame
                phash = imagehash.phash(image, hash_size=hash_size)
                frame_hashes.append(str(phash))

                # Compute crop-resistant hash for key frames
                if include_crop_resistant and len(all_crop_segments) < 8:
                    try:
                        crop_hash = compute_crop_resistant_hash(image)
                        if crop_hash:
                            # Add first 2 segments from each frame
                            segments = crop_hash.split(';')[:2]
                            all_crop_segments.extend(segments)
                    except Exception:
                        pass

        cap.release()

        if not frame_hashes:
            bt.logging.warning(f"Could not extract any frames from video: {video_path}")
            return None

        # Combine frame hashes into single string
        # Format: "primary_hash_framecount" or "primary_hash|crop_segments_framecount"
        primary_hash = frame_hashes[0]
        
        if include_crop_resistant and all_crop_segments:
            # Deduplicate and limit segments
            unique_segments = list(dict.fromkeys(all_crop_segments))[:6]
            crop_part = ";".join(unique_segments)
            combined_hash = f"{primary_hash}|{crop_part}_{len(frame_hashes)}"
        else:
            combined_hash = f"{primary_hash}_{len(frame_hashes)}"
        
        return combined_hash

    except Exception as e:
        bt.logging.warning(f"Failed to compute video hash: {e}")
        return None


def compute_media_hash(
    media_data: Union[bytes, str, Path],
    modality: str = "image",
    hash_size: int = 16,
) -> Optional[str]:
    """
    Compute perceptual hash for media based on modality.

    Args:
        media_data: Media as bytes or file path
        modality: "image" or "video"
        hash_size: Size of the hash

    Returns:
        Hex string representation of the hash, or None if failed
    """
    if modality == "image":
        return compute_image_hash(media_data, hash_size=hash_size)
    elif modality == "video":
        if isinstance(media_data, bytes):
            # For video bytes, we need to write to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(media_data)
                tmp_path = tmp.name
            try:
                result = compute_video_hash(tmp_path, hash_size=hash_size)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
            return result
        else:
            return compute_video_hash(media_data, hash_size=hash_size)
    else:
        bt.logging.warning(f"Unsupported modality for hashing: {modality}")
        return None


def extract_phash(full_hash: str) -> str:
    """
    Extract the primary pHash from a combined hash string.
    
    Hash format: "phash" or "phash|crop_segments" or "phash_framecount"
    """
    # Handle video hashes with frame count suffix
    if '_' in full_hash:
        full_hash = full_hash.split('_')[0]
    
    # Handle combined hash with crop-resistant segments
    if '|' in full_hash:
        return full_hash.split('|')[0]
    
    return full_hash


def extract_crop_segments(full_hash: str) -> List[str]:
    """
    Extract crop-resistant hash segments from a combined hash string.
    
    Hash format: "phash|segment1;segment2;segment3"
    """
    if '|' not in full_hash:
        return []
    
    parts = full_hash.split('|')
    if len(parts) < 2:
        return []
    
    crop_part = parts[1]
    return crop_part.split(';') if crop_part else []


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Calculate Hamming distance between two hex hash strings.
    Extracts the primary pHash from combined hash formats.

    Args:
        hash1: First hash as hex string (may include crop-resistant segments)
        hash2: Second hash as hex string (may include crop-resistant segments)

    Returns:
        Hamming distance (number of differing bits)
    """
    if not IMAGEHASH_AVAILABLE:
        return -1

    try:
        # Extract primary pHash from combined formats
        h1 = extract_phash(hash1)
        h2 = extract_phash(hash2)

        # Convert hex strings back to imagehash objects
        ihash1 = imagehash.hex_to_hash(h1)
        ihash2 = imagehash.hex_to_hash(h2)

        return ihash1 - ihash2  # imagehash overloads subtraction for Hamming distance

    except Exception as e:
        bt.logging.warning(f"Failed to compute Hamming distance: {e}")
        return -1


def count_crop_segment_matches(
    hash1: str,
    hash2: str,
    segment_threshold: int = 5,
) -> int:
    """
    Count matching crop-resistant hash segments between two hashes.
    
    Args:
        hash1: First hash with crop segments
        hash2: Second hash with crop segments
        segment_threshold: Maximum Hamming distance for segment match

    Returns:
        Number of matching segments
    """
    if not IMAGEHASH_AVAILABLE:
        return 0

    try:
        segments1 = extract_crop_segments(hash1)
        segments2 = extract_crop_segments(hash2)

        if not segments1 or not segments2:
            return 0

        matches = 0
        for seg1 in segments1:
            for seg2 in segments2:
                try:
                    ih1 = imagehash.hex_to_hash(seg1)
                    ih2 = imagehash.hex_to_hash(seg2)
                    distance = ih1 - ih2
                    if distance <= segment_threshold:
                        matches += 1
                        break  # Count each segment1 only once
                except Exception:
                    continue

        return matches

    except Exception as e:
        bt.logging.debug(f"Failed to count crop segment matches: {e}")
        return 0


def is_duplicate(
    hash1: str,
    hash2: str,
    threshold: int = DEFAULT_HAMMING_THRESHOLD,
    crop_match_threshold: int = DEFAULT_CROP_RESISTANT_MATCH_THRESHOLD,
) -> bool:
    """
    Check if two hashes represent duplicate/near-duplicate content.
    Uses both pHash comparison and crop-resistant segment matching.

    Args:
        hash1: First perceptual hash (may include crop segments)
        hash2: Second perceptual hash (may include crop segments)
        threshold: Maximum Hamming distance for pHash duplicate
        crop_match_threshold: Minimum matching crop segments for duplicate

    Returns:
        True if media are duplicates (by pHash OR crop segments), False otherwise
    """
    # Check primary pHash
    distance = hamming_distance(hash1, hash2)
    if distance >= 0 and distance <= threshold:
        return True

    # Check crop-resistant segments (catches cropped duplicates)
    crop_matches = count_crop_segment_matches(hash1, hash2)
    if crop_matches >= crop_match_threshold:
        return True

    return False


def find_duplicates(
    new_hash: str,
    existing_hashes: List[Tuple[str, str]],  # List of (media_id, hash)
    threshold: int = DEFAULT_HAMMING_THRESHOLD,
    crop_match_threshold: int = DEFAULT_CROP_RESISTANT_MATCH_THRESHOLD,
) -> List[Tuple[str, int]]:
    """
    Find all duplicates of a new hash in a list of existing hashes.
    Uses both pHash and crop-resistant matching.

    Args:
        new_hash: Hash of new media to check
        existing_hashes: List of (media_id, hash) tuples
        threshold: Maximum Hamming distance for pHash duplicate
        crop_match_threshold: Minimum matching crop segments for duplicate

    Returns:
        List of (media_id, hamming_distance) for all matches
        Note: For crop-resistant matches, distance is reported as 0
    """
    duplicates = []

    for media_id, existing_hash in existing_hashes:
        if not existing_hash:
            continue

        # Check pHash distance
        distance = hamming_distance(new_hash, existing_hash)
        if distance >= 0 and distance <= threshold:
            duplicates.append((media_id, distance))
            continue

        # Check crop-resistant segments
        crop_matches = count_crop_segment_matches(new_hash, existing_hash)
        if crop_matches >= crop_match_threshold:
            # Report as distance 0 to indicate strong match via crop segments
            duplicates.append((media_id, 0))

    # Sort by distance (closest matches first)
    duplicates.sort(key=lambda x: x[1])
    return duplicates


def check_duplicate_in_db(
    content_db,
    new_hash: str,
    threshold: int = DEFAULT_HAMMING_THRESHOLD,
    crop_match_threshold: int = DEFAULT_CROP_RESISTANT_MATCH_THRESHOLD,
    limit: int = 1000,
    prompt_id: Optional[str] = None,
) -> Optional[Tuple[str, int]]:
    """
    Check if a hash has duplicates in the database.
    Uses both pHash and crop-resistant matching to catch cropped duplicates.

    Note: For large databases, consider using a more efficient similarity
    search structure (LSH, FAISS, etc.) instead of linear scan.

    Args:
        content_db: ContentDB instance
        new_hash: Hash to check (may include crop-resistant segments)
        threshold: Maximum Hamming distance for pHash duplicate
        crop_match_threshold: Minimum crop segment matches for duplicate
        limit: Maximum number of hashes to check (most recent)
        prompt_id: If provided, only check duplicates within this prompt

    Returns:
        Tuple of (media_id, distance) for closest match, or None if no duplicate
    """
    try:
        with content_db._get_db_connection() as conn:
            if prompt_id:
                cursor = conn.execute(
                    """
                    SELECT id, perceptual_hash FROM media 
                    WHERE perceptual_hash IS NOT NULL AND source_type = 'miner' AND prompt_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (prompt_id, limit,)
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT id, perceptual_hash FROM media 
                    WHERE perceptual_hash IS NOT NULL AND source_type = 'miner'
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,)
                )
            rows = cursor.fetchall()

        if not rows:
            return None

        existing_hashes = [(row[0], row[1]) for row in rows]
        duplicates = find_duplicates(
            new_hash, 
            existing_hashes, 
            threshold, 
            crop_match_threshold
        )

        return duplicates[0] if duplicates else None

    except Exception as e:
        bt.logging.error(f"Error checking duplicates in database: {e}")
        return None

