from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import bittensor as bt
import numpy as np
from PIL import Image
import torch
import cv2
import gc

import clip

# Global cache for multiple CLIP models
_clip_model_cache = {}

CLIP_MODELS = [
    "ViT-B/32",
    "RN50", 
    "RN101",
]


def preload_clip_models():
    bt.logging.info("Preloading CLIP models for consensus verification...")

    for model_name in CLIP_MODELS:
        try:
            if model_name not in _clip_model_cache:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                bt.logging.info(f"Loading CLIP model: {model_name}")
                model, preprocess = clip.load(model_name, device=device)
                _clip_model_cache[model_name] = (model, preprocess, device)
                bt.logging.info(f"Loaded {model_name} on {device}")
            else:
                bt.logging.info(f"{model_name} already cached")
        except Exception as e:
            bt.logging.error(f"Failed to load {model_name}: {e}")

    loaded_models = len([m for m in CLIP_MODELS if m in _clip_model_cache])
    bt.logging.info(
        f"CLIP model preloading complete: {loaded_models}/{len(CLIP_MODELS)} models ready"
    )


def clear_clip_models():
    global _clip_model_cache

    if not _clip_model_cache:
        bt.logging.debug("No CLIP models to clear from cache")
        return

    bt.logging.info("Clearing CLIP models from memory...")

    cleared_models = list(_clip_model_cache.keys())
    _clip_model_cache.clear()
    gc.collect()

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            bt.logging.info("Cleared CUDA cache")
    except Exception as e:
        bt.logging.debug(f"Could not clear CUDA cache: {e}")

    bt.logging.info(
        f"Cleared {len(cleared_models)} CLIP models from memory: {cleared_models}"
    )


def extract_temporal_frames(video_path: str) -> List[np.ndarray]:
    """
    Extract 4 frames uniformly from a video for temporal analysis.
    
        Args:
            video_path: Path to video file
            
        Returns:
            List of 4 frames as numpy arrays (RGB format), or empty list if video is invalid
    """
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            bt.logging.debug(f"Could not open video (likely corrupted): {video_path}")
            return []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            bt.logging.debug(f"Video has no frames (likely corrupted): {video_path}")
            return []
        
        # Check if video is too short (single frame or near-single frame)
        if total_frames < 3:
            bt.logging.debug(f"Video has too few frames ({total_frames}), likely a static image: {video_path}")
            return []
            
        frames = []
        failed_reads = 0
        
        # Sample 8 frames uniformly across the video duration
        frame_indices = np.linspace(0, total_frames - 1, min(4, total_frames), dtype=int)
            
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            else:
                failed_reads += 1
                
        # If we failed to read more than half the frames, video is corrupted
        if failed_reads > len(frame_indices) / 2 or len(frames) < 2:
            bt.logging.debug(f"Could not read enough frames from video ({len(frames)}/{len(frame_indices)} succeeded): {video_path}")
            return []
        
        # Check if all frames are identical (miners trying to game with static images)
        if len(frames) >= 2:
            first_frame = frames[0]
            all_identical = all(np.array_equal(first_frame, frame) for frame in frames[1:])
            if all_identical:
                bt.logging.debug(f"All frames are identical (static image posing as video): {video_path}")
                return []
                
        bt.logging.debug(f"Extracted {len(frames)} valid frames from {video_path}")
        return frames
        
    except Exception as e:
        bt.logging.debug(f"Error extracting frames from {video_path}: {e}")
        return []
    finally:
        if cap is not None:
            cap.release()


def process_video_temporal(
    video_path: str,
    model,
    preprocess,
    device: str,
) -> Optional[torch.Tensor]:
    """
    Process a video using 4 uniformly sampled frames with mean aggregation.
    
    Args:
        video_path: Path to video file
        model: CLIP model to use for encoding
        preprocess: CLIP preprocessing function
        device: Device to run on
        
    Returns:
        Mean-aggregated video features tensor or None if failed
    """
    try:
        frames = extract_temporal_frames(video_path)
        if not frames:
            return None
            
        # Process each frame
        frame_tensors = []
        for frame in frames:
            try:
                image = Image.fromarray(frame)
                image_tensor = preprocess(image)
                frame_tensors.append(image_tensor)
            except Exception as e:
                bt.logging.warning(f"Error processing frame: {e}")
                continue
                
        if not frame_tensors:
            return None
            
        # Stack frames and move to device
        video_tensor = torch.stack(frame_tensors).to(device)
        
        # Get frame features using the provided model
        with torch.no_grad():
            frame_features = model.encode_image(video_tensor)
            frame_features = frame_features / frame_features.norm(dim=-1, keepdim=True)
        
        # Aggregate temporal features using mean
        video_features = torch.mean(frame_features, dim=0, keepdim=True)
            
        # Normalize aggregated features
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        
        return video_features
        
    except Exception as e:
        bt.logging.error(f"Error in temporal video processing for {video_path}: {e}")
        return None


def calculate_clip_alignment(
    media_paths: List[str],
    prompts: List[str],
    model_name: str = "ViT-B/32",
    batch_size: int = 32,
) -> Optional[List[float]]:
    """
    Calculate CLIP embedding cosine similarities for a batch of multiple media-prompt pairs

    Args:
        media_paths: List of paths to image or video files
        prompts: List of text prompts to compare against (must match length of media_paths)
        model_name: CLIP model variant to use
        batch_size: Batch size for processing (adjust based on GPU memory)

    Returns:
        List of similarity scores between 0 and 1, or None if failed
    """
    try:
        if len(media_paths) != len(prompts):
            bt.logging.error(
                f"Mismatch: {len(media_paths)} media files vs {len(prompts)} prompts"
            )
            return None

        if len(media_paths) == 0:
            return []

        # Load model
        if model_name not in _clip_model_cache:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load(model_name, device=device)
            _clip_model_cache[model_name] = (model, preprocess, device)
            bt.logging.debug(f"Loaded CLIP model: {model_name} on {device}")

        model, preprocess, device = _clip_model_cache[model_name]

        # Batch process text prompts
        bt.logging.debug(f"Processing {len(prompts)} text prompts in batches")
        all_text_features = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            text_tokens = clip.tokenize(batch_prompts, truncate=True).to(device)

            with torch.no_grad():
                batch_text_features = model.encode_text(text_tokens)
                batch_text_features = batch_text_features / batch_text_features.norm(
                    dim=-1, keepdim=True
                )
                all_text_features.append(batch_text_features)

        # Combine all text features
        text_features = torch.cat(all_text_features, dim=0)

        bt.logging.info(f"[{model_name}] Processing {len(media_paths)} media files in batches of {batch_size}")
        all_media_features = []

        for i in range(0, len(media_paths), batch_size):
            batch_paths = media_paths[i : i + batch_size]
            batch_images = []
            batch_valid_indices = []

            # Load and preprocess images/videos for this batch
            for j, media_path in enumerate(batch_paths):
                try:
                    media_path = Path(media_path)
                    if not media_path.exists():
                        bt.logging.warning(f"Media file not found: {media_path}")
                        continue

                    # Handle images
                    if media_path.suffix.lower() in [
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp",
                        ".webp",
                    ]:
                        image = Image.open(media_path).convert("RGB")
                        image_tensor = preprocess(image)
                        batch_images.append(image_tensor)
                        batch_valid_indices.append(i + j)

                    # Handle videos with temporal processing 
                    elif media_path.suffix.lower() in [
                        ".mp4",
                        ".avi",
                        ".mov",
                        ".mkv",
                        ".webm",
                    ]:
                        # Videos will be processed separately with temporal features
                        # For now, mark this position for later temporal processing
                        batch_images.append(None)  # Placeholder for video
                        batch_valid_indices.append(i + j)

                except Exception as e:
                    bt.logging.warning(f"Error processing {media_path}: {e}")
                    continue

            # Separate images and videos for processing
            image_tensors = []
            image_indices = []
            video_paths_in_batch = []
            video_indices = []
            
            for k, (media_tensor, orig_idx) in enumerate(zip(batch_images, batch_valid_indices)):
                if media_tensor is not None:
                    # This is an image
                    image_tensors.append(media_tensor)
                    image_indices.append((k, orig_idx))
                else:
                    # This is a video placeholder
                    local_idx = orig_idx - i
                    video_path = batch_paths[local_idx]
                    video_paths_in_batch.append((str(video_path), k, orig_idx))
            
            # Initialize features for this batch
            batch_features_padded = torch.zeros(
                len(batch_paths), text_features.shape[1], device=device
            )
            
            # Process images in batch
            if image_tensors:
                image_batch_tensor = torch.stack(image_tensors).to(device)
                with torch.no_grad():
                    image_features = model.encode_image(image_batch_tensor)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Store image features
                for feat_idx, (local_k, orig_idx) in enumerate(image_indices):
                    local_idx = orig_idx - i
                    batch_features_padded[local_idx] = image_features[feat_idx]
            
            # Process videos individually with temporal processing
            if video_paths_in_batch:
                bt.logging.info(f"[{model_name}] Processing {len(video_paths_in_batch)} videos in batch {i//batch_size + 1}")
            
            for video_idx, (video_path, local_k, orig_idx) in enumerate(video_paths_in_batch, 1):
                try:
                    bt.logging.debug(f"[{model_name}] Processing video {video_idx}/{len(video_paths_in_batch)}: {Path(video_path).name}")
                    video_features = process_video_temporal(
                        video_path, model, preprocess, device
                    )
                    if video_features is not None:
                        local_idx = orig_idx - i
                        batch_features_padded[local_idx] = video_features[0]  # Remove batch dimension
                        bt.logging.debug(f"[{model_name}] Completed video {video_idx}/{len(video_paths_in_batch)}")
                    else:
                        # Video is corrupted/invalid, leave features as zeros (will result in low score)
                        bt.logging.debug(f"Failed to process video (corrupted/invalid): {video_path}")
                except Exception as e:
                    bt.logging.debug(f"Error processing video {video_path}: {e}")
            
            all_media_features.append(batch_features_padded)

        media_features = torch.cat(all_media_features, dim=0)
        similarities = torch.cosine_similarity(media_features, text_features, dim=-1)
        final_scores = similarities.cpu().numpy().tolist()

        bt.logging.info(
            f"CLIP model {model_name} processing complete: {len(final_scores)} scores computed"
        )
        return final_scores

    except Exception as e:
        bt.logging.error(f"Error in CLIP alignment calculation: {e}")
        return None


def calculate_clip_alignment_consensus(
    media_paths: List[str], prompts: List[str], batch_size: int = 32
) -> Optional[List[Dict[str, Any]]]:
    """
    Calculate CLIP alignment using multiple models for consensus scoring.

    Args:
        media_paths: List of paths to image or video files
        prompts: List of text prompts to compare against
        batch_size: Batch size for processing

    Returns:
        List of dictionaries with consensus scores and individual model scores, or None if failed
    """
    try:
        if len(media_paths) != len(prompts):
            bt.logging.error(
                f"Mismatch: {len(media_paths)} media files vs {len(prompts)} prompts"
            )
            return None

        if len(media_paths) == 0:
            return []

        bt.logging.debug(
            f"Starting consensus CLIP alignment for {len(media_paths)} samples"
        )

        all_model_scores = {}
        failed_models = []

        for model_name in CLIP_MODELS:
            try:
                bt.logging.info(f"Running verification with model {model_name}")
                scores = calculate_clip_alignment(
                    media_paths, prompts, model_name, batch_size
                )
                if scores is not None and len(scores) == len(media_paths):
                    all_model_scores[model_name] = scores
                    bt.logging.info(f"Model {model_name} complete: got {len(scores)} scores")
                else:
                    failed_models.append(model_name)
                    bt.logging.warning(f"Model {model_name} failed to calculate scores")
            except Exception as e:
                failed_models.append(model_name)
                bt.logging.error(f"Error with model {model_name}: {e}")

        if not all_model_scores:
            bt.logging.error("All CLIP models failed to calculate scores")
            return None

        consensus_results = []
        num_samples = len(media_paths)

        # Calculate consensus scores for each sample
        for i in range(num_samples):
            sample_scores = []
            individual_scores = {}

            for model_name, model_score_list in all_model_scores.items():
                score = model_score_list[i]
                sample_scores.append(score)
                individual_scores[model_name] = score

            # Check if this is a corrupted video (all scores are 0 or very close to 0)
            # This happens when video frames couldn't be extracted
            is_corrupted = all(score < 0.001 for score in sample_scores) if sample_scores else True
            
            if is_corrupted:
                consensus_result = {
                    "consensus_score": None,
                    "individual_scores": individual_scores,
                    "num_models": len(sample_scores),
                    "corrupted": True,
                    "score_std": 0.0,
                    "score_range": [0.0, 0.0],
                    "failed_models": failed_models,
                }
            else:
                consensus_score = sum(sample_scores) / len(sample_scores)
                score_std = np.std(sample_scores) if len(sample_scores) > 1 else 0.0
                min_score = min(sample_scores)
                max_score = max(sample_scores)

                consensus_result = {
                    "consensus_score": consensus_score,
                    "individual_scores": individual_scores,
                    "num_models": len(sample_scores),
                    "corrupted": False,
                    "score_std": score_std,
                    "score_range": [min_score, max_score],
                    "failed_models": failed_models,
                }

            consensus_results.append(consensus_result)

        bt.logging.info(
            f"Consensus CLIP processing complete: {len(consensus_results)} consensus scores"
        )
        bt.logging.info(
            f"Used {len(all_model_scores)}/{len(CLIP_MODELS)} models successfully"
        )

        if failed_models:
            bt.logging.warning(f"Failed models: {failed_models}")

        return consensus_results

    except Exception as e:
        bt.logging.error(f"Error in consensus CLIP alignment calculation: {e}")
        return None


