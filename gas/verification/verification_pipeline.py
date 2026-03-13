from typing import List, Dict, Any, Optional
import bittensor as bt

from gas.cache.content_manager import ContentManager
from gas.cache.types import MediaEntry, VerificationResult
from .clip_utils import calculate_clip_alignment_consensus, preload_clip_models


def run_verification(
    content_manager: ContentManager,
    batch_size: Optional[int] = 10,
    threshold: float = 0.25,
    clip_batch_size: int = 32,
) -> List[VerificationResult]:
    """
    Run CLIP verification on unverified miner media.

    Args:
        content_manager: ContentManager instance
        batch_size: Number of media entries to process from database (None = process all)
        threshold: Verification threshold
        clip_batch_size: Batch size for CLIP operations (adjust based on GPU memory)
    """
    bt.logging.info(
        f"Starting verification batch (batch_size={batch_size}, clip_batch_size={clip_batch_size})"
    )

    pending_media = content_manager.get_miner_media(verification_status="pending")
    bt.logging.info(f"Found {len(pending_media)} pending verification miner media entries")
    if not pending_media:
        bt.logging.info("No pending verification miner media found")
        return []
            
    preload_clip_models()
    bt.logging.info("CLIP models loaded, starting verification")

    if batch_size is None:
        batch = pending_media
        bt.logging.debug(f"Processing all {len(batch)} pending media entries")
    else:
        batch = pending_media[:batch_size]
        bt.logging.info(f"Processing {len(batch)} of {len(pending_media)} pending media entries")

    results = verify_media(
        content_manager=content_manager,
        media_entries=batch,
        threshold=threshold,
        batch_size=clip_batch_size,
    )

    bt.logging.info("Verification batch complete, updating database")
    for result in results:
        if result.verification_score:  # Only process if verification actually ran
            try:
                if result.passed:
                    content_manager.mark_miner_media_verified(result.media_entry.id)
                    bt.logging.debug(f"Marked media {result.media_entry.id} as verified")
                else:
                    content_manager.mark_miner_media_failed_verification(result.media_entry.id)
                    bt.logging.debug(f"Marked media {result.media_entry.id} as failed verification")
            except Exception as e:
                bt.logging.error(f"Error marking media verification status: {e}")

    successful_verifications = sum(
        1 for r in results if r.verification_score is not None
    )
    passed_verifications = sum(1 for r in results if r.passed)
    failed_verifications = sum(1 for r in results if r.verification_score is not None and not r.passed)

    bt.logging.info(
        f"Verification batch complete: {successful_verifications}/{len(results)} processed, "
        f"{passed_verifications} passed, {failed_verifications} failed verification"
    )

    return results


def verify_media(
    content_manager: ContentManager,
    media_entries: List[MediaEntry],
    threshold: float = 0.25,
    batch_size: int = 32,
) -> List[VerificationResult]:
    """
    Verify a batch of miner media entries using CLIP consensus scoring.

    Args:
        content_manager: ContentManager instance
        media_entries: List of MediaEntry objects to verify
        threshold: Verification threshold
        batch_size: Batch size for CLIP operations

    Returns:
        List of VerificationResult objects
    """
    try:
        if not media_entries:
            return []

        bt.logging.debug(
            f"Starting verification for {len(media_entries)} media entries"
        )

        media_paths = []
        prompts = []
        valid_entries = []

        for media_entry in media_entries:
            if not media_entry.prompt_id:
                bt.logging.warning(
                    f"Media entry {media_entry.id} has no associated prompt_id"
                )
                continue

            original_prompt = content_manager.get_prompt_by_id(media_entry.prompt_id)
            if not original_prompt:
                bt.logging.warning(
                    f"Could not retrieve original prompt {media_entry.prompt_id}"
                )
                continue

            media_paths.append(media_entry.file_path)
            prompts.append(original_prompt)
            valid_entries.append(media_entry)

        if not valid_entries:
            bt.logging.warning("No valid media entries found for batch verification")
            return []

        # Separate images and videos for optimal batching
        image_entries = []
        image_paths = []
        image_prompts = []
        video_entries = []
        video_paths = []
        video_prompts = []
        
        for entry, path, prompt in zip(valid_entries, media_paths, prompts):
            from pathlib import Path
            if Path(path).suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
                video_entries.append(entry)
                video_paths.append(path)
                video_prompts.append(prompt)
            else:
                image_entries.append(entry)
                image_paths.append(path)
                image_prompts.append(prompt)
        
        bt.logging.info(
            f"Split into {len(image_entries)} images and {len(video_entries)} videos"
        )

        all_results = []
        
        # Process images with large batch size (128)
        if image_entries:
            bt.logging.info(f"Processing {len(image_entries)} images (batch_size=128)")
            image_consensus = calculate_clip_alignment_consensus(
                image_paths, image_prompts, batch_size=128
            )
            if image_consensus:
                all_results.extend(zip(image_entries, image_prompts, image_consensus))
        
        # Process videos with small batch size (32)
        if video_entries:
            bt.logging.info(f"Processing {len(video_entries)} videos (batch_size=32)")
            video_consensus = calculate_clip_alignment_consensus(
                video_paths, video_prompts, batch_size=32
            )
            if video_consensus:
                all_results.extend(zip(video_entries, video_prompts, video_consensus))
        
        if not all_results:
            bt.logging.error("Consensus verification failed for all media")
            return [
                VerificationResult(media_entry=entry, passed=False)
                for entry in valid_entries
            ]

        # Create verification results and detect corrupted videos
        verification_results = []
        corrupted_videos = []

        for media_entry, prompt, consensus_result in all_results:
            # Check if video was corrupted/unreadable (indicated by None score from all models)
            if consensus_result.get("corrupted", False) or consensus_result["consensus_score"] is None:
                bt.logging.warning(
                    f"Corrupted/invalid video detected: {media_entry.file_path} (ID: {media_entry.id})"
                )
                corrupted_videos.append(media_entry)
                # Still create a failed result for this entry
                result = VerificationResult(
                    media_entry=media_entry,
                    original_prompt=prompt,
                    verification_score=None,
                    passed=False,
                )
                verification_results.append(result)
                continue
            
            consensus_score = consensus_result["consensus_score"]
            passed = consensus_score >= threshold

            verification_score = {
                "score": consensus_score,
                "clip_score": consensus_score,
                "original_prompt": prompt,
                "method": "clip_consensus",
                "consensus_details": consensus_result,
                "passed": passed,
                "threshold": threshold,
            }

            result = VerificationResult(
                media_entry=media_entry,
                original_prompt=prompt,
                verification_score=verification_score,
                passed=passed,
            )

            verification_results.append(result)

            # Log individual results
            bt.logging.debug(
                f"Media {media_entry.id}: score={consensus_score:.3f}, "
                f"passed={'✅' if passed else '❌'}, "
                f"models={consensus_result['num_models']}/3"
            )

        # Delete corrupted/invalid videos
        if corrupted_videos:
            bt.logging.warning(
                f"Deleting {len(corrupted_videos)} corrupted/invalid video(s) from storage"
            )
            for media_entry in corrupted_videos:
                try:
                    success = content_manager.delete_media(media_id=media_entry.id)
                    if success:
                        bt.logging.info(f"Deleted corrupted video: {media_entry.file_path}")
                    else:
                        bt.logging.error(f"Failed to delete corrupted video: {media_entry.file_path}")
                except Exception as e:
                    bt.logging.error(f"Error deleting corrupted video {media_entry.file_path}: {e}")

        # Summary logging
        successful_count = len(verification_results)
        passed_count = sum(1 for r in verification_results if r.passed)
        # Only compute average from results that have valid scores
        valid_scores = [r.verification_score["score"] for r in verification_results if r.verification_score]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        bt.logging.info(
            f"✅ Verification complete: {successful_count}/{len(media_entries)} processed, "
            f"{passed_count} passed (pass rate: {passed_count/successful_count:.1%}, "
            f"avg score: {avg_score:.3f})"
        )

        return verification_results

    except Exception as e:
        bt.logging.error(f"Error in batch verification: {e}")
        return [
            VerificationResult(media_entry=entry, passed=False)
            for entry in media_entries
        ]


def get_verification_summary(results: List[VerificationResult]) -> Dict[str, Any]:
    """
    Generate a summary of verification results.

    Args:
        results: List of VerificationResult objects

    Returns:
        Dictionary with summary statistics
    """
    if not results:
        return {
            "total": 0,
            "successful": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "error_rate": 0.0,
            "pass_rate": 0.0,
            "fail_rate": 0.0,
            "average_score": 0.0,
        }

    total = len(results)
    successful = sum(1 for r in results if r.verification_score is not None)
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if r.verification_score is not None and not r.passed)
    errors = sum(1 for r in results if r.verification_score is None)

    scores = [
        (
            r.verification_score.get("score", 0.0)
            if isinstance(r.verification_score, dict)
            else r.verification_score
        )
        for r in results
        if r.verification_score is not None
    ]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "total": total,
        "successful": successful,  # Successfully processed (passed + failed)
        "passed": passed,          # Passed verification threshold
        "failed": failed,          # Failed verification threshold  
        "errors": errors,          # Errors during verification process
        "error_rate": errors / total if total > 0 else 0.0,
        "pass_rate": passed / successful if successful > 0 else 0.0,
        "fail_rate": failed / successful if successful > 0 else 0.0,
        "average_score": avg_score,
    }

