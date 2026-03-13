import math
import time
from typing import Dict, Optional

import bittensor as bt


def get_discriminator_rewards(
    runs,
    metagraph,
    image_score_weight: float = 0.5,
    video_score_weight: float = 0.5,
    binary_score_weight: float = 0.5,
    multiclass_score_weight: float = 0.5,
):
    """
    Process discriminator results to extract rewards from MCC scores.
    Handles separate image and video detection models per miner.

    Args:
        runs: List of DiscriminatorResult objects from API
        metagraph: Bittensor metagraph for SS58 to UID mapping
        image_score_weight: Weight for image modality rewards
        video_score_weight: Weight for video modality rewards
        binary_score_weight: Weight for binary MCC
        multiclass_score_weight: Weight for multiclass MCC

    Returns:
        dict: Mapping of UID to combined reward score for discriminators
    """
    
    # Store rewards by UID and modality
    miner_modality_rewards = {}
    ss58_to_uid = {hotkey: uid for uid, hotkey in enumerate(metagraph.hotkeys)}

    if not runs:
        bt.logging.warning("No discriminator runs data provided")
        return {}

    try:
        for result in runs:
            if not isinstance(result, dict):
                bt.logging.warning(f"Invalid result format: {type(result)}")
                continue

            ss58_address = result.get("discriminator_address")
            if not ss58_address or ss58_address not in ss58_to_uid:
                continue

            uid = ss58_to_uid[ss58_address]
            modality = result.get("modality")
            if not modality:
                bt.logging.warning(f"Missing modality for UID {uid}")
                continue

            if uid not in miner_modality_rewards:
                miner_modality_rewards[uid] = {}

            # Handle potential None or non-numeric values
            binary_mcc = result.get("binary_mcc", 0)
            multiclass_mcc = result.get("multiclass_mcc", 0)

            try:
                binary_mcc = float(binary_mcc) if binary_mcc is not None else 0
                multiclass_mcc = float(multiclass_mcc) if multiclass_mcc is not None else 0
            except (ValueError, TypeError):
                bt.logging.warning(f"Invalid MCC values for UID {uid}: binary={binary_mcc}, multiclass={multiclass_mcc}")
                binary_mcc = 0
                multiclass_mcc = 0

            miner_modality_rewards[uid][modality] = binary_score_weight * max(
                0, binary_mcc
            ) + multiclass_score_weight * max(0, multiclass_mcc)

    except Exception as e:
        bt.logging.error(f"Error processing discriminator rewards: {e}")
        import traceback
        bt.logging.error(traceback.format_exc())

    # Combine image and video rewards for each miner
    final_multipliers = {}
    for uid, modality_rewards in miner_modality_rewards.items():
        final_multipliers[uid] = image_score_weight * modality_rewards.get(
            "image", 0.0
        ) + video_score_weight * modality_rewards.get("video", 0.0)

    return final_multipliers


def get_generator_base_rewards(verification_stats):
    """
    Compute base rewards for generators based on their verification pass rates.

    Args:
        verification_stats: Dict mapping hotkey to verification stats from
                            ContentManager.get_unrewarded_verification_stats()
            Expected format:
            {
                "hotkey": {
                    "uid": int,
                    "total_verified": int,
                    "total_failed": int,
                    "total_evaluated": int,
                    "pass_rate": float,
                    "media_ids": List[str]
                }
            }

    Returns:
        tuple: (uid_rewards_dict, media_ids_to_mark)
            - uid_rewards_dict: Mapping of UID to base reward score
            - media_ids_to_mark: List of media IDs to mark as rewarded
    """
    try:
        if not verification_stats:
            return {}, []

        # Convert to UID-based rewards and collect media IDs
        uid_rewards = {}
        all_media_ids = []

        for hotkey, stats in verification_stats.items():
            uid = int(stats["uid"])
            pass_rate = stats["pass_rate"]
            verified_count = stats["total_verified"]

            # Simple base reward calculation (can be customized)
            # Higher pass rate = higher reward, with bonus for volume
            base_reward = pass_rate * min(verified_count, 10)  # Cap volume bonus at 10

            uid_rewards[uid] = base_reward
            all_media_ids.extend(stats["media_ids"])

        bt.logging.info(f"Computed base rewards for {len(uid_rewards)} miners: {uid_rewards}")

        return uid_rewards, all_media_ids

    except Exception as e:
        bt.logging.error(f"Error in get_generator_base_rewards: {e}")
        import traceback

        bt.logging.error(traceback.format_exc())
        return {}, []


def get_generator_reward_multipliers(
    generator_results, 
    metagraph,
    generator_liveness: Optional[Dict[str, float]] = None,
    max_inactive_hours: int = 24,
):
    """
    Process generator results to extract rewards from fool counts.
    
    Optionally filters out inactive generators based on liveness tracking.

    Args:
        generator_results: List of GeneratorResult objects from API
        metagraph: Bittensor metagraph for SS58 to UID mapping
        generator_liveness: Optional dict mapping hotkey to last activity timestamp.
                           If provided, generators not seen within max_inactive_hours will get 0 rewards.
        max_inactive_hours: Maximum hours of inactivity before generator is considered inactive (default: 48)

    Returns:
        dict: Mapping of UID to reward score for generators
    """    
    rewards = {}
    ss58_to_uid = {hotkey: uid for uid, hotkey in enumerate(metagraph.hotkeys)}

    if not generator_results:
        bt.logging.warning("No generator results data provided")
        return {}

    # Build set of active generators if liveness data provided
    active_generators = None
    inactive_count = 0
    if generator_liveness:
        current_time = time.time()
        max_inactive_seconds = max_inactive_hours * 3600
        active_generators = {
            hotkey for hotkey, last_seen in generator_liveness.items()
            if (current_time - last_seen) <= max_inactive_seconds
        }
        bt.logging.info(f"Liveness filter: {len(active_generators)} active generators (within {max_inactive_hours}h)")

    # Aggregate fool counts by generator (ss58_address)
    generator_fooled_counts = {}
    generator_not_fooled_counts = {}

    try:
        for result in generator_results:
            if not isinstance(result, dict):
                bt.logging.warning(f"Invalid result format: {type(result)}")
                continue                
            ss58_address = result.get("ss58_address")
            if not ss58_address or ss58_address not in ss58_to_uid:
                continue

            # Skip inactive generators if liveness filter is enabled
            if active_generators is not None and ss58_address not in active_generators:
                inactive_count += 1
                bt.logging.debug(f"Skipping inactive generator {ss58_address[:16]}...")
                continue

            fooled_count = result.get("fooled_count", 0)
            not_fooled_count = result.get("not_fooled_count", 0)
            
            try:
                fooled_count = int(fooled_count) if fooled_count is not None else 0
                not_fooled_count = int(not_fooled_count) if not_fooled_count is not None else 0
            except (ValueError, TypeError):
                bt.logging.warning(f"Invalid counts for {ss58_address}: fooled={fooled_count}, not_fooled={not_fooled_count}")
                fooled_count = 0
                not_fooled_count = 0

            if ss58_address not in generator_fooled_counts:
                generator_fooled_counts[ss58_address] = 0
                generator_not_fooled_counts[ss58_address] = 0

            generator_fooled_counts[ss58_address] += fooled_count
            generator_not_fooled_counts[ss58_address] += not_fooled_count

        # Calculate fool rate for each generator from accumulated counts with sample size bonus
        for ss58_address in generator_fooled_counts:
            if ss58_address in ss58_to_uid:
                uid = ss58_to_uid[ss58_address]
                total_fooled = generator_fooled_counts[ss58_address]
                total_not_fooled = generator_not_fooled_counts[ss58_address]
                total_count = total_fooled + total_not_fooled

                if total_count > 0:
                    # Base fool rate
                    fool_rate = total_fooled / total_count

                    # Sample size multiplier: rewards higher sample sizes
                    # Uses logarithmic scaling to provide diminishing returns
                    # Reference count of 20 gives multiplier of 1.0, higher counts get bonus
                    reference_count = 20
                    max_multiplier = 2.0  # Cap the maximum multiplier

                    if total_count >= reference_count:
                        sample_size_multiplier = min(max_multiplier, 1.0 + math.log(total_count / reference_count))
                    else:
                        # Penalize very small sample sizes
                        sample_size_multiplier = max(0.5, total_count / reference_count)

                    # Final reward combines fool rate with sample size bonus
                    base_reward = fool_rate * sample_size_multiplier
                    rewards[uid] = max(0, min(2.0, base_reward))  # Allow rewards up to 2.0 for high sample sizes

                    bt.logging.debug(f"Generator {ss58_address[:8]}... UID {uid}: fool_rate={fool_rate:.3f}, "
                                   f"sample_size={total_count}, sample_size_multiplier={sample_size_multiplier:.3f}, "
                                   f"final_multiplier={rewards[uid]:.3f}")
                else:
                    bt.logging.warning(f"Zero total count for generator {ss58_address}")

        inactive_msg = f", skipped {inactive_count} inactive" if inactive_count > 0 else ""
        bt.logging.info(f"Processed {len(generator_results)} generator results, computed rewards for {len(rewards)} generators{inactive_msg}")

    except Exception as e:
        bt.logging.error(f"Error processing generator rewards: {e}")
        import traceback
        bt.logging.error(traceback.format_exc())

    return rewards
