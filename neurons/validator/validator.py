import asyncio
import sys
import json
import traceback
from time import sleep
from threading import Thread

import numpy as np
from dotenv import load_dotenv
import bittensor as bt
import torch

from gas import __spec_version__ as spec_version
from gas.protocol.validator_requests import get_benchmark_results, get_escrow_addresses
from gas.utils.autoupdater import autoupdate
from gas.cache import ContentManager
from gas.utils.metagraph import create_set_weights
from gas.types import (
    NeuronType,
    MinerType,
)

from gas.utils import (
    on_block_interval,
    print_info,
)
from gas.utils.state_manager import load_validator_state, save_validator_state
from gas.utils.wandb_utils import init_wandb, clean_wandb_cache
from neurons.base import BaseNeuron
from gas.evaluation import (
    GenerativeChallengeManager,
    MinerTypeTracker,
    get_generator_base_rewards,
    get_generator_reward_multipliers,
)

try:
    load_dotenv(".env.validator")
except Exception:
    pass


MAINNET_UID = 34
SS58_ADDRESSES = {
    "burn": "5HjBSeeoz52CLfvDWDkzupqrYLHz1oToDPHjdmJjc4TF68LQ",
    "video_escrow": "5G6BJ1Z6LeDptRn5GTw74QSDmG1FP3eqVque5JhUb5zeEyQa",
    "image_escrow": "5EUJFyH4ZSSiD3C8sM698nsVE26Tq98LoBwkmopmWZqaZqCA",
    "audio_escrow": "5F9Qo4jqurfx3qHsC2kQtvge7Si5aW1BfYKwpxnnpVxouPyF",
}


class Validator(BaseNeuron):
    neuron_type = NeuronType.VALIDATOR

    def __init__(self, config=None):
        super().__init__(config=config)

        self.initialization_complete = False

        ## CHECK IF REGG'D
        if (
            not self.metagraph.validator_permit[self.uid]
            and self.config.netuid == MAINNET_UID
        ):
            bt.logging.error("Validator does not have vpermit")
            sys.exit(1)

        self.init()

    def init(self):
        self.heartbeat_thread: Thread = None
        self.step = 0

        self._state_lock = asyncio.Lock()

        self.content_manager = ContentManager(self.config.cache.base_dir)

        ## Typesafety
        self.set_weights_fn = create_set_weights(spec_version, self.config.netuid)
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)
        bt.logging.info(f"Initialized scores vector for {len(self.scores)} miners")

        if not self.config.wandb_off:
            wandb_dir = self.config.neuron.full_path
            clean_wandb_cache(wandb_dir)
            self.wandb_run = init_wandb(
                self.config,
                "validator",
                self.uid,self.wallet.hotkey,
                wandb_dir
            )

        bt.logging.info(self.config)
        bt.logging.info(f"Last updated at block {self.metagraph.last_update[self.uid]}")

        ## REGISTER BLOCK CALLBACKS
        self.block_callbacks.extend(
            [
                self.log_on_block,
                self.issue_generator_challenge,
                self.set_weights,
            ]
        )

        self.reward_config = {
            "window": self.config.scoring.window,
            "image_score_weight": self.config.scoring.image_weight,
            "video_score_weight": self.config.scoring.video_weight,
            "binary_score_weight": self.config.scoring.binary_weight,
            "multiclass_score_weight": self.config.scoring.multiclass_weight,
        }

        # SETUP HEARTBEAT THREAD
        if self.config.neuron.heartbeat:
            self.heartbeat_thread = Thread(name="heartbeat", target=self.heartbeat)
            self.heartbeat_thread.start()


    async def run(self):

        bt.logging.info(
            f"Running validator {self.uid} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )

        self.miner_type_tracker = MinerTypeTracker(
            self.config, self.wallet, self.metagraph, self.subtensor
        )

        self.generative_challenge_manager = GenerativeChallengeManager(
            self.config,
            self.wallet,
            self.metagraph,
            self.subtensor,
            self.miner_type_tracker,
        )

        await self.load_state()

        dataset_counts = self.content_manager.get_dataset_media_counts()
        total_dataset_media = sum(dataset_counts.values())
        if total_dataset_media == 0:
            bt.logging.warning("No dataset media found.")
        else:
            bt.logging.info(
                f"Found {total_dataset_media} dataset media entries: {dataset_counts}"
            )

        await self.start_substrate_subscription()

        self.initialization_complete = True
        bt.logging.success(
            "\N{GRINNING FACE WITH SMILING EYES}",
            f"Initialization Complete. Validator starting at block: {self.subtensor.block}",
        )

        while not self.exit_context.isExiting:
            self.step += 1
            if self.config.autoupdate and (self.step == 0 or not self.step % 300):
                bt.logging.debug("Checking autoupdate")
                autoupdate(branch="main", install_deps=True)

            self.check_substrate_connection()

            await asyncio.sleep(1)

        await self.shutdown()

    @on_block_interval("generator_challenge_interval")
    async def issue_generator_challenge(self, block):
        """Issue generative challenges and save state to preserve active tasks"""
        await self.generative_challenge_manager.issue_generative_challenge()
        await self.save_state()

    @on_block_interval("epoch_length")
    async def set_weights(self, block):
        """
        Query orchestrator for results, computes rewards, updates scores, set weights
        """
        bt.logging.info(f"Updating scores at block {block}")
        generator_uids = await self.update_scores()
        
        if generator_uids is None:
            generator_uids = []
            bt.logging.warning("No generator rewards available; using empty generator_uids")

        escrow_addresses = await get_escrow_addresses(
            self.wallet.hotkey,
            base_url=self.config.benchmark.api_url,
        )
        
        # Use fetched addresses or fall back to defaults
        if escrow_addresses:
            bt.logging.info("Using escrow addresses from API")
            active_ss58_addresses = {
                "burn": SS58_ADDRESSES["burn"],
                "video_escrow": escrow_addresses.get("video_escrow", SS58_ADDRESSES["video_escrow"]),
                "image_escrow": escrow_addresses.get("image_escrow", SS58_ADDRESSES["image_escrow"]),
                "audio_escrow": escrow_addresses.get("audio_escrow", SS58_ADDRESSES["audio_escrow"]),
            }
        else:
            bt.logging.warning("Failed to fetch escrow addresses from API, using defaults")
            active_ss58_addresses = SS58_ADDRESSES
        
        async with self._state_lock:
            bt.logging.debug("set_weights() acquired state lock")
            try:
                bt.logging.info(f"Setting weights at block {block}")
                self.subtensor = bt.subtensor(
                    config=self.config, network=self.config.subtensor.chain_endpoint
                )
                uids = list(range(self.metagraph.n))

                if np.isnan(self.scores).any():
                    bt.logging.warning(
                        "Scores contain NaN values. This may be due to a lack of "
                        "responses from miners, or a bug in your reward functions."
                    )

                # Weight budget (must sum to 1.0)
                burn_pct      = .40
                video_pct     = .31
                image_pct     = .21
                audio_pct     = .01
                generator_pct = 1 - burn_pct - video_pct - image_pct - audio_pct  # .07

                burn_uid = self.subtensor.get_uid_for_hotkey_on_subnet(
                    hotkey_ss58=active_ss58_addresses["burn"],
                    netuid=self.config.netuid,
                    block=block
                )
                video_escrow_uid = self.subtensor.get_uid_for_hotkey_on_subnet(
                    hotkey_ss58=active_ss58_addresses["video_escrow"],
                    netuid=self.config.netuid,
                    block=block
                )
                image_escrow_uid = self.subtensor.get_uid_for_hotkey_on_subnet(
                    hotkey_ss58=active_ss58_addresses["image_escrow"],
                    netuid=self.config.netuid,
                    block=block
                )
                audio_escrow_uid = self.subtensor.get_uid_for_hotkey_on_subnet(
                    hotkey_ss58=active_ss58_addresses["audio_escrow"],
                    netuid=self.config.netuid,
                    block=block
                )

                special_uids = {burn_uid, image_escrow_uid, video_escrow_uid, audio_escrow_uid}

                # Compute norm excluding specials
                norm = np.ones_like(self.scores)
                active_uids = [uid for uid in generator_uids if uid not in special_uids]
                if active_uids:
                    norm[active_uids] = np.linalg.norm(self.scores[active_uids], ord=1)

                if np.any(norm == 0) or np.isnan(norm).any():
                    norm = np.ones_like(norm)

                normed_weights = self.scores / norm

                active_uids_set = set(active_uids)
                for uid in range(len(normed_weights)):
                    if uid not in special_uids and uid not in active_uids_set:
                        normed_weights[uid] = 0.0

                active_mask = np.array([uid in active_uids_set for uid in range(len(normed_weights))])
                normed_weights[active_mask] *= generator_pct

                normed_weights[burn_uid]         = burn_pct
                normed_weights[video_escrow_uid] = video_pct
                normed_weights[image_escrow_uid] = image_pct
                normed_weights[audio_escrow_uid] = audio_pct

                # Verify allocations
                total_weight = np.sum(normed_weights)
                actual_burn_rate = normed_weights[burn_uid] / total_weight if total_weight > 0 else 0
                bt.logging.info(f"Total weight sum: {total_weight:.4f}, Actual burn rate: {actual_burn_rate:.4f} (target: {burn_pct})")

                self.set_weights_fn(
                    self.wallet, self.metagraph, self.subtensor, (uids, normed_weights)
                )

            except Exception as e:
                bt.logging.error(f"Error in set_weights_on_interval: {e}")
                bt.logging.error(traceback.format_exc())
                return False

        return True

    async def update_scores(self):
        """
        Update self.scores with exponential moving average of rewards.
        """
        # Verification stats from last 2h (all verified, rewarded or not) for base rewards.
        verification_stats = self.content_manager.get_verification_stats_last_n_hours(
            lookback_hours=4.0
        )
        generator_base_rewards, media_ids = get_generator_base_rewards(verification_stats)

        generator_results = await get_benchmark_results(
            self.wallet.hotkey, self.metagraph, base_url=self.config.benchmark.api_url
        )

        # Get generator liveness data for filtering inactive generators
        generator_liveness = None
        max_inactive_hours = 24  # Hours of inactivity before generator rewards are zeroed
        if hasattr(self, 'generative_challenge_manager') and self.generative_challenge_manager:
            generator_liveness = self.generative_challenge_manager.get_all_generator_last_seen()
            if generator_liveness:
                bt.logging.debug(f"Using liveness data for {len(generator_liveness)} generators")
        
        reward_multipliers = get_generator_reward_multipliers(
            generator_results, 
            self.metagraph,
            generator_liveness=generator_liveness,
            max_inactive_hours=max_inactive_hours,
        )
        all_generator_uids = set(generator_base_rewards.keys()) & set(reward_multipliers.keys())
        rewards = {
            uid: generator_base_rewards.get(uid, 0) * reward_multipliers.get(uid, .01)
            for uid in all_generator_uids 
        }

        if len(rewards) == 0:
            if not generator_base_rewards:
                bt.logging.info(
                    "No generator rewards: no verified submissions on this validator in the last 2h."
                )
            else:
                bt.logging.trace(
                    "No generator rewards: no overlap with benchmark multipliers."
                )
            return

        extend_scores = max(list(rewards.keys())) - len(self.scores) + 1
        if extend_scores > 0:
            self.scores = np.append(self.scores, np.zeros(extend_scores))

        reward_arr = np.array([rewards.get(i, 0) for i in range(len(self.scores))])

        # Alpha for generator score EMA - higher = faster decay, less reward persistence
        # 0.5 = 50% new rewards, 50% historical (aggressive decay for inactive miners)
        alpha = 0.5
        self.scores = alpha * reward_arr + (1 - alpha) * self.scores

        # Hard cutoff: zero out scores for generators not active within liveness window
        # This prevents inactive miners from retaining any accumulated score via EMA decay
        if generator_liveness:
            inactive_count = 0
            for uid in range(len(self.scores)):
                if uid < len(self.metagraph.hotkeys):
                    hotkey = self.metagraph.hotkeys[uid]
                    if hotkey not in generator_liveness and self.scores[uid] > 0:
                        self.scores[uid] = 0
                        inactive_count += 1
            if inactive_count > 0:
                bt.logging.info(f"Zeroed scores for {inactive_count} inactive generators (not seen in {max_inactive_hours}h)")

        bt.logging.info(
            f"Updated scores for {len(rewards)} miners with EMA (alpha={alpha})"
        )

        if media_ids:
            success = self.content_manager.mark_media_rewarded(media_ids)
            if success:
                bt.logging.info(f"Marked {len(media_ids)} media entries as rewarded")
            else:
                bt.logging.warning("Failed to mark media as rewarded")

        return list(all_generator_uids)

    async def log_on_block(self, block):
        """
        Log information about validator state at regular intervals.
        """
        try:
            blocks_till = self.config.epoch_length - (block % self.config.epoch_length)
            bt.logging.info(
                f"Forward Block: {block} | Blocks till Set Weights: {blocks_till}"
            )
            print_info(
                self.metagraph,
                self.wallet.hotkey.ss58_address,
                block,
            )
        
        except Exception as e:
            bt.logging.warning(f"Error in log_on_block: {e}")

    async def save_state(self):
        """
        Atomically save validator state (scores + challenge tasks)
        Maintains the current state and one backup.
        """
        async with self._state_lock:
            bt.logging.debug("save_state() acquired state lock")
            try:
                state_data = {"scores.npy": self.scores}
                state_objects = [
                    (self.generative_challenge_manager, "challenge_tasks.pkl")
                ]

                success = save_validator_state(
                    base_dir=self.config.neuron.full_path,
                    state_data=state_data,
                    state_objects=state_objects,
                    max_backup_age_hours=24,
                )
                if success:
                    bt.logging.success("Saved validator state")
                else:
                    bt.logging.error("Failed to save validator state")
            except Exception as e:
                bt.logging.error(f"Error during state save: {str(e)}")
                bt.logging.error(traceback.format_exc())
            finally:
                bt.logging.debug("save_state() releasing state lock")

    async def load_state(self):
        """
        Load validator state, falling back to backup if needed.
        """
        try:
            state_data_keys = ["scores.npy"]
            state_objects = [
                (self.generative_challenge_manager, "challenge_tasks.pkl")
            ]

            loaded_state = load_validator_state(
                base_dir=self.config.neuron.full_path,
                state_data_keys=state_data_keys,
                state_objects=state_objects,
                max_backup_age_hours=24,
            )

            if loaded_state is not None and "scores.npy" in loaded_state:
                self.scores = loaded_state["scores.npy"]
                bt.logging.info(f"Loaded scores vector for {len(self.scores)} miners")
                return True
            else:
                bt.logging.warning("No valid state found")
                return False

        except Exception as e:
            bt.logging.error(f"Error during state load: {str(e)}")
            bt.logging.error(traceback.format_exc())
            return False

    async def shutdown(self):
        """Shutdown the validator and clean up resources."""
        await self.shutdown_substrate()
        if self.generative_challenge_manager:
           await self.generative_challenge_manager.shutdown()

    def heartbeat(self):
        bt.logging.info("Starting Heartbeat")
        last_step = self.step
        stuck_count = 0
        while True:
            sleep(self.config.neuron.heartbeat_interval_seconds)
            if last_step == self.step:
                stuck_count += 1
            if last_step != self.step:
                stuck_count = 0
            if stuck_count >= self.config.neuron.max_stuck_count:
                bt.logging.error(
                    "Heartbeat detecting main process hang, attempting restart"
                )
                autoupdate(force=True, install_deps=True)
                sys.exit(0)
            last_step = self.step
            bt.logging.info("Heartbeat")


if __name__ == "__main__":
    try:
        validator = Validator()
        asyncio.run(validator.run())
    except KeyboardInterrupt:
        bt.logging.info("Validator interrupted by KeyboardInterrupt, shutting down")
    except Exception as e:
        bt.logging.error(f"Unhandled exception: {e}")
        bt.logging.error(traceback.format_exc())
