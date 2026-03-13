import asyncio
import io
import os
import pickle
import random
import tempfile
import threading
import time

import aiohttp
import bittensor as bt
import cv2
import numpy as np
import requests
import uvicorn
from bittensor.core.axon import FastAPIThreadedServer
from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.responses import Response
from pathlib import Path
from PIL import Image
from typing import Any, Dict, Optional

from gas.cache.content_db import PromptEntry
from gas.cache.content_manager import ContentManager
from gas.protocol.epistula import generate_header, get_verifier
from gas.protocol.validator_requests import query_generative_miner
from gas.types import MediaType, MinerType, Modality
from gas.verification.c2pa_verification import verify_c2pa
from gas.verification.duplicate_detection import compute_media_hash, DEFAULT_HAMMING_THRESHOLD


class GenerativeChallengeManager:
    def __init__(
        self,
        config,
        wallet,
        metagraph,
        subtensor,
        miner_type_tracker,
    ):
        self.config = config
        self.wallet = wallet
        self.metagraph = metagraph
        self.subtensor = subtensor
        self.miner_type_tracker = miner_type_tracker

        self.content_manager = ContentManager(self.config.cache.base_dir)

        self.challenge_tasks = {}
        # Use threading.Lock instead of asyncio.Lock because FastAPIThreadedServer
        # runs in a separate thread with its own event loop
        self.challenge_lock = threading.Lock()

        # Track generator liveness: hotkey -> last activity timestamp
        # Updated when a generator successfully responds to a challenge
        self.generator_last_seen: Dict[str, float] = {}

        self.external_port = (
            getattr(self.config.neuron, 'external_callback_port', None) or
            self.config.neuron.callback_port
        )

        try:
            self.external_ip = requests.get("https://checkip.amazonaws.com", timeout=10).text.strip()
        except Exception as e:
            bt.logging.error(f"Failed to get external IP: {e}. Using fallback.")
            self.external_ip = "localhost"
        self.generative_callback_url = f"http://{self.external_ip}:{self.external_port}/generative_callback"

        self.init_fastapi()

    async def issue_generative_challenge(self):
        await self.miner_type_tracker.update_miner_types()
        miner_uids = self.miner_type_tracker.get_miners_by_type(MinerType.GENERATOR)

        if len(miner_uids) > self.config.neuron.sample_size:
            miner_uids = np.random.choice(
                miner_uids,
                size=self.config.neuron.sample_size,
                replace=False,
            ).tolist()

        if not miner_uids:
            bt.logging.trace("No generative miners found to challenge.")
            return

        bt.logging.info(f"Issuing generative challenge to UIDs: {miner_uids}")

        # First select a random modality, then sample a prompt matching that modality
        modality = random.choice([Modality.IMAGE, Modality.VIDEO])

        retries = 3
        prompt_entry = None
        for _ in range(retries):
            # Try to get a prompt for the selected modality
            prompts = self.content_manager.sample_prompts(k=1, modality=modality.value, remove=False)
            if len(prompts) > 0:
                prompt_entry = prompts[0]
                break
            # Fallback: try without modality filter (for backwards compatibility with old prompts)
            prompts = self.content_manager.sample_prompts(k=1, remove=False)
            if len(prompts) > 0:
                prompt_entry = prompts[0]
                bt.logging.debug(f"Using legacy prompt without modality for {modality.value} challenge")
                break

        if not prompt_entry:
            bt.logging.info(
                "Waiting for prompt cache to be populated. Skipping generative challenge."
            )
            return

        await asyncio.gather(
            *[self.send_generative_request(uid, prompt_entry, modality) for uid in miner_uids]
        )

    async def send_generative_request(self, uid: int, prompt_entry, modality: Modality):
        """Scoring is handled by the callback in GeneratorEvaluator"""

        #parameters = {"width": 1024, "height": 1024}

        async with aiohttp.ClientSession() as session:
            response_data = await query_generative_miner(
                uid=uid,
                axon_info=self.metagraph.axons[uid],
                session=session,
                hotkey=self.wallet.hotkey,
                prompt=prompt_entry.content,
                modality=modality,
                webhook_url=self.generative_callback_url,
                parameters=None,
                total_timeout=self.config.neuron.miner_total_timeout,
            )

        if response_data and response_data.get("task_id"):
            miner_task_id = response_data.get("task_id")
            with self.challenge_lock:
                self.challenge_tasks[miner_task_id] = {
                    "uid": uid,
                    "prompt_id": prompt_entry.id,
                    "prompt_content": prompt_entry.content,
                    "modality": modality,
                    "media_type": MediaType.SYNTHETIC,
                    "status": "pending",
                    "sent_at": time.time(),
                }
            bt.logging.info(
                f"Stored challenge task {miner_task_id} for UID {uid}. Total active tasks: {len(self.challenge_tasks)}"
            )
        else:
            error = response_data.get("error") if response_data else "Unknown error"
            bt.logging.error(f"Failed to send challenge to UID {uid}. Error: {error}")

    async def generative_callback(self, request: Request):
        """Callback endpoint for generative challenges.
        Accepts direct binary image, video, and application/octet-stream payloads.
        """
        content_type = request.headers.get("content-type", "").lower()
        task_id = request.headers.get("task-id")
        client_ip = request.client.host if request.client else "unknown"

        uid = "unknown"
        signed_by = request.headers.get("Epistula-Signed-By")
        if task_id and task_id in self.challenge_tasks:
            uid = self.challenge_tasks[task_id].get("uid", "unknown")
        elif signed_by and signed_by in self.metagraph.hotkeys:
            try:
                uid = self.metagraph.hotkeys.index(signed_by)
            except (ValueError, AttributeError):
                pass

        bt.logging.debug(f"Generative callback request from UID {uid} (IP: {client_ip}), task_id: {task_id}, content_type: {content_type}")

        # Helper function to format UID with hotkey for better debugging
        def format_uid_info():
            if uid == "unknown" and signed_by:
                return f"UID {uid} (signed-by: {signed_by})"
            return f"UID {uid}"

        if not (
            content_type.startswith("image/") or
            content_type.startswith("video/") or
            content_type == "application/octet-stream"
        ):
            bt.logging.error(f"Invalid content type: {content_type} from {format_uid_info()} (IP: {client_ip}). Only image/*, video/*, and application/octet-stream are supported.")
            return Response(status_code=400, content="Only image, video, and application/octet-stream content types are supported")

        if not task_id:
            bt.logging.error(f"Binary upload missing task-id header from {format_uid_info()} (IP: {client_ip})")
            return Response(status_code=400, content="Missing task-id header")

        binary_data = await request.body()
        if not binary_data:
            bt.logging.error(f"Task {task_id} from {format_uid_info()} (IP: {client_ip}): Empty binary payload received")
            return Response(status_code=400, content="Empty binary payload")

        # copy the challenge info and release the lock before doing async work.
        with self.challenge_lock:
            bt.logging.debug(f"Callback for task {task_id}: Current active tasks: {list(self.challenge_tasks.keys())}")
            if task_id not in self.challenge_tasks:
                # Check if this might be a stale task from a previous session
                bt.logging.debug(f"Received binary upload for unknown task_id: {task_id} from {format_uid_info()} (IP: {client_ip}), content_type: {content_type}, size: {len(binary_data)} bytes")
                # Accept the upload gracefully but don't process it - this reduces 404 spam
                # while still indicating the task wasn't found in our debug logs
                return Response(status_code=200, content="Task not found in current session")

            # Copy the challenge info so we can release the lock before async work
            challenge_info = self.challenge_tasks[task_id].copy()
            generator_uid = challenge_info["uid"]

        auth_uid_msg = f" (auth UID: {uid})" if uid != generator_uid and uid != "unknown" else ""
        bt.logging.info(
            f"Received binary upload for task {task_id}, UID {generator_uid}{auth_uid_msg}, "
            f"type: {content_type}, size: {len(binary_data)} bytes (IP: {client_ip})"
        )

        # Perform async storage work outside the lock
        filepath, error_message = await self.store_binary_content(
            binary_data, content_type, generator_uid, task_id
        )

        # Re-acquire lock to update task status and clean up
        with self.challenge_lock:
            # Check if task still exists (could have been cleaned up by another process)
            if task_id not in self.challenge_tasks:
                if filepath:
                    bt.logging.debug(f"Task {task_id} already removed, but content was stored at {filepath}")
                    return Response(status_code=200, content="Binary content received")
                return Response(status_code=200, content="Task already processed")

            if filepath:
                self.challenge_tasks[task_id]["status"] = "completed"
                self.challenge_tasks[task_id]["filepath"] = filepath
                bt.logging.success(f"Task {task_id} completed with binary upload: {filepath}")

                # Track generator liveness - record when they successfully responded
                miner_hotkey = self.metagraph.hotkeys[generator_uid]
                self.generator_last_seen[miner_hotkey] = time.time()
                bt.logging.debug(f"Updated liveness for generator {miner_hotkey[:16]}... (UID {generator_uid})")

                del self.challenge_tasks[task_id]
                return Response(status_code=200, content="Binary content received")
            else:
                self.challenge_tasks[task_id]["status"] = "failed"
                self.challenge_tasks[task_id]["error_message"] = error_message or "Failed to store binary content"

                del self.challenge_tasks[task_id]
                return Response(status_code=400, content=error_message or "Failed to store binary content")

    async def store_binary_content(
        self, binary_data: bytes, content_type: str, generator_uid: int, task_id: str
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Store binary content directly uploaded by miner using ContentManager.

        Performs pre-storage validation and REJECTS:
        - Duplicate content (perceptual hash match)
        - Content without valid C2PA from trusted AI generators
        - Corrupted/unreadable media

        Only content that passes all checks is stored and eligible for HuggingFace upload.

        Returns:
            Tuple of (filepath, error_message). On success, filepath is set and error is None.
            On failure, filepath is None and error contains the rejection reason.
        """
        try:
            bt.logging.trace(f"Storing binary content for task {task_id}, size: {len(binary_data)} bytes")

            # Get task info from challenge tracker
            task_info = self.challenge_tasks.get(task_id)
            if not task_info:
                bt.logging.error(f"Task {task_id} not found in challenge tasks")
                return None, "Task not found in challenge tasks"

            modality = task_info["modality"]
            media_type = task_info["media_type"]
            prompt_id = task_info["prompt_id"]

            # Get modality string for checks
            modality_str = modality.value if hasattr(modality, 'value') else str(modality)
            miner_hotkey = self.metagraph.hotkeys[generator_uid]

            # Step 1: Check for corrupted/unreadable media
            try:
                is_corrupted = self._check_media_corrupted(binary_data, modality_str)
                if is_corrupted:
                    bt.logging.warning(
                        f"REJECTED corrupted media from UID {generator_uid} task {task_id}: "
                        f"media is unreadable or invalid"
                    )
                    return None, "Corrupted or unreadable media"
            except Exception as e:
                bt.logging.debug(f"Corruption check error (allowing): {e}")

            # Step 2: Compute perceptual hash and check for duplicates within same prompt
            perceptual_hash = None
            try:
                perceptual_hash = compute_media_hash(binary_data, modality=modality_str)
                if perceptual_hash:
                    duplicate_info = self.content_manager.check_duplicate(
                        perceptual_hash,
                        threshold=DEFAULT_HAMMING_THRESHOLD,
                        prompt_id=prompt_id,
                    )
                    if duplicate_info:
                        dup_media_id, dup_distance = duplicate_info
                        bt.logging.warning(
                            f"REJECTED duplicate from UID {generator_uid} task {task_id}: "
                            f"matches media {dup_media_id} with distance {dup_distance}"
                        )
                        return None, "Duplicate content detected"
            except Exception as e:
                bt.logging.debug(f"Duplicate detection skipped: {e}")

            # Step 3: Verify C2PA content credentials - REQUIRE trusted issuer
            c2pa_verified = False
            c2pa_issuer = None
            c2pa_model_name = None
            try:
                c2pa_result = verify_c2pa(binary_data)
                if c2pa_result.verified and c2pa_result.is_trusted_issuer:
                    c2pa_verified = True
                    c2pa_issuer = c2pa_result.issuer
                    c2pa_model_name = c2pa_result.model_name
                    bt.logging.info(
                        f"C2PA verified for UID {generator_uid}: issuer={c2pa_issuer}, model_name={c2pa_model_name}"
                    )
                else:
                    rejection_reason = c2pa_result.error or (
                        "untrusted issuer" if c2pa_result.signature_valid else "no valid C2PA manifest"
                    )
                    bt.logging.warning(
                        f"REJECTED from UID {generator_uid} task {task_id}: "
                        f"C2PA check failed ({rejection_reason})"
                    )
                    return None, f"C2PA verification failed: {rejection_reason}"
            except Exception as e:
                bt.logging.error(
                    f"REJECTED from UID {generator_uid} task {task_id}: "
                    f"C2PA verification error: {e}"
                )
                return None, f"C2PA verification error: {e}"

            # Step 4: All checks passed - store the media
            filepath = self.content_manager.write_miner_media(
                modality=modality,
                media_type=media_type,
                prompt_id=prompt_id,
                uid=generator_uid,
                hotkey=miner_hotkey,
                media_content=binary_data,
                content_type=content_type,
                task_id=task_id,
                model_name=c2pa_model_name,
                perceptual_hash=perceptual_hash,
                c2pa_verified=c2pa_verified,
                c2pa_issuer=c2pa_issuer,
            )

            if filepath:
                bt.logging.success(
                    f"Stored verified content: {filepath} (size: {len(binary_data)} bytes, "
                    f"hash: {perceptual_hash[:16] if perceptual_hash else 'N/A'}..., "
                    f"c2pa_issuer: {c2pa_issuer}, model_name: {c2pa_model_name})"
                )
                return filepath, None
            else:
                bt.logging.error("ContentManager failed to store binary content")
                return None, "Failed to store binary content"

        except Exception as e:
            bt.logging.error(f"Error storing binary content: {e}")
            return None, f"Internal error: {e}"

    def _check_media_corrupted(self, binary_data: bytes, modality: str) -> bool:
        """
        Check if media data is corrupted/unreadable.

        Returns:
            True if corrupted, False if valid
        """
        try:
            if modality == "image":
                img = Image.open(io.BytesIO(binary_data))
                img.verify()  # Verify it's a valid image
                return False
            elif modality == "video":
                # Write to temp file and try to open with cv2
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(binary_data)
                    tmp_path = tmp.name

                try:
                    cap = cv2.VideoCapture(tmp_path)
                    if not cap.isOpened():
                        return True

                    # Try to read at least one frame
                    ret, frame = cap.read()
                    cap.release()

                    if not ret or frame is None:
                        return True

                    return False
                finally:
                    os.unlink(tmp_path)
            else:
                return False  # Unknown modality, don't reject

        except Exception as e:
            bt.logging.debug(f"Media corruption check failed: {e}")
            return True  # If we can't verify, assume corrupted


    def init_fastapi(self):
        """Initialize the FastAPI server for generative challenge callbacks."""
        if self.config.neuron.callback_port is None:
            bt.logging.warning(
                "Callback port not set, generative challenges will not have a callback."
            )
            return

        self.api = FastAPI()
        self.router = APIRouter()

        verifier = get_verifier(
            self.wallet, self.metagraph, no_force_validator_permit=True
        )

        self.router.add_api_route(
            "/generative_callback",
            self.generative_callback,
            dependencies=[Depends(verifier)],
            methods=["POST"],
            include_in_schema=False,
        )
        self.api.include_router(self.router)

        fast_config = uvicorn.Config(
            self.api,
            host="0.0.0.0",
            port=self.config.neuron.callback_port,
            log_level="info",
            loop="asyncio",
        )
        self.fast_api = FastAPIThreadedServer(config=fast_config)
        self.fast_api.start()

        bt.logging.info(
            f"FastAPI server started on port {self.config.neuron.callback_port}"
        )
        if self.external_port != self.config.neuron.callback_port:
            bt.logging.info(
                f"Advertising external port {self.external_port} to miners"
            )
        bt.logging.info(
            f"Callback URL for miners: {self.generative_callback_url}"
        )

    def get_generator_last_seen(self, hotkey: str) -> Optional[float]:
        """Get the last activity timestamp for a generator.

        Args:
            hotkey: The generator's hotkey (SS58 address)

        Returns:
            Unix timestamp of last activity, or None if never seen
        """
        return self.generator_last_seen.get(hotkey)

    def get_all_generator_last_seen(self) -> Dict[str, float]:
        """Get all generator last seen timestamps.

        Returns:
            Dict mapping hotkey to last activity timestamp
        """
        return self.generator_last_seen.copy()

    def is_generator_active(self, hotkey: str, max_inactive_hours: int = 24) -> bool:
        """Check if a generator has been active within the specified window.

        Args:
            hotkey: The generator's hotkey (SS58 address)
            max_inactive_hours: Maximum hours of inactivity before considered inactive

        Returns:
            True if generator was active within the window, False otherwise
        """
        last_seen = self.generator_last_seen.get(hotkey)
        if last_seen is None:
            return False

        hours_since_seen = (time.time() - last_seen) / 3600
        return hours_since_seen <= max_inactive_hours

    def get_active_generators(self, max_inactive_hours: int = 24) -> Dict[str, float]:
        """Get all generators that have been active within the specified window.

        Args:
            max_inactive_hours: Maximum hours of inactivity before considered inactive

        Returns:
            Dict mapping hotkey to last activity timestamp for active generators
        """
        current_time = time.time()
        max_inactive_seconds = max_inactive_hours * 3600

        return {
            hotkey: last_seen
            for hotkey, last_seen in self.generator_last_seen.items()
            if (current_time - last_seen) <= max_inactive_seconds
        }

    async def shutdown(self):
        """Shutdown the webhook server gracefully"""
        if hasattr(self, "fast_api"):
            bt.logging.info("Shutting down webhook server...")
            self.fast_api.stop()
            bt.logging.info("Webhook server stopped")

    def save_state(self, save_dir: str, filename: str):
        """Save challenge tasks state and generator liveness to disk"""
        try:
            with self.challenge_lock:
                bt.logging.info(f"Saving challenge tasks state: {len(self.challenge_tasks)} active tasks")
                current_time = time.time()
                stale_tasks = []
                for task_id, task_info in self.challenge_tasks.items():
                    if current_time - task_info.get("sent_at", 0) > 7200:
                        stale_tasks.append(task_id)

                for task_id in stale_tasks:
                    del self.challenge_tasks[task_id]
                    bt.logging.debug(f"Removed stale task {task_id} during state save")

                tasks_to_save = self.challenge_tasks.copy()

            filepath = os.path.join(save_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(tasks_to_save, f)
            bt.logging.info(f"Successfully saved {len(tasks_to_save)} active challenge tasks to {filepath}")

            liveness_filepath = os.path.join(save_dir, "generator_liveness.pkl")
            max_liveness_age = 7 * 24 * 3600
            valid_liveness = {
                hotkey: ts
                for hotkey, ts in self.generator_last_seen.items()
                if (current_time - ts) <= max_liveness_age
            }
            with open(liveness_filepath, 'wb') as f:
                pickle.dump(valid_liveness, f)
            bt.logging.info(f"Successfully saved liveness data for {len(valid_liveness)} generators")
        except Exception as e:
            bt.logging.error(f"Failed to save challenge tasks state: {e}")

    def load_state(self, save_dir: str, filename: str):
        """Load challenge tasks state and generator liveness from disk"""
        try:
            filepath = os.path.join(save_dir, filename)
            valid_tasks = {}
            if not os.path.exists(filepath):
                bt.logging.debug(f"No challenge tasks state file found at {filepath}")
            else:
                with open(filepath, 'rb') as f:
                    loaded_tasks = pickle.load(f)

                current_time = time.time()
                for task_id, task_info in loaded_tasks.items():
                    if current_time - task_info.get("sent_at", 0) <= 7200:
                        valid_tasks[task_id] = task_info

            valid_liveness = {}
            liveness_filepath = os.path.join(save_dir, "generator_liveness.pkl")
            if os.path.exists(liveness_filepath):
                with open(liveness_filepath, 'rb') as f:
                    loaded_liveness = pickle.load(f)

                current_time = time.time()
                max_liveness_age = 7 * 24 * 3600  # 7 days
                valid_liveness = {
                    hotkey: ts
                    for hotkey, ts in loaded_liveness.items()
                    if (current_time - ts) <= max_liveness_age
                }

            with self.challenge_lock:
                self.challenge_tasks = valid_tasks
                self.generator_last_seen = valid_liveness

            bt.logging.info(f"Loaded {len(valid_tasks)} active challenge tasks from {filepath}")
            bt.logging.info(f"Loaded liveness data for {len(valid_liveness)} generators")

            return True
        except Exception as e:
            bt.logging.error(f"Failed to load challenge tasks state: {e}")
            with self.challenge_lock:
                self.challenge_tasks = {}
                self.generator_last_seen = {}
            return False
