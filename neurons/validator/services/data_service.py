import argparse
import asyncio
import inspect
import sys
import os
import tempfile
import time
import traceback
from itertools import zip_longest
from threading import Event, Thread
from typing import List, Optional
from pathlib import Path
import math

import bittensor as bt
from bittensor.core.settings import SS58_FORMAT, TYPE_REGISTRY

from gas.cache import ContentManager
from gas.config import add_args, add_data_service_args
from gas.datasets import load_all_datasets
from gas.datasets.download import download_and_extract
from gas.scraping import GoogleScraper
from gas.types import MediaType, Modality, SourceType
from gas.utils import on_block_interval
from gas.utils.metagraph import SubstrateConnectionManager


class DataService:
    """
    Simplified data service that handles both scraping and dataset downloads directly.
    Eliminates unnecessary OOP abstractions and manages threading internally.
    """

    def __init__(self, config):
        self.config = config

        if hasattr(self.config, "cache") and hasattr(self.config.cache, "base_dir"):
            self.config.cache.base_dir = str(
                Path(self.config.cache.base_dir).expanduser()
            )

        # Ensure temporary files live under the cache directory to avoid /tmp OOM
        try:
            tmp_base = Path(self.config.cache.base_dir) / "tmp"
            tmp_base.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("TMPDIR", str(tmp_base))
            os.environ.setdefault("TEMP", str(tmp_base))
            os.environ.setdefault("TMP", str(tmp_base))
            tempfile.tempdir = str(tmp_base)
            bt.logging.info(f"[DATA-SERVICE] Using temp dir: {tmp_base}")
        except Exception as e:
            bt.logging.warning(f"[DATA-SERVICE] Failed to set temp dir under cache: {e}")

        # used for filesystem/db writes
        self.content_manager = ContentManager(
            base_dir=self.config.cache.base_dir,
            max_per_source=self.config.max_per_source,
            enable_source_limits=self.config.enable_source_limits,
            prune_strategy=self.config.prune_strategy,
            remove_on_sample=self.config.remove_on_sample,
            min_source_threshold=self.config.min_source_threshold,
        )

        # Upload configuration and HF credentials
        self.hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        self.hf_org = "gasstation"
        self.hf_dataset_repos = {
            "image": f"{self.hf_org}/gs-images-v3",
            "video": f"{self.hf_org}/gs-videos-v3",
        }
        
        # Validator wallet (for tagging uploads with hotkey, if available)
        try:
            self.validator_wallet = bt.wallet(config=self.config)
        except Exception:
            self.validator_wallet = None

        # Validator UID (looked up from metagraph if wallet is available)
        self.validator_uid = None
        if self.validator_wallet:
            try:
                subtensor = bt.subtensor(config=self.config)
                metagraph = subtensor.metagraph(netuid=self.config.netuid)
                hotkey = self.validator_wallet.hotkey.ss58_address
                hotkeys_list = list(metagraph.hotkeys)
                if hotkey in hotkeys_list:
                    self.validator_uid = hotkeys_list.index(hotkey)
                else:
                    bt.logging.warning(f"[DATA-SERVICE] Validator hotkey {hotkey[:16]}... not found in metagraph")
            except Exception as e:
                bt.logging.warning(f"[DATA-SERVICE] Could not lookup validator UID: {e}")
        
        bt.logging.info(f"[DATA-SERVICE] Validator UID: {self.validator_uid}, Hotkey: {self.validator_wallet.hotkey.ss58_address[:16] if self.validator_wallet else 'None'}...")

        # scraper initialization
        self.scrapers = [
            GoogleScraper(
                headless=True,
                max_year=2017,
                min_width=128,
                min_height=128,
            )
        ]

        # Dataset download vars
        self.all_datasets = load_all_datasets()
        self.scraper_thread: Optional[Thread] = None
        self.downloader_thread: Optional[Thread] = None
        self.uploader_thread: Optional[Thread] = None
        self.uploader_thread_start_time: Optional[float] = None
        self.stop_event = Event()

        # Service state
        self.is_running = False
        self.step = 0

        # block callbacks setup
        self.initialization_complete = True

        self.substrate_manager = SubstrateConnectionManager(
            url=self.config.subtensor.chain_endpoint,
            ss58_format=SS58_FORMAT,
            type_registry=TYPE_REGISTRY
        )
        self.substrate_task = None

        self.block_callbacks = [
            self.log_on_block,
            #self.start_scraper_cycle,
            self.start_dataset_download,
            self.start_upload_cycle,
            self.start_cleanup_cycle,
        ]

    async def start(self):
        """Start the data service with substrate connection."""
        bt.logging.info("[DATA-SERVICE] Starting data service")

        # Enforce hard caps at startup in case max_per_source was lowered
        try:
            pruned = self.content_manager.enforce_source_caps()
            if pruned:
                bt.logging.info(
                    f"[DATA-SERVICE] Enforced source caps on startup: {pruned}"
                )
        except Exception as e:
            bt.logging.warning(f"[DATA-SERVICE] Failed to enforce caps on startup: {e}")

        self.is_running = True
        bt.logging.success("[DATA-SERVICE] Data service started successfully")

        await self.start_dataset_download(0)
        #await self.start_scraper_cycle(0)

        # Start substrate subscription
        self.substrate_task = self.substrate_manager.start_subscription_task(self.run_callbacks)
        bt.logging.info("🚀 Data service substrate subscription started")

        # Main service loop
        while self.is_running:
            self.step += 1

            # Check substrate connection health
            if self.substrate_task is not None and self.substrate_task.done():
                bt.logging.info("Data service substrate connection lost, restarting...")
                try:
                    self.substrate_task = self.substrate_manager.start_subscription_task(self.run_callbacks)
                    bt.logging.info("✅ Data service substrate connection restarted")
                except Exception as e:
                    bt.logging.error(f"Failed to restart data service substrate task: {e}")

            if self.step % 60 == 0:
                self.log_status()

            await asyncio.sleep(1)

    async def stop(self):
        """Stop the data service."""
        bt.logging.info("[DATA-SERVICE] Stopping data service")
        self.is_running = False

        # Clean shutdown of async substrate connection
        if hasattr(self, 'substrate_manager') and self.substrate_manager:
            self.substrate_manager.stop()

        if hasattr(self, 'substrate_task') and self.substrate_task and not self.substrate_task.done():
            self.substrate_task.cancel()
            try:
                await self.substrate_task
            except asyncio.CancelledError:
                pass

        bt.logging.info("✅ Data service shutdown complete")
        self.stop_event.set()

        if self.scraper_thread and self.scraper_thread.is_alive():
            self.scraper_thread.join(timeout=10)

        if self.downloader_thread and self.downloader_thread.is_alive():
            self.downloader_thread.join(timeout=10)

        bt.logging.success("[DATA-SERVICE] Data service stopped")

    async def run_callbacks(self, block):
        if (
            hasattr(self, "initialization_complete")
            and not self.initialization_complete
        ):
            bt.logging.debug(
                f"Skipping callbacks at block {block} during initialization"
            )
            return

        for callback in self.block_callbacks:
            try:
                res = callback(block)
                if inspect.isawaitable(res):
                    await res
            except Exception as e:
                bt.logging.error(
                    f"Failed running callback {callback.__name__}: {str(e)}"
                )
                bt.logging.error(traceback.format_exc())

    @on_block_interval("scraper_interval")
    async def start_scraper_cycle(self, block):
        """Start a scraper cycle at the specified block interval."""
        try:
            bt.logging.info(f"[DATA-SERVICE] Starting scraper cycle at block {block}")

            # Launch scraper worker which will sample queries internally
            if not self.scraper_thread or not self.scraper_thread.is_alive():
                self.scraper_thread = Thread(
                    target=self._scraper_worker,
                    daemon=True,
                )
                self.scraper_thread.start()
                bt.logging.info("[DATA-SERVICE] Scraper thread started")

        except Exception as e:
            bt.logging.error(f"[DATA-SERVICE] Error starting scraper cycle: {e}")
            bt.logging.error(traceback.format_exc())

    def _scraper_worker(self):
        """Worker thread that handles scraping operations."""
        try:
            # Determine which scrapers need data by checking counts against thresholds
            eligible_scrapers = []
            for s in self.scrapers:
                name = s.__class__.__name__
                if self.content_manager.needs_more_data(SourceType.SCRAPER, name):
                    eligible_scrapers.append(name)

            if not eligible_scrapers:
                bt.logging.info("[DATA-SERVICE] No scraper sources currently need data")
                return

            bt.logging.info(f"[DATA-SERVICE] Scraper sources needing data: {eligible_scrapers}")

            total_saved = 0
            for scraper in self.scrapers:
                scraper_name = scraper.__class__.__name__

                # Check if this scraper needs more data
                if scraper_name not in eligible_scrapers:
                    continue
                if not self.content_manager.needs_more_data(SourceType.SCRAPER, scraper_name):
                    continue

                bt.logging.debug(f"[DATA-SERVICE] Processing {scraper_name}")

                try:
                    search_queries = self.content_manager.sample_search_queries(
                        k=self.config.scraper_batch_size,
                        remove=False,
                        strategy="random",
                    )

                    if not search_queries:
                        bt.logging.warning(f"[DATA-SERVICE] No search queries available for {scraper_name}")
                        continue

                    bt.logging.debug(f"[DATA-SERVICE] Scraping for {len(search_queries)} queries")

                    for query_entry in search_queries:
                        if self.stop_event.is_set():
                            break

                        # Check if scraper still needs data
                        if not self.content_manager.needs_more_data(SourceType.SCRAPER, scraper_name):
                            break

                        # Scrape images for this query
                        for query_id, image_data in scraper.download_images(
                            queries=[query_entry.content],
                            query_ids=[query_entry.id],
                            source_image_paths=None,
                            limit=10,
                        ):
                            if self.stop_event.is_set():
                                break

                            # Check again if scraper still needs data
                            if not self.content_manager.needs_more_data(SourceType.SCRAPER, scraper_name):
                                break

                            # Add metadata
                            image_data['query_id'] = query_id
                            image_data['scraper_name'] = scraper_name
                            image_data['modality'] = str(scraper.modality.value).lower()
                            image_data['media_type'] = str(scraper.media_type.value).lower()

                            saved = self._write_media('scraper', image_data, scraper_name=scraper_name)
                            if saved:
                                total_saved += 1

                    bt.logging.debug(f"[DATA-SERVICE] Completed {scraper_name}")

                except Exception as e:
                    bt.logging.error(f"[DATA-SERVICE] Error in {scraper_name}: {e}")
                    bt.logging.error(traceback.format_exc())

            bt.logging.success(f"[DATA-SERVICE] Scraper worker completed. Saved {total_saved} images.")

        except Exception as e:
            bt.logging.error(f"[DATA-SERVICE] Error in scraper worker: {e}")
            bt.logging.error(traceback.format_exc())

    @on_block_interval("dataset_interval")
    async def start_dataset_download(self, block):
        """Start dataset download at the specified block interval."""
        try:
            bt.logging.info(
                f"[DATA-SERVICE] Starting dataset download at block {block}"
            )

            # Start downloader thread if not already running
            if not self.downloader_thread or not self.downloader_thread.is_alive():
                self.downloader_thread = Thread(
                    target=self._downloader_worker, daemon=True
                )
                self.downloader_thread.start()
                bt.logging.info("[DATA-SERVICE] Dataset downloader thread started")

        except Exception as e:
            bt.logging.error(f"[DATA-SERVICE] Error starting dataset download: {e}")
            bt.logging.error(traceback.format_exc())

    def _downloader_worker(self):
        try:
            datasets_to_download = [
                d for d in self.all_datasets
                if self.content_manager.needs_more_data(SourceType.DATASET, d.path)
            ]

            if not datasets_to_download:
                bt.logging.warning("[DATA-SERVICE] No datasets need downloading")
                return

            # Prioritize datasets with zero items first, then by ascending count
            dataset_with_counts = [
                (d, self.content_manager.get_source_count(SourceType.DATASET, d.path))
                for d in datasets_to_download
            ]
            prioritized_datasets = [
                d for d, _ in sorted(dataset_with_counts, key=lambda t: (t[1] != 0, t[1]))
            ]

            names = [d.path for d in prioritized_datasets]
            bt.logging.info(f"[DATA-SERVICE] Starting download for {len(prioritized_datasets)} datasets (priority): {names}")

            total_saved = 0
            for dataset in prioritized_datasets:
                # Check if we still need data for this dataset
                if not self.content_manager.needs_more_data(SourceType.DATASET, dataset.path):
                    continue

                for media_bytes, metadata in download_and_extract(
                    dataset,
                    images_per_parquet=self.config.dataset_images_per_parquet,
                    videos_per_zip=self.config.dataset_videos_per_zip,
                    parquet_per_dataset=self.config.dataset_parquet_per_dataset,
                    zips_per_dataset=self.config.dataset_zips_per_dataset,
                    temp_dir=str(Path(self.config.cache.base_dir) / "tmp"),
                ):
                    if self.stop_event.is_set():
                        break

                    # Check again if we still need data
                    if not self.content_manager.needs_more_data(SourceType.DATASET, dataset.path):
                        break

                    saved = self._write_media('dataset', (media_bytes, metadata))
                    if saved:
                        total_saved += 1

            bt.logging.success(
                f"[DATA-SERVICE] Dataset download completed. Saved {total_saved} media files."
            )

        except Exception as e:
            bt.logging.error(f"[DATA-SERVICE] Error in dataset download worker: {e}")
            bt.logging.error(traceback.format_exc())

    def _write_media(self, source_type: str, data, **kwargs) -> bool:
        """
        Write media using the appropriate ContentManager method based on source type.

        Args:
            source_type: 'scraper' or 'dataset'
            data: For scraped media, this is image_data dict. For dataset media,
                this is (media_bytes, metadata) tuple
            **kwargs: Additional arguments like scraper_name for scraped media

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if source_type == "scraper":
                return self._write_scraped_media(data, kwargs.get("scraper_name"))
            elif source_type == "dataset":
                media_content, metadata = data
                return self._write_dataset_media(media_content, metadata)
            else:
                bt.logging.error(f"[DATA-SERVICE] Unknown source type: {source_type}")
                return False

        except Exception as e:
            bt.logging.error(f"[DATA-SERVICE] Error writing {source_type} media: {e}")
            return False

    def _write_scraped_media(self, image_data: dict, scraper_name: str) -> bool:
        """Write scraped media using ContentManager."""
        try:
            save_path = self.content_manager.write_scraped_media(
                modality=Modality(image_data["modality"].lower()),
                media_type=MediaType(image_data["media_type"].lower()),
                prompt_id=image_data["query_id"],
                media_content=image_data["image_content"],
                download_url=image_data["url"],
                scraper_name=scraper_name,
                resolution=(image_data["width"], image_data["height"]),
            )

            if save_path:
                bt.logging.trace(f"[DATA-SERVICE] Saved scraped media: {save_path}")
                return True
            else:
                bt.logging.warning(
                    f"[DATA-SERVICE] Failed to save scraped media: {image_data['url']}"
                )
                return False

        except Exception as e:
            bt.logging.error(
                f"[DATA-SERVICE] Error writing scraped media {image_data.get('url', 'unknown')}: {e}"
            )
            return False

    def _write_dataset_media(self, media_content, metadata: dict) -> bool:
        """Write dataset media using ContentManager."""
        try:
            dataset_source_file = metadata.get("source_parquet") or metadata.get(
                "source_zip", "unknown"
            )
            dataset_index = metadata.get("original_index") or metadata.get(
                "path_in_zip", "unknown"
            )
            dataset_name = metadata.get("dataset_path", "unknown")

            save_path = self.content_manager.write_dataset_media(
                modality=Modality(metadata.get("modality").lower()),
                media_type=MediaType(metadata.get("media_type")),
                media_content=media_content,
                dataset_name=dataset_name,
                dataset_source_file=dataset_source_file,
                dataset_index=str(dataset_index),
                resolution=(
                    tuple(metadata["image_size"]) if "image_size" in metadata else None
                ),
            )

            if save_path:
                bt.logging.trace(f"[DATA-SERVICE] Saved dataset media: {save_path}")
                return True
            else:
                bt.logging.warning(
                    f"[DATA-SERVICE] Failed to save dataset media from {dataset_name}"
                )
                return False

        except Exception as e:
            bt.logging.error(f"[DATA-SERVICE] Error writing dataset media: {e}")
            return False
    
    @on_block_interval("upload_check_interval")
    async def start_upload_cycle(self, block):
        """Start uploader thread if not already running and threshold met."""
        if self.uploader_thread and self.uploader_thread.is_alive():
            # Treat threads running longer than upload_max_duration_hours as hung
            max_duration = self.config.upload_max_duration_hours * 3600
            elapsed = time.time() - (self.uploader_thread_start_time or 0)
            if elapsed < max_duration:
                return
            bt.logging.warning(
                f"[DATA-SERVICE] Uploader thread has been alive for {elapsed / 3600:.1f}h "
                f"(max {self.config.upload_max_duration_hours:.1f}h) — treating as hung, allowing new upload cycle"
            )
            self.uploader_thread = None

        try:
            total = 0
            imgs = self.content_manager.get_unuploaded_media(
                limit=100000, modality="image"
            )
            vids = self.content_manager.get_unuploaded_media(
                limit=100000, modality="video"
            )
            total = (len(imgs) if imgs else 0) + (len(vids) if vids else 0)
            
            if total < self.config.upload_threshold:
                bt.logging.debug(f"[DATA-SERVICE] Upload threshold not met: {total} < {self.config.upload_threshold}")
                return
            
            batches = min(
                self.config.upload_max_batches,
                total // self.config.upload_threshold
            )
            
            bt.logging.info(
                f"[DATA-SERVICE] Upload threshold met: {total} >= {self.config.upload_threshold}. "
                f"Will upload {batches} batch(es)"
            )
        except Exception as e:
            bt.logging.warning(f"[DATA-SERVICE] Unable to compute upload threshold: {e}",)
            return

        def _worker():
            try:
                if batches <= 0:
                    bt.logging.info("[DATA-SERVICE] Upload skipped (waiting for upload batch minimum)")
                    return
                bt.logging.info(
                    f"[DATA-SERVICE] Uploading {batches} batch(es) of {self.config.upload_batch_size} files per modality"
                )
                self.content_manager.upload_batch_to_huggingface(
                    hf_token=self.hf_token,
                    hf_dataset_repos=self.hf_dataset_repos,
                    upload_batch_size=self.config.upload_batch_size,
                    images_per_archive=self.config.images_per_archive,
                    videos_per_archive=self.config.videos_per_archive,
                    validator_hotkey=(self.validator_wallet.hotkey.ss58_address if self.validator_wallet else None),
                    validator_uid=self.validator_uid,
                    num_batches=batches,
                )
            except Exception as e:
                bt.logging.error(f"[DATA-SERVICE] Uploader error: {e}")
                bt.logging.error(traceback.format_exc())
            finally:
                self.uploader_thread = None
                self.uploader_thread_start_time = None

        self.uploader_thread = Thread(target=_worker, daemon=True)
        self.uploader_thread_start_time = time.time()
        self.uploader_thread.start()

    async def log_on_block(self, block):
        """Log service status on block events."""
        try:
            bt.logging.info(f"[DATA-SERVICE] Block: {block} | Step: {self.step}")
        except Exception as e:
            bt.logging.warning(f"[DATA-SERVICE] Error in log_on_block: {e}")

    @on_block_interval("cleanup_interval")
    async def start_cleanup_cycle(self, block):
        """
        Clean up uploaded+rewarded media to free disk space.
        Runs once per cleanup_interval (default ~24 hours).
        """
        try:
            bt.logging.info(f"[DATA-SERVICE] Starting cleanup cycle at block {block}")
            
            min_age_hours = getattr(self.config, 'cleanup_min_age_hours', 48.0)
            batch_size = getattr(self.config, 'cleanup_batch_size', 1000)
            
            stats = self.content_manager.cleanup_uploaded_media(
                min_age_hours=min_age_hours,
                require_rewarded=True,
                batch_size=batch_size,
            )
            
            if stats['media_deleted'] > 0:
                bt.logging.success(
                    f"[DATA-SERVICE] Cleanup complete: {stats['media_deleted']} media, "
                    f"{stats['prompts_deleted']} prompts, {stats['files_deleted']} files removed"
                )
            else:
                bt.logging.info("[DATA-SERVICE] Cleanup complete: no eligible media to clean")
                
        except Exception as e:
            bt.logging.error(f"[DATA-SERVICE] Error in cleanup cycle: {e}")
            bt.logging.error(traceback.format_exc())

    def log_status(self):
        """Log current service status."""
        try:
            scraper_status = (
                "Running"
                if (self.scraper_thread and self.scraper_thread.is_alive())
                else "Stopped"
            )
            downloader_status = (
                "Running"
                if (self.downloader_thread and self.downloader_thread.is_alive())
                else "Stopped"
            )

            bt.logging.info(
                f"[DATA-SERVICE] Status - Scraper: {scraper_status} | Dataset: {downloader_status}"
            )
        except Exception as e:
            bt.logging.warning(f"[DATA-SERVICE] Error getting status: {e}")


async def main():
    parser = argparse.ArgumentParser(
        description="Combined data service (scraper + dataset)"
    )

    add_args(parser)
    add_data_service_args(parser)
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)

    config = bt.config(parser)

    bt.logging(config=config, logging_dir=config.neuron.full_path)
    bt.logging.set_info()
    if config.logging.debug:
        bt.logging.set_debug(True)
    if config.logging.trace:
        bt.logging.set_trace(True)

    bt.logging.success(config)

    # Create and run service
    service = DataService(config)

    try:
        await service.start()
    except KeyboardInterrupt:
        bt.logging.info("[DATA-SERVICE] Shutting down data service")
        await service.stop()
    except Exception as e:
        bt.logging.error(f"[DATA-SERVICE] Unhandled exception: {e}")
        bt.logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
