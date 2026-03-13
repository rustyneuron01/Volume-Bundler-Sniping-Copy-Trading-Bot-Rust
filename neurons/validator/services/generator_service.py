import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from threading import Thread, Event, Lock
from typing import Optional
from collections import deque
import random
import json

import bittensor as bt
import torch
import gc
import numpy as np
from PIL import Image

from gas.config import add_args, add_generation_service_args
from gas.generation.tps import nano_banana
from gas.generation import (
    GenerationPipeline,
    initialize_model_registry,
    PromptGenerator,
)
from gas.cache.content_manager import ContentManager
from gas.types import Modality, MediaType


class GeneratorService:
    """
    Enhanced LocalGenerator that can run as a standalone service.
    Combines core generation functionality with service management.
    """

    def __init__(self, config):
        self.config = config

        if hasattr(self.config, "cache") and hasattr(self.config.cache, "base_dir"):
            self.config.cache.base_dir = str(
                Path(self.config.cache.base_dir).expanduser()
            )

        self._output_dir = Path(self.config.cache.base_dir)

        self.validator_wallet = bt.wallet(config=self.config)

        self._service_running = False
        self._generation_running = False
        self._stop_requested = Event()
        self._lock = Lock()

        self._generation_thread: Optional[Thread] = None
        self.content_manager = ContentManager(
            base_dir=self.config.cache.base_dir,
            max_per_source=self.config.max_per_source,
            enable_source_limits=self.config.enable_source_limits,
            prune_strategy=self.config.prune_strategy,
            remove_on_sample=self.config.remove_on_sample,
            min_source_threshold=self.config.min_source_threshold,
        )
        self.generation_pipeline: Optional[GenerationPipeline] = None
        self.prompt_generator: Optional[PromptGenerator] = None
        self.model_registry = None
        self.model_names = []

        self.hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if self.hf_token:
            bt.logging.info(
                f"[GENERATOR-SERVICE] HuggingFace token loaded: {self.hf_token[:10]}..."
            )
        else:
            bt.logging.warning(
                "[GENERATOR-SERVICE] No HuggingFace token found in environment"
            )

        self.hf_org = "gasstation"
        self.hf_dataset_repos = {
            "image": f"{self.hf_org}/gs-image-v2",
            "video": f"{self.hf_org}/gs-video-v2",
        }

        self.tp_generators = {"nano_banana": nano_banana.generate_image}

        self._first_run_profiled = False
        self._job_profiles = {}  
        # job_name -> {"peak_vram_gb": float, "avg_duration_s": float, "count": int}
        # Per-model profiling: model_name -> {"max_peak_vram_gb": float, "avg_gen_s": float, "count": int, "last_gen_s": float}
        self._model_profiles = {}
        self._headroom_gb = 3.0
        self.gen_batch_size = getattr(self.config, 'local_batch_size', 1)
        try:
            self._load_profiles_from_cache()
        except Exception as e:
            bt.logging.debug(f"[GENERATOR-SERVICE] Could not load cached profiles: {e}")

        self._verification_max_batch = 512

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, new_dir):
        """Update the output directory. Only works when not running."""
        if self._generation_running:
            raise RuntimeError(
                "Cannot change output directory while generator is running"
            )
        self._output_dir = Path(new_dir)
        bt.logging.info(
            f"[GENERATOR-SERVICE] Generator output directory updated to: {self._output_dir}"
        )

    def start(self):
        """Start the generator service."""
        bt.logging.info("[GENERATOR-SERVICE] Starting generator service")

        try:
            self._service_running = True
            self._service_loop()
            return True

        except Exception as e:
            bt.logging.error(
                f"[GENERATOR-SERVICE] Failed to start generator service: {e}"
            )
            bt.logging.error(traceback.format_exc())
            return False

    def stop(self):
        """Stop the generator service."""
        bt.logging.info("[GENERATOR-SERVICE] Stopping generator service")
        self._service_running = False
        self._stop_generation()

    def _stop_generation(self):
        """Stop the generation process."""
        with self._lock:
            if not self._generation_running:
                bt.logging.warning(
                    "[GENERATOR-SERVICE] Generation process is not running"
                )
                return False

            self._stop_requested.set()
            if self._generation_thread:
                self._generation_thread.join(timeout=10)
            self._generation_running = False
            bt.logging.info("[GENERATOR-SERVICE] Generation process stopped")
            return True

    def _start_generation(self):
        """Start the generation process in a separate thread."""
        with self._lock:
            if self._generation_running:
                bt.logging.warning(
                    "[GENERATOR-SERVICE] Generation process is already running"
                )
                return False

            self._stop_requested.clear()
            self._generation_thread = Thread(
                target=self._generation_worker, daemon=True
            )
            self._generation_thread.start()
            self._generation_running = True
            return True

    def _service_loop(self):
        """Main service loop that manages the generation process lifecycle."""
        bt.logging.info("[GENERATOR-SERVICE] Service loop started")

        while self._service_running:
            try:
                if not self._start_generation():
                    bt.logging.error(
                        "[GENERATOR-SERVICE] Failed to start generation process"
                    )
                    time.sleep(60)
                    continue

                while self._generation_running and self._service_running:
                    time.sleep(30)
                    bt.logging.info("[GENERATOR-SERVICE] Service heartbeat")

                # If service is still running but generation stopped, restart it
                if self._service_running:
                    bt.logging.info(
                        "[GENERATOR-SERVICE] Generation process stopped, restarting..."
                    )
                    time.sleep(10)

            except KeyboardInterrupt:
                bt.logging.info("[GENERATOR-SERVICE] Received interrupt signal")
                break
            except Exception as e:
                bt.logging.error(f"[GENERATOR-SERVICE] Error in service loop: {e}")
                bt.logging.error(traceback.format_exc())
                time.sleep(60)  # Wait before retry

        bt.logging.info("[GENERATOR-SERVICE] Service loop stopped")

    def _generation_worker(self):
        """Worker thread that performs the actual generation work."""
        try:
            self._initialize_pipelines()
            self._generation_work_loop()
        except Exception as e:
            bt.logging.error(
                f"[GENERATOR-SERVICE] Fatal error in generation worker: {e}"
            )
            bt.logging.error(traceback.format_exc())
        finally:
            self._cleanup()

    def _initialize_pipelines(self):
        """Initialize the generation models and components."""
        bt.logging.info("[GENERATOR-SERVICE] Initializing models...")

        self.model_registry = initialize_model_registry()
        self.generation_pipeline = GenerationPipeline(
            model_registry=self.model_registry,
        )
        self.prompt_generator = PromptGenerator()

        self.model_names = self.model_registry.get_interleaved_model_names()
        bt.logging.info(
            f"[GENERATOR-SERVICE] Initialized with models: {self.model_names}"
        )

    def _generation_work_loop(self):
        """Main generation loop."""
        bt.logging.info("[GENERATOR-SERVICE] Starting generation work loop")

        while not self._stop_requested.is_set():
            try:
                self._await_media(min_needed=1, max_wait_s=1200, poll_s=10)
                self._run_job(
                    {
                        "kind": "prompts",
                        "args": {"batch_size": self.config.prompt_batch_size}
                    }
                )

                if self._first_run_profiled is False:
                    self._first_run_profiled = True
                    try:
                        self._save_profiles_to_cache()
                    except Exception:
                        pass

                if self._stop_requested.is_set():
                    break

                self._run_job({"kind": "gen_tps", "args": {"k": self.config.tps_batch_size}})
                self._run_job({"kind": "gen_local", "args": {"k": self.gen_batch_size}})
                self._run_verification()

            except Exception as e:
                bt.logging.error(f"[GENERATOR-SERVICE] Error in generation work loop: {e}")
                bt.logging.error(traceback.format_exc())
                time.sleep(3)

    def _run_job(self, job):
        kind = job.get("kind")
        args = job.get("args", {}) or {}

        torch.cuda.reset_peak_memory_stats()
        vram_before_free, _ = torch.cuda.mem_get_info()
        start = time.time()

        if kind == "prompts":
            count, ready = self._has_min_source_media(min_needed=1)
            if not ready:
                bt.logging.info(f"[GENERATOR-SERVICE] Skipping prompt generation; waiting for base media ({count}/1)")
                return
            start = time.time()
            self._generate_text("prompt", args.get("batch_size", 1))
            bt.logging.info(f"[GENERATOR-SERVICE] Generated prompts in {time.time()-start:.2f} seconds")
        elif kind == "gen_tps":
            self._generate_media(use_local=False, k=args.get("k", 1))
        elif kind == "gen_local":
            self._generate_media(use_local=True, k=args.get("k", self.gen_batch_size))
        else:
            bt.logging.warning(f"[GENERATOR-SERVICE] Unknown job kind: {kind}")

        duration = time.time() - start
        peak_alloc_bytes = torch.cuda.max_memory_allocated()
        peak_gb = float(peak_alloc_bytes) / (1024**3)
        self._update_profile(kind, peak_gb, duration)
        vram_after_free, _ = torch.cuda.mem_get_info()

        vram_before_gb = float(vram_before_free) / (1024**3)
        vram_after_gb = float(vram_after_free) / (1024**3)
        profiling_results = {
            "job_kind": kind,
            "duration_s": round(duration, 4),
            "peak_vram_gb": round(peak_gb, 4),
            "vram_before_job_gb": round(vram_before_gb, 4),
            "vram_after_job_gb": round(vram_after_gb, 4),
            "vram_diff_gb": round(vram_before_gb - vram_after_gb, 4), # Memory held by the job
        }
        bt.logging.debug(f"[GENERATOR-SERVICE] Job profiling: {json.dumps(profiling_results)}")

    def _update_profile(self, kind: str, peak_gb: float, duration_s: float):
        rec = self._job_profiles.get(kind, {"peak_vram_gb": 0.0, "avg_duration_s": 0.0, "count": 0})
        cnt = rec["count"] + 1
        rec["peak_vram_gb"] = max(rec["peak_vram_gb"], peak_gb)
        # ema
        if rec["count"] == 0:
            rec["avg_duration_s"] = duration_s
        else:
            rec["avg_duration_s"] = (rec["avg_duration_s"] * rec["count"] + duration_s) / cnt
        rec["count"] = cnt
        self._job_profiles[kind] = rec
        try:
            self._save_profiles_to_cache()
        except Exception as e:
            bt.logging.debug(f"[GENERATOR-SERVICE] Failed to persist profiles: {e}")

    def _update_model_metrics(self, model_name: str, peak_gb: float, gen_duration_s: Optional[float]):
        rec = self._model_profiles.get(
            model_name,
            {"max_peak_vram_gb": 0.0, "avg_gen_s": 0.0, "count": 0, "last_gen_s": 0.0},
        )
        # normalize keys
        max_peak = rec.get("max_peak_vram_gb", 0.0)
        avg_gen = rec.get("avg_gen_s", 0.0)
        cnt = rec.get("count", 0)
        if peak_gb is None:
            peak_gb = 0.0
        if gen_duration_s is None:
            gen_duration_s = 0.0
        max_peak = max(max_peak, float(peak_gb))
        if cnt == 0:
            avg_gen = float(gen_duration_s)
        else:
            avg = (avg_gen * cnt + float(gen_duration_s)) / (cnt + 1)
            avg_gen = avg
        rec = {
            "max_peak_vram_gb": max(max_peak, 0.0),
            "avg_gen_s": avg_gen,
            "count": cnt + 1,
            "last_gen_s": float(gen_duration_s),
        }
        self._model_profiles[model_name] = rec
        try:
            self._save_profiles_to_cache()
        except Exception:
            pass

    def _get_vram_gb(self):
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            return free_b / (1024**3), total_b / (1024**3)
        return float("inf"), float("inf")

    def _get_expected_runtime_s(self, kind: str) -> float:
        rec = self._job_profiles.get(kind)
        if rec and rec.get("avg_duration_s"):
            return max(60.0, float(rec["avg_duration_s"]))
        # fallback heuristics
        if kind == "gen_local":
            return 40.0 * 60.0  # 40 minutes default
        if kind == "verify":
            return 2.0 * 60.0
        return 60.0

    def _profiles_cache_path(self) -> Path:
        profiles_dir = Path(self.config.cache.base_dir).expanduser() / "profiles"
        profiles_dir.mkdir(parents=True, exist_ok=True)
        return profiles_dir / "generator_profiles.json"

    def _load_profiles_from_cache(self):
        path = self._profiles_cache_path()
        if not path.exists():
            return
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            cached = data.get("job_profiles")
            if isinstance(cached, dict):
                self._job_profiles = cached
            m = data.get("model_profiles")
            if isinstance(m, dict):
                self._model_profiles = m
            sched = data.get("scheduling")
            if isinstance(sched, dict):
                _ = sched  # preserved for potential future use

    def _save_profiles_to_cache(self):
        path = self._profiles_cache_path()
        exp_heavy = self._get_expected_runtime_s("gen_local")
        window = max(2.0 * exp_heavy, exp_heavy + 60.0)
        doc = {
            "job_profiles": self._job_profiles,
            "model_profiles": self._model_profiles,
            "scheduling": {
                "budget_seconds": exp_heavy,
                "window_seconds": window,
            },
            "version": 1,
        }
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(doc, f)
        tmp.replace(path)

    def _has_min_source_media(self, min_needed: int) -> tuple[int, bool]:
        try:
            total = 0
            min_needed = max(1, int(min_needed))
            for mt in (MediaType.REAL, MediaType.SEMISYNTHETIC, MediaType.SYNTHETIC):
                cache_result = self.content_manager.sample_media_with_content(
                    modality=Modality.IMAGE,
                    media_type=mt,
                    count=min_needed,
                    remove_from_cache=False,
                )
                total += cache_result["count"] if cache_result and "count" in cache_result else 0
                if total >= min_needed:
                    break
            return total, total >= min_needed
        except Exception:
            return 0, False

    def _await_media(self, min_needed: int, max_wait_s: int = 600, poll_s: int = 10) -> bool:
        start = time.time()
        while not self._stop_requested.is_set():
            count, ready = self._has_min_source_media(min_needed=max(1, int(min_needed)))
            if ready:
                return True
            if time.time() - start >= max_wait_s:
                bt.logging.info(f"[GENERATOR-SERVICE] Media readiness wait timed out ({count}/{min_needed}); continuing without prompts")
                return False
            bt.logging.info(f"[GENERATOR-SERVICE] Waiting for base media ({count}/{min_needed})...")
            time.sleep(poll_s)

    def _sample_images(self, count: int = 1):
        """Sample image media with content from any media type, randomly across types."""
        media_types = [MediaType.REAL, MediaType.SEMISYNTHETIC, MediaType.SYNTHETIC]
        random.shuffle(media_types)
        for mt in media_types:
            try:
                result = self.content_manager.sample_media_with_content(
                    modality=Modality.IMAGE,
                    media_type=mt,
                    count=count,
                )
                if result and result.get("count", 0) > 0:
                    return result
            except Exception as e:
                bt.logging.debug(f"[GENERATOR-SERVICE] sample_media_with_content failed for {mt}: {e}")
                continue
        return {"count": 0, "items": []}

    def _generate_text(
        self, content_type: str = "search_query", batch_size: int = 10
    ) -> list:
        """
        Generate a batch of content (search queries or prompts) and save to database.

        Args:
            content_type: Either "search_query" or "prompt"
            batch_size: Number of items to generate

        Returns:
            List of generated content items
        """
        generated = 0
        for _ in range(batch_size):
            if self._stop_requested.is_set():
                break

            try:
                if content_type == "search_query":
                    search_query = self.prompt_generator.generate_search_query()
                    self.content_manager.write_prompt(
                        content=search_query,
                        content_type=content_type,
                        source_media_id=None,
                    )
                    generated += 1

                elif content_type == "prompt":
                    cache_result = self._sample_images(count=1)

                    if not cache_result or not cache_result["count"]:
                        bt.logging.warning("No images available for prompt generation")
                        continue

                    item = cache_result["items"][0]
                    for modality in Modality:
                        prompt = self.prompt_generator.generate_prompt_from_image(
                            Image.fromarray(item["image"]),
                            intended_modality=modality
                        )

                        self.content_manager.write_prompt(
                            content=prompt,
                            content_type=content_type,
                            source_media_id=item["id"],
                            modality=modality.value,  # Store the intended modality
                        )
                        generated += 1

                else:
                    raise ValueError(f"Unknown content_type: {content_type}")

            except Exception as e:
                bt.logging.error(f"[GENERATOR-SERVICE] Error generating {content_type}: {e}")
                continue

            bt.logging.info(f"[GENERATOR-SERVICE] Added {generated} {content_type}s to database")
                
        self.prompt_generator.clear_gpu()
        return generated

    def _generate_media(self, use_local=True, k=1):
        start = time.time()
        entries = self.content_manager.sample_prompts_with_source_media(
            k=k,
            remove=False,
            strategy="least_used",
        )

        if entries:
            for entry in entries:
                prompt = entry["prompt"]
                original_media = entries[0]["media"]
                bt.logging.debug(f"[GENERATOR-SERVICE] Generating media")
                bt.logging.debug(f"- Prompt: {prompt}")
                bt.logging.debug(f"- Models: {self.model_names}")

                if use_local:
                    # Send prompt to all local models
                    try:
                        torch.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass

                    for gen_output in self.generation_pipeline.generate_media(
                        prompt=prompt.content,
                        model_names=self.model_names,
                        image_sample=original_media,
                    ):
                        if self._stop_requested.is_set():
                            break

                        if gen_output:
                            # profiling
                            peak_gb = 0.0
                            try:
                                peak_bytes = torch.cuda.max_memory_allocated()
                                peak_gb = float(peak_bytes) / (1024**3)
                            except Exception:
                                peak_gb = 0.0
                            model_name = gen_output.get("model_name")
                            gen_dur = gen_output.get("gen_duration")
                            if model_name:
                                self._update_model_metrics(model_name, peak_gb, gen_dur)

                            torch.cuda.reset_peak_memory_stats()

                            # save gen outputs
                            save_path = self._write_media(gen_output, prompt.id)
                            if save_path:
                                bt.logging.info(
                                    f"[GENERATOR-SERVICE] Generated and saved media file: {save_path} from prompt '{prompt.id}'"
                                )

                        self._run_verification()
                else:
                    # send prompt to third party services
                    for service_name, generator_fn in self.tp_generators.items():
                        bt.logging.info(
                            f"[GENERATOR-SERVICE] Generating with third party service: {service_name}'"
                        )
                        try:
                            gen_output = generator_fn(prompt.content)
                        except RuntimeError as e:
                            bt.logging.warning(e)
                            continue

                        if gen_output:
                            save_path = self._write_media(gen_output, prompt.id)
                            time.sleep(10)
                            if save_path:
                                bt.logging.info(
                                    f"[GENERATOR-SERVICE] Generated and saved media file: {save_path} from prompt '{prompt.id}'"
                                )

                bt.logging.info(
                    f"[GENERATOR-SERVICE] Completed media generation for prompt '{prompt.id}' in {time.time()-start:.2f} seconds"
                )
        else:
            bt.logging.warning(
                "[GENERATOR-SERVICE] No prompts available for media generation"
            )

    def _write_media(self, media_sample, prompt_id: str):
        """
        Args:
            media_sample: Dictionary containing the generated media (image/video)
            prompt_id: ID of the prompt used for generation
        """
        try:
            modality = media_sample["modality"]
            media_type = media_sample["media_type"]
            model_name = media_sample["model_name"]

            # Handle mask if present
            mask_content = None
            if "mask_image" in media_sample:
                mask_content = np.array(media_sample["mask_image"])

            save_path = self.content_manager.write_generated_media(
                modality=modality,
                media_type=media_type,
                model_name=model_name,
                prompt_id=prompt_id,
                media_content=media_sample[modality],
                mask_content=mask_content,
                generation_args=media_sample.get("generation_args"),
            )

            return save_path

        except Exception as e:
            bt.logging.error(
                f"[GENERATOR-SERVICE] Error writing media with ContentManager: {e}"
            )
            bt.logging.error(traceback.format_exc())
            return None

    def _run_verification(self):
        """Run verification after GPU has been cleared."""
        pending = self.content_manager.get_pending_verification_count()
        if pending == 0:
            return

        try:
            from gas.verification import (
                run_verification,
                get_verification_summary,
                clear_clip_models,
            )

            bt.logging.info(f"[GENERATOR-SERVICE] Starting verification for {pending} pending media")
            start_time = time.time()

            results = run_verification(
                content_manager=self.content_manager,
                batch_size=self._verification_max_batch,
                threshold=0.25,
                clip_batch_size=512,
            )

            if results:
                summary = get_verification_summary(results)
                bt.logging.info(
                    f"[GENERATOR-SERVICE] Verification complete in {time.time()-start_time:.1f}s: "
                    f"{summary['successful']}/{summary['total']} processed, "
                    f"{summary['passed']} passed, {summary['failed']} failed, {summary['errors']} errors "
                    f"(pass rate: {summary['pass_rate']:.1%}, avg score: {summary['average_score']:.3f})"
                )

            clear_clip_models()

        except Exception as e:
            bt.logging.error(f"[GENERATOR-SERVICE] Verification error: {e}")
            bt.logging.error(traceback.format_exc())
            try:
                from gas.verification import clear_clip_models
                clear_clip_models()
            except:
                pass

    def _cleanup(self):
        """Clean up resources."""
        try:
            if self.generation_pipeline:
                self.generation_pipeline.shutdown()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            bt.logging.error(f"[GENERATOR-SERVICE] Error during cleanup: {e}")


def main():
    parser = argparse.ArgumentParser(description="Image+video generation service")

    add_args(parser)
    add_generation_service_args(parser)
    bt.subtensor.add_args(parser)
    bt.wallet.add_args(parser)
    bt.logging.add_args(parser)
    config = bt.config(parser)

    bt.logging(config=config, logging_dir=config.neuron.full_path)
    bt.logging.set_info()
    if config.logging.debug:
        bt.logging.set_debug(True)
    if config.logging.trace:
        bt.logging.set_trace(True)

    bt.logging.success(config)
    service = GeneratorService(config)

    try:
        service.start()
    except KeyboardInterrupt:
        bt.logging.info("[GENERATOR-SERVICE] Shutting down generator service")
        service.stop()
    except Exception as e:
        bt.logging.error(f"[GENERATOR-SERVICE] Unhandled exception: {e}")
        bt.logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
