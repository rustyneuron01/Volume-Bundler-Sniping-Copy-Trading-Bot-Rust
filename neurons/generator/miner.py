import asyncio
import inspect
import os
import requests
import sys
import time
import traceback
import threading
import uuid
from typing import Optional, Dict, Any
from pathlib import Path

import bittensor as bt
import uvicorn
from fastapi import FastAPI, APIRouter, Request, Response, Depends
from bittensor.core.axon import FastAPIThreadedServer
from bittensor.core.extrinsics.serving import serve_extrinsic
from dotenv import load_dotenv

from gas.protocol.epistula import get_verifier
from gas.types import NeuronType, MinerType
from gas.utils import print_info
from gas.verification import detect_media_format
from neurons.base import BaseNeuron
from neurons.generator.task_manager import TaskManager, GenerationTask, TaskStatus
from gas.protocol.webhooks import send_success_webhook, send_failure_webhook
from neurons.generator.services.service_registry import ServiceRegistry

try:
    load_dotenv(".env.gen_miner")
except Exception:
    pass


class GenerativeMiner(BaseNeuron):
    neuron_type = NeuronType.MINER

    def __init__(self, config=None):
        super().__init__(config=config)
        self.init()

    def init(self):
        max_task_age_hours = getattr(self.config.miner, "max_task_age_hours", 24)
        self.task_manager = TaskManager(max_task_age_hours=max_task_age_hours)
        self.service_registry = ServiceRegistry(self.config)

        self.external_ip = self.config.axon.external_ip or self.config.axon.ip
        if not self.external_ip:
            self.external_ip = requests.get(
                "https://checkip.amazonaws.com", timeout=10
            ).text.strip()

        output_dir = getattr(self.config.miner, "output_dir", "generated_content")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self._start_background_workers()

        self.block_callbacks.append(self.log_on_block)

    def _start_background_workers(self):
        """Start background threads for task processing and cleanup."""
        # Start task processing worker
        self.task_processor = threading.Thread(
            target=self._task_processing_loop, daemon=True
        )
        self.task_processor.start()

        # Start cleanup worker
        self.cleanup_worker = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_worker.start()

    def _task_processing_loop(self):
        """Background loop to process pending tasks."""
        while True:
            try:
                pending_tasks = self.task_manager.get_pending_tasks()

                if not pending_tasks:
                    time.sleep(1)  # No tasks, wait briefly
                    continue

                # Check concurrent task limit
                max_concurrent = getattr(self.config.miner, "max_concurrent_tasks", 5)
                processing_count = sum(
                    1
                    for task in self.task_manager.tasks.values()
                    if task.status.value == "processing"
                )

                if processing_count >= max_concurrent:
                    time.sleep(2)  # At capacity, wait longer
                    continue

                # Process one task
                task = pending_tasks[0]
                try:
                    asyncio.run(self._process_task(task))
                except Exception as e:
                    bt.logging.error(f"Error processing task {task.task_id}: {e}")
                    self.task_manager.mark_task_failed(task.task_id, str(e))
                    send_failure_webhook(
                        task,
                        self.wallet.hotkey,
                        self.external_ip,
                        self.config.axon.port,
                        max_retries=self.config.miner.webhook_max_retries,
                        retry_delay=self.config.miner.webhook_retry_delay,
                        timeout=self.config.miner.webhook_timeout,
                    )

            except Exception as e:
                bt.logging.error(f"Error in task processing loop: {e}")
                time.sleep(5)

    async def _process_task(self, task):
        """Process a single task using the appropriate service."""
        bt.logging.info(f"🎯 Processing task {task.task_id}: modality={task.modality}")

        # Check task timeout
        task_timeout = getattr(self.config.miner, "task_timeout", 300)
        task_age = time.time() - task.created_at
        if task_age > task_timeout:
            raise TimeoutError(f"Task exceeded timeout ({task_timeout}s)")

        self.task_manager.mark_task_processing(task.task_id)

        try:
            # Get the appropriate service for this task
            service = self.service_registry.get_service(task.modality)
            bt.logging.info(f"Using service: {service.__class__.__name__ if service else 'None'} for {task.modality}")
            if not service:
                raise ValueError(f"No service available for modality={task.modality}")

            start_time = time.time()
            result = await self._call_service_process(service, task)
            processing_time = time.time() - start_time

            # mark task complete & send webhook response
            result_data = result.get("data")
            bt.logging.info(
                f"Task {task.task_id} result_data: {len(result_data) if result_data else 0} bytes, "
                f"type={type(result_data)}, magic={result_data[:16].hex() if result_data else 'N/A'}"
            )

            self._save_to_output_dir(task, result_data)

            self.task_manager.mark_task_completed(
                task.task_id, result_data, result.get("url")
            )
            send_success_webhook(
                task,
                result,
                self.wallet.hotkey,
                self.external_ip,
                self.config.axon.port,
                max_retries=self.config.miner.webhook_max_retries,
                retry_delay=self.config.miner.webhook_retry_delay,
                timeout=self.config.miner.webhook_timeout,
            )
            bt.logging.success(
                f"✅ Task {task.task_id} completed in {processing_time:.2f}s"
            )

        except Exception as e:
            self.task_manager.mark_task_failed(task.task_id, str(e))
            send_failure_webhook(
                task,
                self.wallet.hotkey,
                self.external_ip,
                self.config.axon.port,
                max_retries=self.config.miner.webhook_max_retries,
                retry_delay=self.config.miner.webhook_retry_delay,
                timeout=self.config.miner.webhook_timeout,
            )
            bt.logging.error(f"❌ Task {task.task_id} failed: {e}")

    async def _call_service_process(self, service, task):
        """Call service.process(), handling both sync and async services."""
        try:
            bt.logging.debug(f"🔧 Calling service: {service.__class__.__name__}")
            if inspect.iscoroutinefunction(service.process):
                result = await service.process(task)
            else:
                result = service.process(task)

            # Validate service result
            if result is None:
                bt.logging.error(
                    f"Service {service.__class__.__name__} returned None for task {task.task_id}"
                )
                raise ValueError(
                    f"Service {service.__class__.__name__} returned no result"
                )

            if not isinstance(result, dict):
                bt.logging.error(
                    f"Service {service.__class__.__name__} returned non-dict result: {type(result)}"
                )
                raise ValueError(
                    f"Service returned invalid result type: {type(result)}"
                )

            bt.logging.debug(f"📊 Service result keys: {list(result.keys())}")

            # Validate required result structure
            if "data" not in result:
                bt.logging.error(f"Service {service.__class__.__name__} result missing 'data' field")
                raise ValueError("Service result missing required 'data' field")

            result_data = result["data"]
            if result_data is None:
                bt.logging.error(f"Service {service.__class__.__name__} returned None for data field")
                raise ValueError("Service result data field is None")

            bt.logging.info(f"✅ Service {service.__class__.__name__} completed successfully")
            return result

        except Exception as e:
            bt.logging.error(f"Error calling service {service.__class__.__name__}: {e}")
            raise

    def _log_available_services(self):
        bt.logging.info("🔍 Checking service availability for modalities...")

        modalities = ["image", "video"]
        available_count = 0
        total_endpoints = len(modalities)

        bt.logging.info("=" * 60)
        bt.logging.info("📋 GENERATIVE MINER SERVICE STATUS")
        bt.logging.info("=" * 60)

        for modality in modalities:
            service = self.service_registry.get_service(modality)

            endpoint_map = {
                "image": "POST /gen_image",
                "video": "POST /gen_video",
            }
            endpoint = endpoint_map.get(modality, modality)

            if service:
                bt.logging.info(f"✅ {endpoint:<18} → {service.name}")
                available_count += 1
            else:
                bt.logging.info(f"❌ {endpoint:<18} → No service available")

        bt.logging.info("=" * 60)
        bt.logging.info(f"📊 Summary: {available_count}/{total_endpoints} endpoints functional")

        if available_count == 0:
            bt.logging.warning("⚠️  No services available! Miner will reject all requests.")
            bt.logging.info("💡 Check API keys in .env.gen_miner and restart")
        elif available_count < total_endpoints:
            bt.logging.warning(
                f"⚠️  Partial functionality: {total_endpoints - available_count} endpoints unavailable"
            )
        else:
            bt.logging.success("🎉 All endpoints fully functional!")

        bt.logging.info("=" * 60)

    def _cleanup_loop(self):
        """Background loop to cleanup old tasks."""
        cleanup_interval = getattr(self.config.miner, "cleanup_interval", 3600)
        while True:
            try:
                removed = self.task_manager.cleanup_old_tasks()
                if removed > 0:
                    bt.logging.info(f"🧹 Cleaned up {removed} old tasks")
                time.sleep(cleanup_interval)
            except Exception as e:
                bt.logging.error(f"Error in cleanup loop: {e}")
                time.sleep(cleanup_interval)

    def _save_to_output_dir(self, task, data: bytes):
        """Save generated content and metadata to output directory."""
        if not data:
            return

        save_locally = os.getenv("MINER_SAVE_LOCALLY", "false").lower() in ("true", "1", "yes")
        if not save_locally:
            return

        ext = self._detect_extension(data, task.modality)
        media_path = self.output_dir / f"{task.task_id}.{ext}"
        meta_path = self.output_dir / f"{task.task_id}.json"

        try:
            media_path.write_bytes(data)
            bt.logging.info(f"💾 Saved {task.modality} to {media_path}")
        except Exception as e:
            bt.logging.warning(f"Failed to save {task.modality} to {media_path}: {e}")

        try:
            import json
            metadata = {
                "task_id": task.task_id,
                "modality": task.modality,
                "prompt": task.prompt,
                "parameters": task.parameters,
                "signed_by": task.signed_by,
                "created_at": task.created_at,
                "completed_at": time.time(),
            }
            meta_path.write_text(json.dumps(metadata, indent=2))
        except Exception as e:
            bt.logging.warning(f"Failed to save metadata to {meta_path}: {e}")

    def _detect_extension(self, data: bytes, modality: str) -> str:
        """Detect file extension from binary data magic bytes."""
        ext = detect_media_format(data)
        if ext != ".bin":
            return ext.lstrip(".")
        return "png" if modality == "image" else "mp4"

    # API Endpoints
    async def _handle_generation_request(self, request: Request, modality: str):
        """Handle generation requests for both image and video."""
        try:
            bt.logging.info(f"Received {modality} generation request")

            # Parse request body
            try:
                body = await request.json()
                bt.logging.debug(f"Request body keys: {list(body.keys()) if body else 'None'}")
            except Exception as e:
                bt.logging.error(f"Failed to parse JSON body: {e}")
                return self._error_response(
                    f"Invalid JSON body: {str(e)}", 400
                )

            # Extract required fields
            prompt = body.get("prompt")
            if not prompt:
                bt.logging.error(f"Missing prompt field. Body: {body}")
                return self._error_response("Missing required field: prompt", 400)

            webhook_url = request.headers.get("X-Webhook-URL")
            if not webhook_url:
                bt.logging.error(f"Missing X-Webhook-URL header. Headers: {dict(request.headers)}")
                return self._error_response(
                    "Missing required header: X-Webhook-URL", 400
                )

            signed_by = request.headers.get("Epistula-Signed-By", "unknown")
            parameters = body.get("parameters", {}) or {}

            service = self.service_registry.get_service(modality)
            if not service:
                bt.logging.warning(f"Rejecting {modality} request: no service available")
                return self._error_response(
                    f"No service available for modality={modality}", 503
                )

            task_id = self.task_manager.create_task(
                modality=modality,
                prompt=prompt,
                parameters=parameters,
                webhook_url=webhook_url,
                signed_by=signed_by,
                input_data=None,  # placeholder for modifications
            )

            bt.logging.info(f"✅ Created {modality} task {task_id}: {prompt[:50]}...")

            created_task = self.task_manager.get_task(task_id)

            return self._success_response(
                {
                    "task_id": task_id,
                    "status": "submitted",
                    "modality": modality,
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "webhook_url": webhook_url,
                    "created_at": (
                        created_task.created_at if created_task else time.time()
                    ),
                },
                201,
            )

        except Exception as e:
            bt.logging.error(f"Error handling {modality} request: {e}")
            return self._error_response(
                f"Failed to process {modality} request: {str(e)}", 500
            )

    async def generate_image(self, request: Request):
        """Submit image generation task."""
        return await self._handle_generation_request(request, "image")

    async def generate_video(self, request: Request):
        """Submit video generation task."""
        return await self._handle_generation_request(request, "video")

    async def get_task_status(self, request: Request):
        """Get status of a specific task."""
        try:
            path_parts = request.url.path.split("/")
            if len(path_parts) < 3:
                return self._error_response("Invalid status URL", 400)

            task_id = path_parts[2]
            task = self.task_manager.get_task(task_id)

            if not task:
                return self._error_response("Task not found", 404)

            response_data = task.to_dict()
            return self._success_response(response_data)

        except Exception as e:
            bt.logging.error(f"Error getting task status: {e}")
            return self._error_response(str(e), 500)

    async def get_miner_info(self, request: Request):
        """Return comprehensive miner information."""
        services = self.service_registry.get_available_services()
        task_stats = self.task_manager.get_task_stats()

        return self._success_response(
            {
                "uid": self.uid,
                "hotkey": self.wallet.hotkey.ss58_address,
                "miner_type": MinerType.GENERATOR.value,
                "services": services,
                "task_stats": task_stats,
                "config": {
                    "max_concurrent_tasks": getattr(
                        self.config.miner, "max_concurrent_tasks", 5
                    ),
                    "task_timeout": getattr(self.config.miner, "task_timeout", 300),
                    "max_task_age_hours": getattr(
                        self.config.miner, "max_task_age_hours", 24
                    ),
                    "cleanup_interval": getattr(
                        self.config.miner, "cleanup_interval", 3600
                    ),
                },
                "version": "1.0.0",
                "uptime": time.time() - getattr(self, "_start_time", time.time()),
            }
        )

    async def get_health_check(self, request: Request):
        """Health check endpoint."""
        try:
            services = self.service_registry.get_available_services()
            available_count = sum(1 for s in services if s["available"])

            if available_count == 0:
                return self._error_response("No services available", 503)

            return self._success_response(
                {
                    "status": "healthy",
                    "services_available": available_count,
                    "services_total": len(services),
                    "timestamp": time.time(),
                }
            )
        except Exception as e:
            return self._error_response(f"Health check failed: {e}", 503)

    def _success_response(self, data: Dict[str, Any], status_code: int = 200):
        """Helper to create success responses."""
        import json

        return Response(
            content=json.dumps(data),
            media_type="application/json",
            status_code=status_code,
        )

    def _error_response(self, message: str, status_code: int = 400):
        """Helper to create error responses."""
        import json

        return Response(
            content=json.dumps({"error": message, "timestamp": time.time()}),
            media_type="application/json",
            status_code=status_code,
        )

    async def log_on_block(self, block):
        """Log information on each block."""
        print_info(
            self.metagraph,
            self.wallet.hotkey.ss58_address,
            block,
        )

    def shutdown(self):
        """Shutdown the miner."""
        bt.logging.info("🛑 Shutting down GenerativeMiner...")
        if hasattr(self, "fast_api"):
            self.fast_api.stop()

    def run(self):
        """Main run method - sets up FastAPI server and starts the miner."""
        self._start_time = time.time()

        success = self._serve_on_network()
        if not success:
            bt.logging.error("❌ Failed to serve miner on network")
            return

        app = self._create_fastapi_app()
        fast_config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=self.config.axon.port,
            log_level="info",
            loop="asyncio",
        )
        self.fast_api = FastAPIThreadedServer(config=fast_config)
        self.fast_api.start()

        bt.logging.success(f"Generative miner started (UID: {self.uid})")
        self._log_available_services()

        # Main loop
        try:
            while not self.exit_context.isExiting:
                time.sleep(1)
        except Exception as e:
            bt.logging.error(f"Error in main loop: {e}")
            bt.logging.error(traceback.format_exc())
        finally:
            self.shutdown()

    def _serve_on_network(self) -> bool:
        """Register the miner on the Bittensor network."""
        try:
            bt.logging.info(
                f"🌐 Serving axon on {self.external_ip}:{self.config.axon.port}"
            )
            bt.logging.info(f"🔗 Network: {self.config.subtensor.chain_endpoint}")
            bt.logging.info(f"🆔 Netuid: {self.config.netuid}")
            return serve_extrinsic(
                subtensor=self.subtensor,
                wallet=self.wallet,
                ip=self.external_ip,
                port=self.config.axon.port,
                protocol=4,
                netuid=self.config.netuid,
                wait_for_finalization=True,
            )
        except Exception as e:
            bt.logging.error(f"Error serving on network: {e}")
            return False

    def _create_fastapi_app(self) -> FastAPI:
        """Create and configure the FastAPI application."""
        app = FastAPI(
            title="GenerativeMiner",
            description="Production-ready generative miner for Bitmind subnet",
            version="1.0.0",
        )
        router = APIRouter()
        verifier = get_verifier(
            self.wallet,
            self.metagraph,
            getattr(self.config, "no_force_validator_permit", False),
        )
        router.add_api_route(
            "/health",
            self.get_health_check,
            dependencies=[Depends(verifier)],
            methods=["GET"],
        )
        router.add_api_route(
            "/miner_info",
            self.get_miner_info,
            dependencies=[Depends(verifier)],
            methods=["GET"],
        )
        router.add_api_route(
            "/status/{task_id}",
            self.get_task_status,
            dependencies=[Depends(verifier)],
            methods=["GET"],
        )

        # endpoints called by validators
        router.add_api_route(
            "/gen_image",
            self.generate_image,
            dependencies=[Depends(verifier)],
            methods=["POST"],
        )
        router.add_api_route(
            "/gen_video",
            self.generate_video,
            dependencies=[Depends(verifier)],
            methods=["POST"],
        )

        app.include_router(router)
        return app


if __name__ == "__main__":
    try:
        miner = GenerativeMiner()
        miner.run()
    except KeyboardInterrupt:
        bt.logging.info("Miner interrupted by KeyboardInterrupt, shutting down")
    except Exception as e:
        bt.logging.error(f"Unhandled exception: {e}")
        bt.logging.error(traceback.format_exc())
