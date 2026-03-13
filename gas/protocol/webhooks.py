import json
import os
import shutil
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, TYPE_CHECKING
from urllib.parse import urlparse

import requests
import bittensor as bt

from gas.protocol.epistula import generate_header

if TYPE_CHECKING:
    from neurons.generator.task_manager import GenerationTask

from gas.verification import detect_media_format

_EXT_TO_MIME = {
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
}


def _detect_content_type(data: bytes, modality: str) -> str:
    """Detect content type from binary data magic bytes."""
    ext = detect_media_format(data)
    if ext in _EXT_TO_MIME:
        return _EXT_TO_MIME[ext]
    # Fallback based on modality
    if modality == "image":
        return "image/png"
    elif modality == "video":
        return "video/mp4"
    return "application/octet-stream"


@dataclass
class ValidatorStats:
    """Stats for a single validator IP."""
    successes: int = 0
    failures: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_success: Optional[str] = None
    last_failure: Optional[str] = None
    last_failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "successes": self.successes,
            "failures": dict(self.failures),
            "total_failures": sum(self.failures.values()),
            "last_success": self.last_success,
            "last_failure": self.last_failure,
            "last_failure_reason": self.last_failure_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidatorStats":
        stats = cls()
        stats.successes = data.get("successes", 0)
        stats.failures = defaultdict(int, data.get("failures", {}))
        stats.last_success = data.get("last_success")
        stats.last_failure = data.get("last_failure")
        stats.last_failure_reason = data.get("last_failure_reason")
        return stats


class WebhookStatsTracker:
    """Tracks webhook success/failure stats per validator IP."""

    # Failure categories
    FAILURE_EMPTY_PAYLOAD = "empty_payload"
    FAILURE_CONNECTION_TIMEOUT = "connection_timeout"
    FAILURE_CONNECTION_ERROR = "connection_error"
    FAILURE_HTTP_400 = "http_400"
    FAILURE_HTTP_401 = "http_401"
    FAILURE_HTTP_404 = "http_404"
    FAILURE_HTTP_500 = "http_500"
    FAILURE_HTTP_OTHER = "http_other"
    FAILURE_UNKNOWN = "unknown"

    # Configuration
    SAVE_INTERVAL_SECONDS = 60  # Save to disk at most every 60 seconds
    SUMMARY_INTERVAL_SECONDS = 300  # Print summary every 5 minutes (0 to disable)

    _instance: Optional["WebhookStatsTracker"] = None
    _lock = threading.Lock()

    def __new__(cls, stats_file: Optional[str] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, stats_file: Optional[str] = None):
        if self._initialized:
            return

        self._stats_file = stats_file or os.path.expanduser("~/.bitmind/webhook_stats.json")
        self._stats: Dict[str, ValidatorStats] = {}
        self._file_lock = threading.Lock()
        self._dirty = False  # Track if stats have changed since last save
        self._last_save_time = 0.0
        self._last_summary_time = 0.0
        self._last_rotation_date: Optional[str] = None
        self._changes_since_save = 0
        self._initialized = True
        self._load_stats()

        # Start background save thread
        self._save_thread = threading.Thread(target=self._periodic_save_loop, daemon=True)
        self._save_thread.start()

    def _load_stats(self):
        """Load stats from file if it exists."""
        try:
            if os.path.exists(self._stats_file):
                with open(self._stats_file, "r") as f:
                    data = json.load(f)
                    for ip, stats_data in data.get("validators", {}).items():
                        self._stats[ip] = ValidatorStats.from_dict(stats_data)
                bt.logging.debug(f"Loaded webhook stats for {len(self._stats)} validators")
        except Exception as e:
            bt.logging.warning(f"Failed to load webhook stats: {e}")
            self._stats = {}

    def _periodic_save_loop(self):
        """Background thread that periodically saves stats and prints summary."""
        while True:
            time.sleep(10)  # Check every 10 seconds

            now = time.time()

            # Check for daily rotation
            self._check_rotation()

            # Save if dirty and enough time has passed
            if self._dirty and (now - self._last_save_time) >= self.SAVE_INTERVAL_SECONDS:
                self._do_save()

            # Print summary periodically if enabled
            if self.SUMMARY_INTERVAL_SECONDS > 0:
                if (now - self._last_summary_time) >= self.SUMMARY_INTERVAL_SECONDS:
                    if self._stats:  # Only print if we have stats
                        self.print_summary()
                    self._last_summary_time = now

    def _check_rotation(self):
        """Check if we need to rotate stats to a new day."""
        today = datetime.now().strftime("%Y-%m-%d")

        # Initialize rotation date on first check
        if self._last_rotation_date is None:
            self._last_rotation_date = today
            return

        # If the date has changed, rotate the file
        if today != self._last_rotation_date:
            self._rotate_stats()
            self._last_rotation_date = today

    def _rotate_stats(self):
        """Rotate current stats to a dated archive file and start fresh."""
        try:
            # Save current stats first
            if self._dirty:
                self._do_save()

            # Only rotate if the current file exists and has data
            if not os.path.exists(self._stats_file) or not self._stats:
                bt.logging.debug("No stats to rotate")
                return

            # Create archive filename with yesterday's date
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            base_dir = os.path.dirname(self._stats_file)
            archive_dir = os.path.join(base_dir, "webhook_stats_archive")
            os.makedirs(archive_dir, exist_ok=True)

            archive_file = os.path.join(archive_dir, f"webhook_stats_{yesterday}.json")

            # Copy current file to archive
            shutil.copy2(self._stats_file, archive_file)
            bt.logging.info(f"Rotated webhook stats to {archive_file}")

            # Reset current stats
            self._stats = {}
            self._do_save()

            # Clean up old archives (keep last 7 days)
            self._cleanup_old_archives(archive_dir, keep_days=7)

        except Exception as e:
            bt.logging.warning(f"Failed to rotate webhook stats: {e}")

    def _cleanup_old_archives(self, archive_dir: str, keep_days: int = 7):
        """Remove archive files older than keep_days."""
        try:
            cutoff = datetime.now() - timedelta(days=keep_days)
            for filename in os.listdir(archive_dir):
                if not filename.startswith("webhook_stats_") or not filename.endswith(".json"):
                    continue

                # Extract date from filename
                try:
                    date_str = filename.replace("webhook_stats_", "").replace(".json", "")
                    file_date = datetime.strptime(date_str, "%Y-%m-%d")
                    if file_date < cutoff:
                        filepath = os.path.join(archive_dir, filename)
                        os.remove(filepath)
                        bt.logging.debug(f"Removed old webhook stats archive: {filename}")
                except ValueError:
                    continue  # Skip files with unexpected names
        except Exception as e:
            bt.logging.debug(f"Error cleaning up old archives: {e}")

    def _mark_dirty(self):
        """Mark stats as needing to be saved."""
        self._dirty = True
        self._changes_since_save += 1

    def _do_save(self):
        """Actually save stats to file."""
        try:
            os.makedirs(os.path.dirname(self._stats_file), exist_ok=True)
            with self._file_lock:
                data = {
                    "last_updated": datetime.now().isoformat(),
                    "validators": {ip: stats.to_dict() for ip, stats in self._stats.items()},
                }
                with open(self._stats_file, "w") as f:
                    json.dump(data, f, indent=2)
            self._dirty = False
            self._last_save_time = time.time()
            self._changes_since_save = 0
        except Exception as e:
            bt.logging.warning(f"Failed to save webhook stats: {e}")

    def _save_stats(self):
        """Mark stats for saving (actual save happens in background thread)."""
        self._mark_dirty()

    def force_save(self):
        """Force an immediate save to disk."""
        self._do_save()

    def _extract_ip(self, webhook_url: str) -> str:
        """Extract IP/host from webhook URL."""
        try:
            parsed = urlparse(webhook_url)
            return parsed.netloc  # includes port, e.g. "192.168.1.1:8080"
        except Exception:
            return webhook_url

    def _get_stats(self, ip: str) -> ValidatorStats:
        """Get or create stats for an IP."""
        if ip not in self._stats:
            self._stats[ip] = ValidatorStats()
        return self._stats[ip]

    def record_success(self, webhook_url: str):
        """Record a successful webhook delivery."""
        ip = self._extract_ip(webhook_url)
        stats = self._get_stats(ip)
        stats.successes += 1
        stats.last_success = datetime.now().isoformat()
        self._save_stats()

    def record_failure(self, webhook_url: str, failure_type: str, reason: Optional[str] = None):
        """Record a failed webhook delivery."""
        ip = self._extract_ip(webhook_url)
        stats = self._get_stats(ip)
        stats.failures[failure_type] += 1
        stats.last_failure = datetime.now().isoformat()
        stats.last_failure_reason = reason
        self._save_stats()

    @classmethod
    def categorize_failure(cls, response: Optional[requests.Response] = None,
                          exception: Optional[Exception] = None,
                          response_text: Optional[str] = None) -> tuple[str, str]:
        """Categorize a failure and return (failure_type, reason)."""
        if exception:
            exc_str = str(exception).lower()
            if "timeout" in exc_str or "timed out" in exc_str:
                return cls.FAILURE_CONNECTION_TIMEOUT, str(exception)[:100]
            elif "connection" in exc_str:
                return cls.FAILURE_CONNECTION_ERROR, str(exception)[:100]
            return cls.FAILURE_UNKNOWN, str(exception)[:100]

        if response is not None:
            text = response_text or response.text[:200] if response.text else ""

            # Check for specific error messages
            if "empty binary payload" in text.lower():
                return cls.FAILURE_EMPTY_PAYLOAD, text

            # Categorize by status code
            if response.status_code == 400:
                return cls.FAILURE_HTTP_400, text
            elif response.status_code == 401:
                return cls.FAILURE_HTTP_401, text
            elif response.status_code == 404:
                return cls.FAILURE_HTTP_404, text
            elif response.status_code >= 500:
                return cls.FAILURE_HTTP_500, text
            else:
                return cls.FAILURE_HTTP_OTHER, f"HTTP {response.status_code}: {text}"

        return cls.FAILURE_UNKNOWN, "Unknown failure"

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all stats."""
        summary = {
            "total_validators": len(self._stats),
            "total_successes": sum(s.successes for s in self._stats.values()),
            "total_failures": sum(sum(s.failures.values()) for s in self._stats.values()),
            "validators": {ip: stats.to_dict() for ip, stats in self._stats.items()},
        }
        return summary

    def print_summary(self):
        """Print a formatted summary to logs."""
        summary = self.get_summary()
        bt.logging.info("=" * 60)
        bt.logging.info("WEBHOOK STATS SUMMARY")
        bt.logging.info("=" * 60)
        bt.logging.info(f"Total validators: {summary['total_validators']}")
        bt.logging.info(f"Total successes: {summary['total_successes']}")
        bt.logging.info(f"Total failures: {summary['total_failures']}")
        bt.logging.info("-" * 60)

        # Sort by total failures descending
        sorted_validators = sorted(
            summary["validators"].items(),
            key=lambda x: x[1]["total_failures"],
            reverse=True
        )

        for ip, stats in sorted_validators:
            success_rate = stats["successes"] / max(1, stats["successes"] + stats["total_failures"]) * 100
            bt.logging.info(f"{ip}: {stats['successes']} ok, {stats['total_failures']} fail ({success_rate:.1f}% success)")
            if stats["failures"]:
                for failure_type, count in stats["failures"].items():
                    bt.logging.info(f"    {failure_type}: {count}")

        bt.logging.info("=" * 60)


# Global stats tracker instance
_stats_tracker: Optional[WebhookStatsTracker] = None


def get_webhook_stats_tracker() -> WebhookStatsTracker:
    """Get the global webhook stats tracker."""
    global _stats_tracker
    if _stats_tracker is None:
        _stats_tracker = WebhookStatsTracker()
    return _stats_tracker


def print_webhook_stats():
    """Print webhook stats summary to logs."""
    get_webhook_stats_tracker().print_summary()


def get_webhook_stats_json() -> Dict[str, Any]:
    """Get webhook stats as a JSON-serializable dict."""
    return get_webhook_stats_tracker().get_summary()


def reset_webhook_stats():
    """Reset all webhook stats."""
    tracker = get_webhook_stats_tracker()
    tracker._stats = {}
    tracker._save_stats()
    bt.logging.info("Webhook stats have been reset")


def send_success_webhook(
    task: "GenerationTask",
    result: Dict[str, Any],
    hotkey: bt.Keypair,
    external_ip: str,
    port: int,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    timeout: float = 30.0
):
    binary_data = result.get("data")
    if not binary_data:
        bt.logging.error(f"send_success_webhook called with empty data for task {task.task_id}")
        return

    payload_size = len(binary_data)
    bt.logging.debug(f"Preparing webhook for task {task.task_id}: {payload_size} bytes")

    result_copy = {
        "data": binary_data,
        "metadata": result.get("metadata"),
        "url": result.get("url"),
    }

    def _send():
        try:
            _send_webhook(task, result_copy, True, hotkey, external_ip, port, max_retries, retry_delay, timeout)
        except Exception as e:
            bt.logging.error(f"Failed to send success webhook for task {task.task_id}: {e}")

    threading.Thread(target=_send, daemon=True).start()


def send_failure_webhook(
    task: "GenerationTask",
    hotkey: bt.Keypair,
    external_ip: str,
    port: int,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    timeout: float = 30.0
):
    def _send():
        try:
            _send_webhook(task, {}, False, hotkey, external_ip, port, max_retries, retry_delay, timeout)
        except Exception as e:
            bt.logging.error(f"Failed to send failure webhook for task {task.task_id}: {e}")

    threading.Thread(target=_send, daemon=True).start()


def _send_webhook(
    task: "GenerationTask",
    result: Dict[str, Any],
    is_success: bool,
    hotkey: bt.Keypair,
    external_ip: str,
    port: int,
    max_retries: int,
    retry_delay: float,
    timeout: float
):
    for attempt in range(max_retries):
        try:
            success = _attempt_webhook_send(task, result, is_success, hotkey, timeout)
            if success:
                bt.logging.info(f"Webhook sent successfully for task {task.task_id}")
                return
            else:
                bt.logging.warning(f"Webhook attempt {attempt + 1} failed for task {task.task_id}")

        except Exception as e:
            bt.logging.error(f"Webhook attempt {attempt + 1} error for task {task.task_id}: {e}")

        if attempt < max_retries - 1:
            time.sleep(retry_delay ** attempt)

    bt.logging.error(f"All webhook attempts failed for task {task.task_id}")


def _attempt_webhook_send(
    task: "GenerationTask",
    result: Dict[str, Any],
    is_success: bool,
    hotkey: bt.Keypair,
    timeout: float
) -> bool:
    stats = get_webhook_stats_tracker()
    binary_data = b""

    try:
        if not is_success:
            content_type = "application/octet-stream"
            headers = {
                "Content-Type": content_type,
                "task-id": task.task_id,
                "task-status": "failed",
                "error-message": task.error_message or "Unknown error",
                **generate_header(hotkey, binary_data, task.signed_by),
            }
        else:
            binary_data = result.get("data")
            if not binary_data:
                bt.logging.error(f"Task {task.task_id} marked as successful but no binary data in result")
                return False

            payload_size = len(binary_data)
            bt.logging.info(f"Webhook payload for task {task.task_id}: {payload_size} bytes of {task.modality} data")

            content_type = _detect_content_type(binary_data, task.modality)

            headers = {
                "Content-Type": content_type,
                "Content-Length": str(payload_size),
                "task-id": task.task_id,
                "task-status": "completed",
                **generate_header(hotkey, binary_data, task.signed_by),
            }

            if result.get("metadata"):
                metadata = result["metadata"]
                for key, value in metadata.items():
                    # Skip complex objects (dicts, lists) - they can't be serialized to HTTP headers
                    # C2PA manifests are already embedded in the binary data
                    if isinstance(value, (dict, list)) or value is None:
                        continue
                    header_key = f"x-meta-{key.replace('_', '-')}"
                    headers[header_key] = str(value)

        bt.logging.info(f"Sending webhook to: {task.webhook_url}")
        response = requests.post(
            task.webhook_url,
            data=binary_data,
            headers=headers,
            timeout=timeout
        )

        if response.status_code >= 300:
            response_text = response.text[:200] if response.text else ""
            bt.logging.warning(
                f"Webhook for task {task.task_id} returned status {response.status_code}: {response_text}"
            )
            # Track the failure
            failure_type, reason = WebhookStatsTracker.categorize_failure(
                response=response, response_text=response_text
            )
            stats.record_failure(task.webhook_url, failure_type, reason)
            return False

        # Track success
        stats.record_success(task.webhook_url)
        return True

    except requests.RequestException as e:
        bt.logging.warning(f"Webhook request failed for task {task.task_id}: {e}")
        # Track the failure
        failure_type, reason = WebhookStatsTracker.categorize_failure(exception=e)
        stats.record_failure(task.webhook_url, failure_type, reason)
        return False
