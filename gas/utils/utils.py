import concurrent.futures
import functools
import json
import os
import traceback

import bittensor as bt


def print_info(metagraph, hotkey, block, isMiner=True):
    uid = metagraph.hotkeys.index(hotkey)
    log = f"UID:{uid} | Block:{block} | Consensus:{metagraph.C[uid]} | "
    if isMiner:
        bt.logging.info(
            log
            + f"Stake:{metagraph.S[uid]} | Trust:{metagraph.T[uid]} | Incentive:{metagraph.I[uid]} | Emission:{metagraph.E[uid]}"
        )
        return
    bt.logging.info(log + f"VTrust:{metagraph.Tv[uid]} | ")


def fail_with_none(message: str = ""):
    def outer(func):
        def inner(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                bt.logging.error(message)
                bt.logging.error(str(e))
                bt.logging.error(traceback.format_exc())
                return None

        return inner

    return outer


def on_block_interval(interval_attr_name):
    """
    Decorator for methods that should only execute at specific block intervals.

    Args:
        interval_attr_name: String name of the config attribute that specifies the interval
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, block, *args, **kwargs):
            if not self.initialization_complete:
                bt.logging.error(f"Block callbacks waiting for validator initialization to complete")
                return 
            interval = getattr(self.config, interval_attr_name)
            if interval is None:
                bt.logging.error(f"No interval found for {interval_attr_name}")
            if (
                block == 0 or block % interval == 0
            ):  # Allow execution on block 0 for initialization
                return await func(self, block, *args, **kwargs)
            return None

        return wrapper

    return decorator


class ExitContext:
    """
    Using this as a class lets us pass this to other threads
    """

    isExiting: bool = False

    def startExit(self, *_):
        if self.isExiting:
            exit()
        self.isExiting = True

    def __bool__(self):
        return self.isExiting


def get_metadata(media_path):
    """Get metadata for a media file if it exists."""
    base_path = os.path.splitext(media_path)[0]
    json_path = f"{base_path}.json"

    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            bt.logging.error(f"Warning: Could not parse JSON file: {json_path}")
            return {}
    return {}


def get_file_modality(filepath: str) -> str:
    """
    Determine the type of media file based on its extension.

    Args:
        filepath: Path to the media file

    Returns:
        "image", "video", or "file" based on the file extension
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]:
        return "image"
    elif ext in [".mp4", ".avi", ".mov", ".webm", ".mkv", ".flv"]:
        return "video"
    else:
        return "file"


def run_in_thread(func, timeout: int = 60):
    """Run a function in a thread with timeout"""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function timed out after {timeout} seconds")
