import json
import asyncio
import aiohttp
import traceback
import bittensor as bt
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from gas.protocol.epistula import generate_header
from gas.types import Modality, MediaType


async def get_miner_type(
    uid: int,
    axon_info: bt.AxonInfo,
    session: aiohttp.ClientSession,
    hotkey: bt.Keypair,
    total_timeout: float,
    connect_timeout: Optional[float] = None,
    sock_connect_timeout: Optional[float] = None,
) -> Optional[str]:
    """
    Query a miner's type by hitting its miner_info endpoint.

    Args:
        uid: miner uid
        axon_info: miner AxonInfo
        session: aiohttp client session
        hotkey: validator hotkey Keypair for signing the request
        total_timeout: Total timeout for the request
        connect_timeout: Connection timeout
        sock_connect_timeout: Socket connection timeout

    Returns:
        The miner's type as a string, or None if the request fails
    """
    response = {"uid": uid, "status": 500, "error": "", "miner_type": None}

    try:
        base_url = f"http://{axon_info.ip}:{axon_info.port}"
        url = f"{base_url}/miner_info"

        headers = generate_header(hotkey, b"", axon_info.hotkey)
        timeout = aiohttp.ClientTimeout(
            total=total_timeout,
            connect=connect_timeout,
            sock_connect=sock_connect_timeout,
        )

        async with session.get(url, headers=headers, timeout=timeout) as res:
            response["status"] = res.status
            if res.status != 200:
                response["error"] = f"HTTP {res.status}"
                return response

            try:
                info = await res.json()
                if not isinstance(info, dict):
                    response["error"] = "Response was not a dictionary"
                    return response

                response["miner_type"] = info.get("miner_type")
                return response

            except json.JSONDecodeError:
                response["error"] = "Failed to decode JSON response"
                return response

    except asyncio.TimeoutError:
        response["status"] = 408
        response["error"] = "Request timed out"
    except aiohttp.ClientConnectorError as e:
        response["status"] = 503
        response["error"] = f"Connection error: {str(e)}"
    except aiohttp.ClientError as e:
        response["status"] = 400
        response["error"] = f"Network error: {str(e)}"
    except Exception as e:
        response["error"] = f"Unknown error: {str(e)}"

    return response


async def query_generative_miner(
    uid: int,
    axon_info: bt.AxonInfo,
    session: aiohttp.ClientSession,
    hotkey: bt.Keypair,
    prompt: str,
    modality: Modality,
    webhook_url: str,
    parameters: Dict[str, Any],
    total_timeout: float,
    connect_timeout: Optional[float] = None,
    sock_connect_timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Query a generative miner with a prompt.

    Args:
        uid: miner uid
        axon_info: miner AxonInfo
        session: aiohttp client session
        hotkey: validator hotkey Keypair for signing the request
        prompt: The prompt for the generative task.
        webhook_url: The URL for the miner to call back to.
        parameters: Additional parameters for the generation.
        total_timeout: Total timeout for the request
        connect_timeout: Connection timeout
        sock_connect_timeout: Socket connection timeout

    Returns:
        Dictionary containing the response.
    """
    response = {
        "uid": uid,
        "hotkey": axon_info.hotkey,
        "status": 500,
        "task_id": None,
        "error": "",
    }

    try:
        base_url = f"http://{axon_info.ip}:{axon_info.port}"
        # action = "gen" if media_type == MediaType.SYNTHETIC else "mod"
        url = f"{base_url}/gen_{modality.value}"

        payload = {
            "prompt": prompt,
            "parameters": parameters,
        }
        body_bytes = json.dumps(payload).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "X-Webhook-URL": webhook_url,
            **generate_header(hotkey, body_bytes, axon_info.hotkey),
        }

        async with session.post(
            url,
            headers=headers,
            data=body_bytes,
            timeout=aiohttp.ClientTimeout(
                total=total_timeout,
                connect=connect_timeout,
                sock_connect=sock_connect_timeout,
            ),
        ) as res:
            response["status"] = res.status
            if res.status not in [201, 202]:
                response["error"] = f"HTTP {res.status} error: {await res.text()}"
                return response
            try:
                data = await res.json()
                if "task_id" not in data:
                    response["error"] = "Missing task_id in response"
                    return response

                response["task_id"] = data["task_id"]
                return response

            except json.JSONDecodeError:
                response["error"] = "Failed to decode JSON response"
                return response

    except asyncio.TimeoutError:
        response["status"] = 408
        response["error"] = "Request timed out"
    except aiohttp.ClientConnectorError as e:
        response["status"] = 503
        response["error"] = f"Connection error: {str(e)}"
    except aiohttp.ClientError as e:
        response["status"] = 400
        response["error"] = f"Network error: {str(e)}"
    except Exception as e:
        response["error"] = f"Unknown error: {str(e)}"

    return response


async def get_escrow_addresses(
    hotkey,
    base_url: str = "https://gas.bitmind.ai",
    activity_hours: int = 24,
) -> Optional[Dict[str, str]]:
    """
    Fetch escrow addresses from the gas-api /escrow-addresses endpoint.

    The endpoint returns escrow addresses for eligible validators based on:
    - Whitelist membership
    - Recent gasstation activity

    Args:
        hotkey: Validator hotkey Keypair for signing the request
        base_url: Base URL for the gas API
        activity_hours: Hours to look back for activity verification

    Returns:
        Dictionary with keys: image_escrow, video_escrow, audio_escrow
        Each value is the SS58 address. Returns None on failure.
    """
    try:
        bt.logging.info(f"Fetching escrow addresses from {base_url}/api/v1/validator/escrow-addresses")

        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            url = f"{base_url}/api/v1/validator/escrow-addresses?activity_hours={activity_hours}"
            epistula_headers = generate_header(hotkey, b"", None)

            async with session.get(url, headers=epistula_headers) as response:
                if response.status == 200:
                    addresses = await response.json()
                    #bt.logging.info(f"Successfully fetched escrow addresses: {list(addresses.keys())}")
                    return addresses
                else:
                    error_text = await response.text()
                    bt.logging.warning(
                        f"Failed to fetch escrow addresses: HTTP {response.status}, response: {error_text}"
                    )
                    return None

    except Exception as e:
        bt.logging.error(f"Error fetching escrow addresses from API: {e}")
        bt.logging.error(traceback.format_exc())
        return None


async def get_benchmark_results(
    hotkey, metagraph: bt.metagraph, base_url: str = "https://gas.bitmind.ai",
    max_retries: int = 3,
):
    """
    Query the remote benchmark API for generator fool rates.
    Only returns results from the last week based on the finished_at field.
    Retries on failure so transient errors don't zero out generator weights.

    Args:
        hotkey: Validator hotkey for authentication
        metagraph: Bittensor metagraph for SS58 to UID mapping
        base_url: Base URL for the benchmark API
        max_retries: Number of attempts (default 3)

    Returns:
        list: generator_results containing API response data
    """
    generator_results = []
    one_week_ago = datetime.now() - timedelta(weeks=1)
    after_timestamp = one_week_ago.isoformat()
    generator_url = f"{base_url}/api/v1/validator/generator-results?validator_address={hotkey.ss58_address}&after={after_timestamp}"
    # API can take 60s+ (DuckDB init + parquet query); avoid client timeout before server responds
    timeout = aiohttp.ClientTimeout(total=90)

    for attempt in range(max_retries):
        try:
            bt.logging.info(
                f"Fetching benchmark results from {base_url}"
                + (f" (attempt {attempt + 1}/{max_retries})" if attempt > 0 else "")
            )
            bt.logging.debug(f"Requesting generator results from: {generator_url}")

            epistula_headers = generate_header(hotkey, b"", None)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(generator_url, headers=epistula_headers) as response:
                    if response.status == 200:
                        generator_results = await response.json()
                        bt.logging.info(
                            f"Successfully fetched {len(generator_results)} generator results from last week"
                        )
                        return generator_results
                    error_text = await response.text()
                    bt.logging.warning(
                        f"Failed to fetch generator results: HTTP {response.status}, response: {error_text}"
                    )
        except Exception as e:
            bt.logging.error(
                f"Error fetching benchmark results (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}"
            )
            bt.logging.debug(traceback.format_exc())
            if attempt < max_retries - 1:
                await asyncio.sleep(2 * (attempt + 1))
            else:
                bt.logging.error(
                    "Benchmark results unavailable after all retries. "
                    "Generator weights will use local verification only (fallback)."
                )

    return generator_results
