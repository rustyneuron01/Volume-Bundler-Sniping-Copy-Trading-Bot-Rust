import json
from hashlib import sha256
from uuid import uuid4
from math import ceil
from typing import Annotated, Any, Dict, Optional

import traceback
import bittensor as bt
import time
import httpx
from substrateinterface import Keypair
from fastapi import Request, HTTPException


EPISTULA_VERSION = str(2)
MIN_VALIDATOR_STAKE = 20000


def generate_header(
    hotkey: Keypair,
    body: Any,
    signed_for: Optional[str] = None,
) -> Dict[str, Any]:
    timestamp = round(time.time() * 1000)
    timestampInterval = ceil(timestamp / 1e4) * 1e4
    uuid = str(uuid4())
    req_hash = None

    if isinstance(body, bytes):
        req_hash = sha256(body).hexdigest()
    elif isinstance(body, dict):
        body_copy = {}
        for key, value in body.items():
            if isinstance(value, bytes):
                body_copy[key] = sha256(value).hexdigest()
            else:
                body_copy[key] = value
        req_hash = sha256(json.dumps(body_copy).encode("utf-8")).hexdigest()
    else:
        req_hash = sha256(json.dumps(body).encode("utf-8")).hexdigest()

    headers = {
        "Epistula-Version": EPISTULA_VERSION,
        "Epistula-Timestamp": str(timestamp),
        "Epistula-Uuid": uuid,
        "Epistula-Signed-By": hotkey.ss58_address,
        "Epistula-Request-Signature": "0x"
        + hotkey.sign(f"{req_hash}.{uuid}.{timestamp}.{signed_for or ''}").hex(),
    }
    if signed_for:
        headers["Epistula-Signed-For"] = signed_for
        headers["Epistula-Secret-Signature-0"] = (
            "0x" + hotkey.sign(str(timestampInterval - 1) + "." + signed_for).hex()
        )
        headers["Epistula-Secret-Signature-1"] = (
            "0x" + hotkey.sign(str(timestampInterval) + "." + signed_for).hex()
        )
        headers["Epistula-Secret-Signature-2"] = (
            "0x" + hotkey.sign(str(timestampInterval + 1) + "." + signed_for).hex()
        )
    return headers


def verify_signature(
    signature, body: bytes, timestamp, uuid, signed_for, signed_by, now
) -> Optional[Annotated[str, "Error Message"]]:
    if not isinstance(signature, str):
        return "Invalid Signature"
    timestamp = int(timestamp)
    if not isinstance(timestamp, int):
        return "Invalid Timestamp"
    if not isinstance(signed_by, str):
        return "Invalid Sender key"
    if not isinstance(signed_for, str):
        return "Invalid receiver key"
    if not isinstance(uuid, str):
        return "Invalid uuid"
    if not isinstance(body, bytes):
        return "Body is not of type bytes"
    ALLOWED_DELTA_MS = 15000
    keypair = Keypair(ss58_address=signed_by)
    if timestamp + ALLOWED_DELTA_MS < now:
        staleness_ms = now - timestamp
        staleness_seconds = staleness_ms / 1000.0
        return f"Request is too stale: {staleness_seconds:.1f}s old (limit: {ALLOWED_DELTA_MS/1000.0}s)"
    message = f"{sha256(body).hexdigest()}.{uuid}.{timestamp}.{signed_for}"
    verified = keypair.verify(message, signature)
    if not verified:
        return "Signature Mismatch"
    return None


def create_header_hook(hotkey, axon_hotkey, model):
    async def add_headers(request: httpx.Request):
        for key, header in generate_header(hotkey, request.read(), axon_hotkey).items():
            request.headers[key] = header

    return add_headers


async def _verify_request(
    request: Request,
    wallet: bt.Wallet,
    metagraph: bt.Metagraph,
    no_force_validator_permit: bool
):
    now = round(time.time() * 1000)

    signed_by = request.headers.get("Epistula-Signed-By")
    signed_for = request.headers.get("Epistula-Signed-For")
    client_ip = request.client.host if request.client else "unknown"

    if signed_for != wallet.hotkey.ss58_address:
        bt.logging.error(f"Request not intended for self from {signed_by} (IP: {client_ip})")
        raise HTTPException(
            status_code=400, detail="Bad Request, message is not intended for self"
        )
    
    if signed_by not in metagraph.hotkeys:
        bt.logging.error(f"Signer not in metagraph: {signed_by} (IP: {client_ip})")
        raise HTTPException(status_code=401, detail="Signer not in metagraph")

    uid = metagraph.hotkeys.index(signed_by)
    stake = metagraph.S[uid].item()

    if not no_force_validator_permit and stake < MIN_VALIDATOR_STAKE:
        bt.logging.warning(
            f"Blacklisting request from {signed_by} [uid={uid}], not enough stake -- {stake}"
        )
        raise HTTPException(status_code=401, detail=f"Stake below minimum: {stake}")

    body = await request.body()
    err = verify_signature(
        request.headers.get("Epistula-Request-Signature"),
        body,
        request.headers.get("Epistula-Timestamp"),
        request.headers.get("Epistula-Uuid"),
        signed_for,
        signed_by,
        now,
    )

    if err:
        bt.logging.error(f"UID {uid} (IP: {client_ip}): {err}")
        raise HTTPException(status_code=400, detail=err)


async def determine_epistula_version_and_verify(
    request: Request,
    wallet: bt.Wallet,
    metagraph: bt.Metagraph,
    no_force_validator_permit: bool
):
    version = request.headers.get("Epistula-Version")
    if version == EPISTULA_VERSION:
        await _verify_request(request, wallet, metagraph, no_force_validator_permit)
        return
    raise HTTPException(status_code=400, detail="Unknown Epistula version")


def get_verifier(
    wallet: bt.Wallet,
    metagraph: bt.Metagraph,
    no_force_validator_permit: bool = False
):
    async def verifier(request: Request):
        await determine_epistula_version_and_verify(
            request,
            wallet,
            metagraph,
            no_force_validator_permit,
        )
    return verifier
