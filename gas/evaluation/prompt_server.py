#!/usr/bin/env python3
"""
Secret Prompt Server - Standalone Edition

A completely standalone FastAPI server that provides prompts to authorized requesters
via Epistula v2 protocol. This script has no dependencies on bitmind-subnet modules
and can be run independently.

Requirements:
    pip install fastapi uvicorn bittensor substrate-interface

Usage:
    # Direct execution
    python prompt_server.py --port 40109 --wallet.name validator --wallet.hotkey default

    # Or from bitmind-subnet directory  
    python -m gas.evaluation.prompt_server --port 40109

    # With PM2
    pm2 start prompt_server.py --name prompt-server --interpreter python -- --port 40109
"""
import argparse
import contextlib
import logging
import random
import sqlite3
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import List, Optional

import bittensor as bt
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from substrateinterface import Keypair


# =============================================================================
# Configuration
# =============================================================================

EPISTULA_VERSION = "2"
ALLOWED_DELTA_MS = 15000  # 15 second signature validity window

# Hardcoded allowed hotkey - only this hotkey can request prompts
DEFAULT_ALLOWED_HOTKEY = "5Fpg38VX1xHBnrMagu3ijf3graUpsAnkCVhMkQEJbt4YdA4G"
DEFAULT_PORT = 40109
DEFAULT_CACHE_DIR = "~/.cache/sn34"


# =============================================================================
# Data Types
# =============================================================================

@dataclass
class PromptEntry:
    """Simple prompt entry from database."""
    id: str
    content: str
    content_type: str = "prompt"
    modality: Optional[str] = None


# =============================================================================
# Epistula Protocol (Inlined)
# =============================================================================

def verify_signature(
    signature: str,
    body: bytes,
    timestamp: str,
    uuid: str,
    signed_for: str,
    signed_by: str,
    now: int,
) -> Optional[str]:
    """
    Verify an Epistula v2 signature.
    
    Returns None if valid, error message string if invalid.
    """
    if not isinstance(signature, str):
        return "Invalid Signature"
    
    try:
        timestamp_int = int(timestamp)
    except (ValueError, TypeError):
        return "Invalid Timestamp"
    
    if not isinstance(signed_by, str):
        return "Invalid Sender key"
    if not isinstance(signed_for, str):
        return "Invalid receiver key"
    if not isinstance(uuid, str):
        return "Invalid uuid"
    if not isinstance(body, bytes):
        return "Body is not of type bytes"
    
    # Check timestamp freshness
    if timestamp_int + ALLOWED_DELTA_MS < now:
        staleness_ms = now - timestamp_int
        staleness_seconds = staleness_ms / 1000.0
        return f"Request is too stale: {staleness_seconds:.1f}s old (limit: {ALLOWED_DELTA_MS/1000.0}s)"
    
    # Verify signature
    keypair = Keypair(ss58_address=signed_by)
    message = f"{sha256(body).hexdigest()}.{uuid}.{timestamp_int}.{signed_for}"
    
    try:
        verified = keypair.verify(message, signature)
        if not verified:
            return "Signature Mismatch"
    except Exception as e:
        return f"Signature verification error: {e}"
    
    return None


# =============================================================================
# Database Access (Minimal, Standalone)
# =============================================================================

class PromptDatabase:
    """
    Minimal database access for reading prompts.
    Directly accesses the SQLite database without ContentManager.
    """
    
    def __init__(self, cache_dir: Path):
        self.db_path = cache_dir / "prompts.db"
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Prompt database not found at {self.db_path}. "
                "Make sure the validator has been running and has populated the cache."
            )
    
    @contextlib.contextmanager
    def _get_connection(self):
        """Get a read-only database connection."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=10.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def sample_prompts(self, k: int = 1) -> List[PromptEntry]:
        """
        Sample k random prompts from the database.
        
        Args:
            k: Number of prompts to sample
            
        Returns:
            List of PromptEntry objects
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, content, content_type, modality
                FROM prompts
                WHERE content_type = 'prompt'
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (k,)
            )
            rows = cursor.fetchall()
            
            return [
                PromptEntry(
                    id=row["id"],
                    content=row["content"],
                    content_type=row["content_type"],
                    modality=row["modality"],
                )
                for row in rows
            ]
    
    def get_prompt_count(self) -> int:
        """Get total number of prompts in database."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM prompts WHERE content_type = 'prompt'"
            )
            return cursor.fetchone()[0]


# =============================================================================
# Prompt Server
# =============================================================================

class PromptServer:
    """
    A FastAPI server that provides prompts to authorized requesters.
    
    Only requests signed by the allowed hotkey and intended for this 
    validator's hotkey are accepted. Uses Epistula v2 protocol for authentication.
    """
    
    def __init__(
        self,
        port: int,
        validator_hotkey: str,
        cache_dir: Path,
        allowed_hotkey: str = DEFAULT_ALLOWED_HOTKEY,
    ):
        """
        Initialize the prompt server.
        
        Args:
            port: Port to listen on
            validator_hotkey: This validator's SS58 hotkey address
            cache_dir: Path to the cache directory containing prompts.db
            allowed_hotkey: Hotkey allowed to request prompts
        """
        self.port = port
        self.validator_hotkey = validator_hotkey
        self.allowed_hotkey = allowed_hotkey
        self.db = PromptDatabase(cache_dir)
        
        self.app = FastAPI(
            title="Secret Prompt Server",
            docs_url=None,
            redoc_url=None,
            openapi_url=None,
        )
        self._setup_routes()
        
        print(f"[PromptServer] Initialized on port {port}")
        print(f"[PromptServer] Validator hotkey: {validator_hotkey}")
        print(f"[PromptServer] Allowed requester: {allowed_hotkey}")
        print(f"[PromptServer] Database: {self.db.db_path}")
        print(f"[PromptServer] Prompt count: {self.db.get_prompt_count()}")

    def _setup_routes(self):
        """Set up the FastAPI routes."""
        
        @self.app.post("/gp")
        async def get_prompt(request: Request):
            """
            Get a random prompt from the cache.
            
            Requires Epistula v2 authentication with:
            - Epistula-Signed-By: Must be the allowed hotkey
            - Epistula-Signed-For: Must be this validator's hotkey
            
            Returns:
                JSON with {"id": prompt_id, "content": prompt_content}
            """
            # Validate Epistula version
            version = request.headers.get("Epistula-Version")
            if version != EPISTULA_VERSION:
                raise HTTPException(status_code=400, detail="Unknown version")

            # Validate sender is allowed
            signed_by = request.headers.get("Epistula-Signed-By")
            signed_for = request.headers.get("Epistula-Signed-For")
            
            if signed_by != self.allowed_hotkey:
                print(f"[PromptServer] Rejected: unauthorized requester {signed_by}")
                raise HTTPException(status_code=403, detail="Forbidden")

            # Validate request is for this validator
            if signed_for != self.validator_hotkey:
                raise HTTPException(status_code=400, detail="Not intended for self")

            # Verify signature
            body = await request.body()
            now = round(time.time() * 1000)
            
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
                print(f"[PromptServer] Signature verification failed: {err}")
                raise HTTPException(status_code=400, detail=err)

            # Get a random prompt
            prompts = self.db.sample_prompts(k=1)
            if not prompts:
                raise HTTPException(status_code=404, detail="No prompts available")

            print(f"[PromptServer] Serving prompt {prompts[0].id[:8]}... to {signed_by[:16]}...")
            return {"id": prompts[0].id, "content": prompts[0].content}

        @self.app.get("/health")
        async def health_check():
            """Simple health check endpoint (no auth required)."""
            try:
                count = self.db.get_prompt_count()
                return {
                    "status": "healthy",
                    "service": "prompt_server",
                    "prompt_count": count,
                }
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "error": str(e),
                }

    def run(self, log_level: str = "warning"):
        """
        Run the server using uvicorn.
        
        Args:
            log_level: Uvicorn log level (default: warning to reduce noise)
        """
        print(f"[PromptServer] Starting on http://0.0.0.0:{self.port}")
        print(f"[PromptServer] Health check: http://localhost:{self.port}/health")
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level=log_level,
            access_log=False,
        )


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for running the prompt server standalone."""
    parser = argparse.ArgumentParser(
        description="Run the secret prompt server (standalone)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python prompt_server.py --port 40109 --wallet.name validator --wallet.hotkey default
    
    # With custom cache directory
    python prompt_server.py --cache-dir /data/sn34-cache
    
    # With PM2
    pm2 start prompt_server.py --name prompt-server --interpreter python -- \\
        --port 40109 --wallet.name validator --wallet.hotkey default
        
    # Test health endpoint
    curl http://localhost:40109/health
        """
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to listen on (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--wallet.name",
        dest="wallet_name",
        type=str,
        default="default",
        help="Bittensor wallet name (default: default)"
    )
    parser.add_argument(
        "--wallet.hotkey",
        dest="wallet_hotkey",
        type=str,
        default="default",
        help="Bittensor hotkey name (default: default)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help=f"Cache directory containing prompts.db (default: {DEFAULT_CACHE_DIR})"
    )
    parser.add_argument(
        "--allowed-hotkey",
        type=str,
        default=DEFAULT_ALLOWED_HOTKEY,
        help="Hotkey allowed to request prompts"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()

    # Suppress noisy logs unless verbose
    if not args.verbose:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    # Get wallet hotkey
    print(f"[PromptServer] Loading wallet: {args.wallet_name}/{args.wallet_hotkey}")
    wallet = bt.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
    validator_hotkey = wallet.hotkey.ss58_address

    # Determine cache directory
    if args.cache_dir:
        cache_dir = Path(args.cache_dir).expanduser()
    else:
        cache_dir = Path(DEFAULT_CACHE_DIR).expanduser()

    print(f"[PromptServer] Cache directory: {cache_dir}")

    # Create and run server
    try:
        server = PromptServer(
            port=args.port,
            validator_hotkey=validator_hotkey,
            cache_dir=cache_dir,
            allowed_hotkey=args.allowed_hotkey,
        )
        
        log_level = "info" if args.verbose else "warning"
        server.run(log_level=log_level)
    except FileNotFoundError as e:
        print(f"[PromptServer] Error: {e}")
        exit(1)
    except KeyboardInterrupt:
        print("\n[PromptServer] Shutting down...")


if __name__ == "__main__":
    main()
