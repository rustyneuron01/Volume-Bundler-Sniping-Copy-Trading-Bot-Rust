#!/usr/bin/env python3
"""
Script to push separate image, video, and audio detector models and register on the Bittensor blockchain.

Usage:
    python push_model.py [--image-model <path>] [--video-model <path>] [--audio-model <path>] [options]

Options:
    --image-model PATH       Path to image detector zip file
    --video-model PATH       Path to video detector zip file
    --audio-model PATH       Path to audio detector zip file
    --wallet-name NAME       Bittensor wallet name (default: default)
    --wallet-hotkey KEY      Bittensor hotkey name (default: default)
    --netuid UID            Subnet UID (default: 34)
    --chain-endpoint URL     Subtensor network endpoint
    --retry-delay SECS      Retry delay in seconds (default: 60)

Example:
    python push_model.py --image-model image_detector.zip --video-model video_detector.zip --wallet-name miner1
"""
import argparse
import asyncio
import hashlib
import os
import sys
import traceback
import time
from typing import Optional

import bittensor as bt

# Try to import colorama for colored output, fallback to no colors if not available
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    HAS_COLORS = True
except ImportError:
    # Fallback colors that do nothing
    class Fore:
        GREEN = ""
        RED = ""
        YELLOW = ""
        BLUE = ""
        CYAN = ""
        WHITE = ""
    class Style:
        BRIGHT = ""
        RESET_ALL = ""
    HAS_COLORS = False

from gas.types import DiscriminatorModelId as ModelId
from gas.utils.chain_model_metadata_store import ChainModelMetadataStore
from gas.protocol.model_uploads import upload_single_modality


MODEL_UPLOAD_ENDPOINT = "https://upload.bitmind.ai/upload"


def print_success(message: str):
    """Print a success message in green"""
    print(f"{Fore.GREEN}✅ {message}{Style.RESET_ALL}")


def print_error(message: str):
    """Print an error message in red"""
    print(f"{Fore.RED}❌ {message}{Style.RESET_ALL}")


def print_warning(message: str):
    """Print a warning message in yellow"""
    print(f"{Fore.YELLOW}⚠️  {message}{Style.RESET_ALL}")


def print_info(message: str):
    """Print an info message in blue"""
    print(f"{Fore.BLUE}⛽ {message}{Style.RESET_ALL}")


def print_step(step_num: int, total_steps: int, message: str):
    """Print a step message with consistent formatting"""
    print(f"{Fore.CYAN}[{step_num}/{total_steps}] {message}{Style.RESET_ALL}")


def _print_submission_count(result: dict, modality: str, hotkey: str):
    """Print submission slot usage if the server returned count info."""
    submissions_used = result.get('submissions_used')
    submissions_max = result.get('submissions_max')
    if submissions_used is not None and submissions_max is not None:
        count_after = submissions_used + 1
        print_info(f"{count_after}/{submissions_max} {modality} models submitted for {hotkey}")


async def push_separate_models(
    image_model_path: Optional[str] = None,
    video_model_path: Optional[str] = None,
    audio_model_path: Optional[str] = None,
    wallet: bt.wallet = None,
    retry_delay_secs: int = 60,
    netuid: int = 34,
    chain_endpoint: Optional[str] = None,
):
    """Pushes separate image, video, and/or audio detector models and registers on the Bittensor blockchain.
    
    At least one model (image, video, or audio) must be provided.
    """
    if not image_model_path and not video_model_path and not audio_model_path:
        raise ValueError("At least one model (--image-model, --video-model, or --audio-model) must be provided")

    if image_model_path and not os.path.exists(image_model_path):
        raise FileNotFoundError(f"Image model file not found: {image_model_path}")
    if video_model_path and not os.path.exists(video_model_path):
        raise FileNotFoundError(f"Video model file not found: {video_model_path}")
    if audio_model_path and not os.path.exists(audio_model_path):
        raise FileNotFoundError(f"Audio model file not found: {audio_model_path}")

    results = {}
    upload_count = 0
    total_uploads = (1 if image_model_path else 0) + \
                    (1 if video_model_path else 0) + \
                    (1 if audio_model_path else 0) # Updated total uploads count

    if image_model_path:
        upload_count += 1
        print()
        print_step(upload_count, total_uploads + 1, "Uploading image model to cloud inference system...")
        try:
            image_result = upload_single_modality(
                wallet,
                image_model_path,
                'image',
                MODEL_UPLOAD_ENDPOINT
            )
            results['image'] = image_result
            
            if not image_result['success']:
                print_error(f"Image model upload failed at step: {image_result.get('step', 'unknown')}")
                print_error(f"Error: {image_result.get('error', 'Unknown error')}")
                if image_result.get('response'):
                    print_error(f"Server response: {image_result['response']}")
                return False
            
            print_success("Image model uploaded successfully!")
            _print_submission_count(image_result, 'image', wallet.hotkey.ss58_address)
        except Exception as e:
            print_error(f"Image model upload failed with exception: {e}")
            return False

    if video_model_path:
        upload_count += 1
        print()
        print_step(upload_count, total_uploads + 1, "Uploading video model to cloud inference system...")
        try:
            video_result = upload_single_modality(
                wallet,
                video_model_path,
                'video',
                MODEL_UPLOAD_ENDPOINT
            )
            results['video'] = video_result
            
            if not video_result['success']:
                print_error(f"Video model upload failed at step: {video_result.get('step', 'unknown')}")
                print_error(f"Error: {video_result.get('error', 'Unknown error')}")
                if video_result.get('response'):
                    print_error(f"Server response: {video_result['response']}")
                return False

            print_success("Video model uploaded successfully!")
            _print_submission_count(video_result, 'video', wallet.hotkey.ss58_address)
        except Exception as e:
            print_error(f"Video model upload failed with exception: {e}")
            return False

    if audio_model_path:
        upload_count += 1
        print()
        print_step(upload_count, total_uploads + 1, "Uploading audio model to cloud inference system...")
        try:
            audio_result = upload_single_modality(
                wallet,
                audio_model_path,
                'audio',
                MODEL_UPLOAD_ENDPOINT
            )
            results['audio'] = audio_result
            
            if not audio_result['success']:
                print_error(f"Audio model upload failed at step: {audio_result.get('step', 'unknown')}")
                print_error(f"Error: {audio_result.get('error', 'Unknown error')}")
                if audio_result.get('response'):
                    print_error(f"Server response: {audio_result['response']}")
                return False
            
            print_success("Audio model uploaded successfully!")
            _print_submission_count(audio_result, 'audio', wallet.hotkey.ss58_address)
        except Exception as e:
            print_error(f"Audio model upload failed with exception: {e}")
            return False

    print()
    print_step(upload_count + 1, total_uploads + 1, "Registering model metadata on blockchain...")
    
    if not chain_endpoint:
        chain_endpoint = "finney" if netuid == 34 else "test"

    print_info(f"Connecting to subnet {netuid} via {chain_endpoint}")
    subtensor = bt.subtensor(network=chain_endpoint)
    metadata_store = ChainModelMetadataStore(subtensor, netuid)

    # Register each model separately on the blockchain
    for modality, result in results.items():
        model_key = result.get("r2_key", "")
        if not model_key:
            print_error(f"{modality.capitalize()} model key not provided in upload response")
            return False

        # Create hash for this specific model
        model_hash = hashlib.sha256(
            f"{result['file_hash']}{wallet.hotkey.ss58_address}".encode()
        ).hexdigest()
        hash_value = str(hash(model_hash))

        model_id = ModelId(key=model_key, hash=hash_value)
        print_info(f"Registering {modality} model (ID: {model_id.key})...")

        while True:
            try:
                await metadata_store.store_model_metadata(wallet, model_id)
                
                # Verify registration
                uid = subtensor.get_uid_for_hotkey_on_subnet(
                    wallet.hotkey.ss58_address, netuid
                )
                model_metadata = await metadata_store.retrieve_model_metadata(
                    uid, wallet.hotkey.ss58_address
                )

                if (
                    not model_metadata
                    or model_metadata.id.to_compressed_str() != model_id.to_compressed_str()
                ):
                    print_error(f"{modality.capitalize()} model metadata verification failed")
                    print_error(f"Expected: {model_id}")
                    print_error(f"Got: {model_metadata}")
                    raise ValueError(f"{modality.capitalize()} metadata verification failed")

                print_success(f"{modality.capitalize()} model registered on blockchain!")
                break  # Success, move to next model

            except Exception as e:
                print_error(f"Failed to register {modality} model on blockchain: {e}")
                print_warning(f"Retrying in {retry_delay_secs} seconds...")
                time.sleep(retry_delay_secs)

    print_success("All models registered successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Push image, video, and/or audio detector models and register on Bittensor. At least one model must be provided."
    )

    # Model paths (at least one required)
    parser.add_argument(
        "--image-model",
        help="Path to image detector zip file",
    )
    parser.add_argument(
        "--video-model",
        help="Path to video detector zip file",
    )
    parser.add_argument(
        "--audio-model",
        help="Path to audio detector zip file",
    )
    parser.add_argument(
        "--wallet-name",
        default="default",
        help="Bittensor wallet name"
    )
    parser.add_argument(
        "--wallet-hotkey",
        default="default",
        help="Bittensor hotkey name"
    )
    parser.add_argument(
        "--netuid",
        type=int,
        default=34,
        help="Subnet UID"
    )
    parser.add_argument(
        "--chain-endpoint",
        default="",
        help="Subtensor network",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=60,
        help="Retry delay in seconds"
    )

    args = parser.parse_args()

    # Validate at least one model is provided
    if not args.image_model and not args.video_model and not args.audio_model:
        parser.error("At least one model must be provided: --image-model, --video-model, or --audio-model")

    print()
    print(f"{Fore.CYAN}{Style.BRIGHT}=== Model Push Configuration ==={Style.RESET_ALL}")
    if args.image_model:
        print(f"Image Model: {args.image_model}")
    if args.video_model:
        print(f"Video Model: {args.video_model}")
    if args.audio_model:
        print(f"Audio Model: {args.audio_model}")
    print(f"Wallet: {args.wallet_name}/{args.wallet_hotkey}")
    print(f"Subnet UID: {args.netuid}")
    print(f"Chain Endpoint: {args.chain_endpoint}")
    print()

    print(f"{Fore.CYAN}{Style.BRIGHT}=== Starting Model Push ==={Style.RESET_ALL}")

    # Initialize wallet
    try:
        wallet = bt.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
        print_info(f"Hotkey: {wallet.hotkey.ss58_address}")
    except Exception as e:
        print_error(f"Failed to initialize wallet: {e}")
        sys.exit(1)

    # Check if hotkey is registered
    netuid = args.netuid
    mg = bt.metagraph(netuid=netuid, network="finney" if netuid == 34 else "test")
    hotkey = wallet.hotkey.ss58_address
    if hotkey not in mg.hotkeys:
        print_warning(f"Hotkey {hotkey} not registered on netuid {netuid}")
        print_warning("This may cause issues with model registration")

    try:
        success = asyncio.run(
            push_separate_models(
                image_model_path=args.image_model,
                video_model_path=args.video_model,
                audio_model_path=args.audio_model,
                wallet=wallet,
                retry_delay_secs=args.retry_delay,
                netuid=netuid,
                chain_endpoint=args.chain_endpoint
            )
        )
        
        print()
        if success:
            print_success("🎉 Model push completed successfully!")
        else:
            print_error("💥 Model push failed!")
            sys.exit(1)

    except Exception as e:
        print()
        print_error(f"💥 Model push failed with error: {e}")
        print_info("Full traceback:")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
