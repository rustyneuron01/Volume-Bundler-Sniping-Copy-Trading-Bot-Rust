import glob
import os
import shutil
import time

import bittensor as bt
import wandb
from gas import __version__


def init_wandb(
    config: bt.config, process: str, uid: int, hotkey: bt.Keypair, wandb_dir: str = None
) -> wandb.run:
    """
    Initialize a Weights & Biases run.

    Args:
        config: Bittensor config object
        process: Valid options are 'validator', 'data-generator', 'media-store'
        uid: Validator uid
        hotkey: Bittensor keypair for signing the run
        wandb_dir: Optional directory for wandb files

    Returns:
        wandb.run: The initialized wandb run, or None if initialization fails
    """
    project = f"subnet-{config.netuid}-{process}"
    run_name = f"{process}-{uid}-{__version__}"
    config.run_name = run_name
    config.uid = uid
    config.hotkey = hotkey.ss58_address
    config.version = __version__

    bt.logging.info(f"Initializing wandb run in '{config.wandb.entity}/{project}'")

    try:
        run = wandb.init(
            name=run_name,
            project=project,
            entity=config.wandb.entity,
            config=config,
            dir=wandb_dir if wandb_dir else config.full_path,
            reinit=True,
        )
    except wandb.UsageError as e:
        bt.logging.warning(e)
        bt.logging.warning("Did you run wandb login?")
        return

    # sign the run to prove it's from this  hotkey
    signature = hotkey.sign(run.id.encode()).hex()
    config.signature = signature
    wandb.config.update(config, allow_val_change=True)

    bt.logging.success(f"Started wandb run {run_name}")
    return run


def clean_wandb_cache(wandb_dir, hours=1):
    """
    Cleans wandb runs except recent ones and latest-run.

    Args:
        wandb_dir: Directory containing wandb run files
        hours: Number of hours to keep runs for (default: 1)
    """
    if not wandb_dir.endswith("wandb"):
        wandb_dir = os.path.join(wandb_dir, "wandb")

    if not os.path.exists(wandb_dir):
        bt.logging.warning(f"W&B directory not found: {wandb_dir}")
        return

    bt.logging.info(f"Attempting to clean wandb cache at {wandb_dir}")
    run_dirs = [
        d for d in glob.glob(os.path.join(wandb_dir, "run-*")) if os.path.isdir(d)
    ]

    if not run_dirs:
        bt.logging.info("No W&B runs found.")
        return

    # Keep recent runs
    current_time = time.time()
    cutoff_time = current_time - (hours * 3600)
    recent_runs = [d for d in run_dirs if os.path.getmtime(d) > cutoff_time]

    # Preserve latest-run target
    latest_run_link = os.path.join(wandb_dir, "latest-run")
    if os.path.exists(latest_run_link) and os.path.isdir(latest_run_link):
        try:
            latest_run_target = os.path.realpath(latest_run_link)
            if latest_run_target not in recent_runs and latest_run_target in run_dirs:
                recent_runs.append(latest_run_target)
                bt.logging.debug(
                    f"Preserving latest-run: {os.path.basename(latest_run_target)}"
                )
        except Exception as e:
            bt.logging.warning(f"Error with latest-run: {e}")

    bt.logging.info(
        f"Keeping {len(recent_runs)} runs (modified in last {hours} hours):"
    )
    for run in recent_runs:
        bt.logging.info(f"  - {os.path.basename(run)}")

    runs_removed = 0
    space_freed = 0
    for run_dir in run_dirs:
        if run_dir not in recent_runs:
            try:
                dir_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, _, filenames in os.walk(run_dir)
                    for filename in filenames
                )
                space_freed += dir_size
                shutil.rmtree(run_dir)
                runs_removed += 1
            except Exception as e:
                bt.logging.warning(f"Error removing {run_dir}: {e}")

    bt.logging.info(
        f"Cleaned {runs_removed} runs, freed {space_freed / (1024*1024):.2f} MB"
    )
