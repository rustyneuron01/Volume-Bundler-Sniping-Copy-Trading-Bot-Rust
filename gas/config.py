import os

import bittensor as bt

MAINNET_UID = 34
TESTNET_UID = 379


def validate_config_and_neuron_path(config):
    r"""Checks/validates the config namespace object."""
    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )
    bt.logging.info(f"Logging path: {full_path}")
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)
    return config


def add_args(parser):
    """
    Adds relevant arguments to the parser for operation.
    """
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=TESTNET_UID)

    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Neuron Name",
        default="bitmind",
    )

    parser.add_argument(
        "--neuron.callback_port",
        type=int,
        help="Port for receiving webhook callbacks from miners",
        default=10525,
    )

    parser.add_argument(
        "--epoch-length",
        type=int,
        help="The default epoch length (how often we set weights, measured in 12 second blocks).",
        default=360,
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode",
        default=False,
    )

    parser.add_argument(
        "--autoupdate-off",
        action="store_false",
        dest="autoupdate",
        help="Disable automatic updates on latest version on Main.",
        default=True,
    )

    parser.add_argument(
        "--cache.base-dir",
        type=str,
        default=os.path.expanduser("~/.cache/sn34"),
        help="Base directory for cache storage",
    )

    parser.add_argument("--wandb.entity", type=str, default="bitmindai")

    parser.add_argument("--wandb.off", action="store_true", default=False)


# Shared source-limit/demand-loading args


def add_source_limit_args(parser):
    parser.add_argument(
        "--max-per-source",
        type=int,
        help="Maximum number of media items per source (dataset/scraper/model) (default: 1000)",
        default=1000,
    )

    parser.add_argument(
        "--enable-source-limits",
        action="store_true",
        help="Enable per-source maximum count limits",
        default=True,
    )

    parser.add_argument(
        "--prune-strategy",
        type=str,
        choices=["oldest", "least_used", "random"],
        help="Strategy for pruning media when limits are exceeded (default: oldest)",
        default="oldest",
    )

    parser.add_argument(
        "--min-source-threshold",
        type=float,
        help="Minimum items per source as fraction of max-per-source (default: 0.8 = 80%)",
        default=0.8,
    )

    parser.add_argument(
        "--remove-on-sample",
        action="store_true",
        help="Remove media items when sampled (instead of pruning on add)",
        default=False,
    )


def add_miner_args(parser):
    """Add miner specific arguments to the parser."""

    parser.add_argument(
        "--no-force-validator-permit",
        action="store_true",
        help="If set, we will not force incoming requests to have a permit.",
        default=False,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for detection models (cuda/cpu)",
    )

    # === GENERATIVE MINER SPECIFIC ARGS ===
    parser.add_argument(
        "--miner.output-dir",
        type=str,
        default="generated_content",
        help="Directory to store generated content",
    )

    parser.add_argument(
        "--miner.max-task-age-hours",
        type=int,
        default=24,
        help="Maximum age (hours) before tasks are cleaned up",
    )

    parser.add_argument(
        "--miner.max-concurrent-tasks",
        type=int,
        default=5,
        help="Maximum number of tasks to process concurrently",
    )

    parser.add_argument(
        "--miner.task-timeout",
        type=float,
        default=300.0,
        help="Maximum time (seconds) to spend on a single task",
    )

    parser.add_argument(
        "--miner.cleanup-interval",
        type=int,
        default=3600,
        help="Interval (seconds) between task cleanup runs",
    )

    parser.add_argument(
        "--miner.worker-threads",
        type=int,
        default=2,
        help="Number of worker threads for task processing",
    )
    
    # Webhook configuration
    parser.add_argument(
        "--miner.webhook-max-retries",
        type=int,
        default=3,
        help="Maximum number of webhook retry attempts",
    )
    parser.add_argument(
        "--miner.webhook-retry-delay",
        type=float,
        default=2.0,
        help="Base delay between webhook retries (exponential backoff)",
    )
    parser.add_argument(
        "--miner.webhook-timeout",
        type=float,
        default=30.0,
        help="Timeout for webhook requests in seconds",
    )


def add_validator_args(parser):
    """Add validator specific arguments to the parser."""

    parser.add_argument(
        "--vpermit-tao-limit",
        type=int,
        help="The maximum number of TAO allowed to query a validator with a vpermit.",
        default=20000,
    )

    parser.add_argument(
        "--generator-challenge-interval",
        type=int,
        help="How often we send challenges to generative miners, measured in 12 second blocks.",
        default=110,
    )

    parser.add_argument(
        "--dataset-download-interval",
        type=int,
        help="How often to download and extract datasets, measured in 12 second blocks (default: 720 = 2.4 hours)",
        default=1800,
    )

    parser.add_argument(
        "--wandb-restart-interval",
        type=int,
        help="How often we restart wandb run to avoid log truncation",
        default=2000,
    )

    parser.add_argument(
        "--neuron.callback-port",
        type=int,
        help="Port for generative challenge callbacks (internal binding)",
        default=10525,
    )

    parser.add_argument(
        "--neuron.external-callback-port",
        type=int,
        help="External port to advertise to miners for callbacks (defaults to callback-port)",
        default=None,
    )

    parser.add_argument(
        "--neuron.max-state-backup-hours",
        type=float,
        help="The oldest backup of validator state to load in the case of a failure to load most recent",
        default=1,
    )

    parser.add_argument(
        "--neuron.miner-total-timeout",
        type=float,
        help="Total timeout for miner requests in seconds",
        default=240.0,
    )

    parser.add_argument(
        "--neuron.miner-connect-timeout",
        type=float,
        help="TCP connection timeout for miner requests in seconds",
        default=6.0,
    )

    parser.add_argument(
        "--neuron.miner-sock-connect-timeout",
        type=float,
        help="Socket connection timeout for miner requests in seconds",
        default=5.0,
    )

    parser.add_argument(
        "--neuron.heartbeat",
        action="store_true",
        help="Run validator heartbeat thread",
        default=False,
    )

    parser.add_argument(
        "--neuron.heartbeat-interval-seconds",
        type=float,
        help="Interval between heartbeat checks in seconds",
        default=60.0,
    )

    parser.add_argument(
        "--neuron.lock-sleep-seconds",
        type=float,
        help="Sleep duration when lock is held in seconds",
        default=5.0,
    )

    parser.add_argument(
        "--neuron.max-stuck-count",
        type=int,
        help="Number of consecutive heartbeats with no progress before restart",
        default=5,
    )

    parser.add_argument(
        "--neuron.sample-size",
        type=int,
        help="Number of miners to query per challenge",
        default=50,
    )

    parser.add_argument(
        "--scoring.moving-average-alpha",
        type=float,
        help="Alpha for miner score EMA",
        default=0.05,
    )

    parser.add_argument(
        "--scoring.window",
        type=int,
        help="Number of recent predictions to consider in evaluation",
        default=200,
    )

    parser.add_argument(
        "--scoring.image-weight",
        type=float,
        help="Weight for image modality scoring",
        default=0.6,
    )

    parser.add_argument(
        "--scoring.video-weight",
        type=float,
        help="Weight for video modality scoring",
        default=0.4,
    )

    parser.add_argument(
        "--scoring.binary-weight",
        type=float,
        help="Weight for binary classification scoring",
        default=0.75,
    )

    parser.add_argument(
        "--benchmark.api-url",
        type=str,
        help="Base URL for the benchmark API",
        default=os.environ.get("BENCHMARK_API_URL", "https://gas.bitmind.ai"),
    )

    parser.add_argument(
        "--scoring.multiclass-weight",
        type=float,
        help="Weight for multiclass classification scoring",
        default=0.25,
    )

    parser.add_argument(
        "--challenge.image-prob",
        type=float,
        help="Probability of selecting image modality for challenges",
        default=0.5,
    )

    parser.add_argument(
        "--challenge.video-prob",
        type=float,
        help="Probability of selecting video modality for challenges",
        default=0.5,
    )

    parser.add_argument(
        "--challenge.real-prob",
        type=float,
        help="Probability of selecting real media for challenges",
        default=0.4,
    )

    parser.add_argument(
        "--challenge.synthetic-prob",
        type=float,
        help="Probability of selecting synthetic media for challenges",
        default=0.3,
    )

    parser.add_argument(
        "--challenge.semisynthetic-prob",
        type=float,
        help="Probability of selecting semisynthetic media for challenges",
        default=0.3,
    )

    parser.add_argument(
        "--challenge.generator-prob",
        type=float,
        help="Probability of selecting generator content for synthetic and semisynthetic challenges",
        default=0.5,
    )

    parser.add_argument(
        "--challenge.multi-video-prob",
        type=float,
        help="Probability of stitching together two videos of the same media type",
        default=0.2,
    )

    parser.add_argument(
        "--challenge.min-clip-duration",
        type=float,
        help="Minimum video clip duration in seconds",
        default=1.0,
    )

    parser.add_argument(
        "--challenge.max-clip-duration",
        type=float,
        help="Maximum video clip duration in seconds",
        default=6.0,
    )

    parser.add_argument(
        "--challenge.max-frames",
        type=int,
        help="Maximum number of video frames to sample for a challenge",
        default=24,
    )


def add_generation_service_args(parser):
    """Add generation service specific arguments to the parser."""

    parser.add_argument(
        "--device",
        type=str,
        help="Device to use for generation models (cuda/cpu)",
        default="cuda",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for generation",
        default=1,
    )

    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        help="List of generation tasks to run",
        default=["search_query", "prompt", "t2i", "i2i", "t2v", "i2v"],
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        help="Maximum number of retries for generation",
        default=3,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout for generation operations in seconds",
        default=300,
    )

    parser.add_argument(
        "--prompt-batch-size",
        type=int,
        help="Number of prompts to generate per cycle",
        default=20,
    )

    parser.add_argument(
        "--query-batch-size",
        type=int,
        help="Number of search queries to generate per cycle",
        default=10,
    )

    parser.add_argument(
        "--local-batch-size",
        type=int,
        help="Number of local generations to run per cycle",
        default=1,
    )

    parser.add_argument(
        "--tps-batch-size",
        type=int,
        help="Number of third-party service generations to run per cycle",
        default=10,
    )

    # Shared source-limit args
    add_source_limit_args(parser)


def add_data_service_args(parser):
    """Add data service specific arguments to the parser."""

    parser.add_argument(
        "--scraper-interval",
        type=int,
        help="Scraper interval in blocks (default: 300 = 1 hour)",
        default=300,
    )

    parser.add_argument(
        "--upload-check-interval",
        type=int,
        help="Upload interval  in blocks (default: 300 = 1 hour)",
        default=300,
    )

    parser.add_argument(
        "--dataset-interval",
        type=int,
        help="Dataset download interval in blocks (default: 1800 = 6 hours)",
        default=1800,
    )

    parser.add_argument(
        "--exclude-tags",
        type=str,
        help="Comma-separated list of tags to exclude from downloads",
        default="",
    )

    parser.add_argument(
        "--scraper-batch-size",
        type=int,
        help="Batch size for scraper operations",
        default=10,
    )

    parser.add_argument(
        "--dataset-images-per-parquet",
        type=int,
        help="Number of images to extract per parquet file (default: 100)",
        default=100,
    )

    parser.add_argument(
        "--dataset-videos-per-zip",
        type=int,
        help="Number of videos to extract per zip file (default: 50)",
        default=200,
    )

    parser.add_argument(
        "--dataset-parquet-per-dataset",
        type=int,
        help="Number of parquet files to download per dataset (default: 5)",
        default=2,
    )

    parser.add_argument(
        "--dataset-zips-per-dataset",
        type=int,
        help="Number of zip files to download per dataset (default: 2)",
        default=2,
    )

    parser.add_argument(
        "--upload-batch-size",
        type=int,
        help="Maximum number of media files to upload to HuggingFace per batch per modality",
        default=1000,
    )

    parser.add_argument(
        "--upload-threshold",
        type=int,
        help="Minimum number of unuploaded media files to trigger an upload cycle",
        default=1000,
    )

    parser.add_argument(
        "--upload-max-batches",
        type=int,
        help="Maximum number of upload batches to process per cycle",
        default=5,
    )

    parser.add_argument(
        "--upload-max-duration-hours",
        type=float,
        help="Safety measure:Maximum hours an uploader thread may run before being treated as hung (default: 2)",
        default=2.0,
    )

    parser.add_argument(
        "--images-per-archive",
        type=int,
        help="Number of images per tar archive (target ~100-200MB per archive)",
        default=500,
    )

    parser.add_argument(
        "--videos-per-archive",
        type=int,
        help="Number of videos per tar archive (target ~650MB per archive)",
        default=200,
    )

    parser.add_argument(
        "--cleanup-interval",
        type=int,
        help="Cleanup interval in blocks for removing uploaded media (default: 7200 = ~24 hours)",
        default=7200,
    )

    parser.add_argument(
        "--cleanup-min-age-hours",
        type=float,
        help="Minimum age in hours before uploaded media can be cleaned up (default: 48)",
        default=48.0,
    )

    parser.add_argument(
        "--cleanup-batch-size",
        type=int,
        help="Number of media entries to clean up per batch (default: 1000)",
        default=1000,
    )

    # Shared source-limit args
    add_source_limit_args(parser)
