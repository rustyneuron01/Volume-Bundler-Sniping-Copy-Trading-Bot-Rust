#!/usr/bin/env bash
# Entrypoint for bitmind-subnet validator stack.
# Set SERVICE=validator|generator|data to run one process per container.
# All three share the same image; compose runs three containers with shared volumes.

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
: "${WALLET_NAME:=default}"
: "${WALLET_HOTKEY:=default}"
: "${CHAIN_ENDPOINT:=wss://entrypoint-finney.opentensor.ai:443}"
: "${CALLBACK_PORT:=10525}"
: "${DEVICE:=cuda}"
: "${LOGLEVEL:=info}"
: "${BENCHMARK_API_URL:=https://gas.bitmind.ai}"
: "${SCRAPER_INTERVAL:=300}"
: "${DATASET_INTERVAL:=1800}"
: "${HEARTBEAT:=false}"
: "${WANDB_API_KEY:=}"

# ── Derive netuid from chain endpoint ───────────────────────────────────────
if [[ -z "${NETUID:-}" ]]; then
    if [[ "$CHAIN_ENDPOINT" == *"test"* ]]; then
        NETUID=379
    elif [[ "$CHAIN_ENDPOINT" == *"finney"* ]]; then
        NETUID=34
    else
        echo "ERROR: Cannot derive NETUID from CHAIN_ENDPOINT=$CHAIN_ENDPOINT"
        echo "       Set NETUID explicitly in your .env.validator file."
        exit 1
    fi
fi

# ── Map log level to CLI param ──────────────────────────────────────────────
case "${LOGLEVEL,,}" in
    trace) LOG_PARAM="--logging.trace" ;;
    debug) LOG_PARAM="--logging.debug" ;;
    *)     LOG_PARAM="--logging.info"  ;;
esac

# ── Build validator command ─────────────────────────────────────────────────
VALIDATOR_CMD="/app/.venv/bin/python neurons/validator/validator.py"
VALIDATOR_CMD+=" --wallet.name ${WALLET_NAME}"
VALIDATOR_CMD+=" --wallet.hotkey ${WALLET_HOTKEY}"
VALIDATOR_CMD+=" --netuid ${NETUID}"
VALIDATOR_CMD+=" --subtensor.chain_endpoint ${CHAIN_ENDPOINT}"
VALIDATOR_CMD+=" --neuron.callback_port ${CALLBACK_PORT}"
VALIDATOR_CMD+=" --cache.base-dir /root/.cache/sn34"
VALIDATOR_CMD+=" --benchmark.api-url ${BENCHMARK_API_URL}"
VALIDATOR_CMD+=" ${LOG_PARAM}"
VALIDATOR_CMD+=" --autoupdate-off"

if [[ -n "${EXTERNAL_CALLBACK_PORT:-}" ]]; then
    VALIDATOR_CMD+=" --neuron.external-callback-port ${EXTERNAL_CALLBACK_PORT}"
fi

if [[ "${HEARTBEAT,,}" == "true" ]]; then
    VALIDATOR_CMD+=" --heartbeat"
fi

# ── Build generator command ─────────────────────────────────────────────────
GENERATOR_CMD="/app/.venv/bin/python neurons/validator/services/generator_service.py"
GENERATOR_CMD+=" --wallet.name ${WALLET_NAME}"
GENERATOR_CMD+=" --wallet.hotkey ${WALLET_HOTKEY}"
GENERATOR_CMD+=" --cache.base-dir /root/.cache/sn34"
GENERATOR_CMD+=" --device ${DEVICE}"
GENERATOR_CMD+=" --log-level ${LOGLEVEL}"

# ── Build data service command ──────────────────────────────────────────────
DATA_CMD="/app/.venv/bin/python neurons/validator/services/data_service.py"
DATA_CMD+=" --wallet.name ${WALLET_NAME}"
DATA_CMD+=" --wallet.hotkey ${WALLET_HOTKEY}"
DATA_CMD+=" --netuid ${NETUID}"
DATA_CMD+=" --subtensor.chain_endpoint ${CHAIN_ENDPOINT}"
DATA_CMD+=" --cache.base-dir /root/.cache/sn34"
DATA_CMD+=" --scraper-interval ${SCRAPER_INTERVAL}"
DATA_CMD+=" --dataset-interval ${DATASET_INTERVAL}"
DATA_CMD+=" ${LOG_PARAM}"

# ── Shared env ──────────────────────────────────────────────────────────────
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"
mkdir -p /root/.cache/sn34/tmp /root/.cache/huggingface

# ── Run one service ────────────────────────────────────────────────────────
case "${SERVICE:-}" in
    validator)
        echo "========================================"
        echo "  bitmind-subnet Validator"
        echo "  WALLET: ${WALLET_NAME}/${WALLET_HOTKEY}  NETUID: ${NETUID}  CALLBACK: ${CALLBACK_PORT}"
        echo "========================================"
        export WANDB_API_KEY="${WANDB_API_KEY}"
        exec /bin/bash -c "cd /app && $VALIDATOR_CMD"
        ;;
    generator)
        echo "========================================"
        echo "  bitmind-subnet Generator"
        echo "  WALLET: ${WALLET_NAME}/${WALLET_HOTKEY}  DEVICE: ${DEVICE}"
        echo "========================================"
        exec /bin/bash -c "cd /app && $GENERATOR_CMD"
        ;;
    data)
        echo "========================================"
        echo "  bitmind-subnet Data Service"
        echo "  WALLET: ${WALLET_NAME}/${WALLET_HOTKEY}  NETUID: ${NETUID}"
        echo "========================================"
        Xvfb :99 -screen 0 1280x1024x24 -nolisten tcp &
        export DISPLAY=:99
        export TMPDIR=/root/.cache/sn34/tmp TEMP=/root/.cache/sn34/tmp TMP=/root/.cache/sn34/tmp
        exec /bin/bash -c "cd /app && $DATA_CMD"
        ;;
    *)
        echo "ERROR: Set SERVICE=validator|generator|data (e.g. in docker-compose)"
        exit 1
        ;;
esac
