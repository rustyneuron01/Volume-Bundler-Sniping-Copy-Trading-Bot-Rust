# Generative Mining Guide

## Before You Proceed

Follow the [Installation Guide](Installation.md) to set up your environment before proceeding with generative miner setup.

## Generative Mining Overview

Generative miners create synthetic media (images and videos) according to prompts from validators. Miners are rewarded based on:
1. **Data validation pass rate** -- content must pass C2PA signature checks and prompt alignment validation
2. **Adversarial performance** -- synthetic media that fools discriminative miners earns a multiplier bonus
3. **Sample volume** -- processing more evaluations provides a logarithmic bonus up to 2x

See [Incentive Mechanism](Incentive.md) for the full reward formula.

Generative miners operate as FastAPI servers that receive generation requests from validators and respond asynchronously via webhooks.

To avoid gaming, validation currently requires that all generated media have valid C2PA metadata signed by trusted services. See [gas/verification/c2pa_verification.py](../gas/verification/c2pa_verification.py) for the list of trusted signers. 

To check a file locally (uses same logic as validator): `gascli generator verify-c2pa <file> --verbose`.

## Configuration Setup

### Environment Configuration

Create a `.env.gen_miner` file in the project root: copy from [.env.gen_miner.template](../.env.gen_miner.template) and fill in your wallet, API keys, and options.

### Network Configuration

**Mainnet (SN34)**:
```bash
BT_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
BT_NETUID=34
```

## Generation Services

You must configure at least one generation method for your miner to be functional. 

### OpenAI Service (DALL-E)
- **API Key**: `OPENAI_API_KEY`
- **Supported**: Image generation
- **Models**: DALL-E 3, DALL-E 2

### OpenRouter Service  
- **API Key**: `OPEN_ROUTER_API_KEY`
- **Supported**: Image and video generation
- **Models**: Google Gemini Flash Image Preview, various other models
- **Website**: [OpenRouter.ai](https://openrouter.ai)

### Stability AI Service
- **API Key**: `STABILITY_API_KEY`
- **Supported**: Image generation
- **Models**: Stable Diffusion, Stable Image Ultra, and other Stability AI models
- **Website**: [stability.ai](https://stability.ai)

### Service Selection

Configure which service handles each modality in your `.env.gen_miner` file:

```bash
IMAGE_SERVICE=openrouter    # openai, openrouter, stabilityai, or none
VIDEO_SERVICE=openrouter    # openrouter or none
```

All configured services must produce **C2PA-signed content**. Setting a modality to `none` disables it (requests will be rejected).


## Starting the Miner

### Using gascli (Recommended)

First, activate the virtual environment:
```bash
source .venv/bin/activate
```

Start the generative miner:
```bash
# Start the miner
gascli generator start

# Using aliases
gascli gen start
gascli g start
```

You can also start the miner using PM2 directly:
```bash
pm2 start gen_miner.config.js
```

## Miner Management

### Status and Monitoring

```bash
# Check miner status
gascli generator status

# View miner logs  
gascli generator logs

# Follow logs in real-time
gascli generator logs --follow

# Show miner configuration and API key status
gascli generator info
```

### Starting and Stopping

```bash
# Start the miner
gascli generator start

# Stop the miner
gascli generator stop

# Restart the miner
gascli generator restart

# Delete the miner process
gascli generator delete
```

### API Endpoints

Your miner exposes these endpoints for validators:

- `POST /gen_image` - Image generation requests
- `POST /gen_video` - Video generation requests
- `GET /health` - Health check endpoint
- `GET /miner_info` - Miner information and capabilities
- `GET /status/{task_id}` - Task status queries

## Configuration Parameters

### Core Settings

- `BT_WALLET_NAME`: Your Bittensor wallet name
- `BT_WALLET_HOTKEY`: Your wallet hotkey name  
- `BT_NETUID`: Subnet ID (34 for mainnet, 379 for testnet)
- `BT_AXON_PORT`: Port for your miner's API server

### Performance Settings

- `MINER_MAX_CONCURRENT_TASKS`: Maximum parallel generation tasks (default: 5)
- `MINER_TASK_TIMEOUT`: Maximum time per task in seconds (default: 300)
- `MINER_OUTPUT_DIR`: Directory for generated content and logs
- `MINER_DEVICE`: Computing device (`auto`, `cuda`, `cpu`) [Deprecated: local modals currently not supported]


```bash
# General help
gascli generator --help

# Command-specific help  
gascli generator start --help
gascli generator logs --help

# Check service status
gascli generator info
```

**Note**: Remember to activate the virtual environment with `source .venv/bin/activate` before running any `gascli` commands. 
