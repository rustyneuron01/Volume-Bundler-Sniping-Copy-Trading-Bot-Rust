# Validator Guide

## Required .env.validator variables (PM2 and Docker)

You **must** set these in `.env.validator` before starting; the rest have defaults in the template.

| Variable | Required? | Notes |
|----------|-----------|--------|
| `WALLET_NAME` | **Yes** | Your Bittensor wallet name (e.g. `default` or a name you created). |
| `WALLET_HOTKEY` | **Yes** | Hotkey for this validator (e.g. `default`). Must be registered on the subnet. |
| `HUGGINGFACE_HUB_TOKEN` | **Yes** | Needed to download models and upload data. Create at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Reach out to a member of the BitMind team to have your account added to the `gasstation` org |
| `CHAIN_ENDPOINT` | No | Default is mainnet. Set to testnet URL if needed. |
| `WANDB_API_KEY` | No | Optional; for Weights & Biases logging. |
| Cache / Docker paths | No | `SN34_CACHE_DIR`, `HF_HOME`, `WALLET_PATH`, `BT_LOGGING_LOGGING_DIR` have defaults; override if you use different paths. |

See [.env.validator.template](../.env.validator.template) for every option.

---

## PM2 setup

Follow the [Installation Guide](Installation.md) to set up your environment before proceeding with validator setup.

> Create a `.env.validator` file and **set the required variables above** (wallet, API keys). Then:

```bash
$ cp .env.validator.template .env.validator
# Edit .env.validator: fill in WALLET_NAME, WALLET_HOTKEY, HUGGINGFACE_HUB_TOKEN at minimum
```

See [.env.validator.template](../.env.validator.template) for all options. Then activate the virtual environment and start your validator processes:

```bash
$ source .venv/bin/activate
$ gascli validator start
```
The above command will create 3 pm2 processes:
```bash
┌────┬───────────────────┬─────────────┬─────────┬─────────┬──────────┬────────┬──────┬───────────┬──────────┬──────────┬──────────┬──────────┐
│ id │ name              │ namespace   │ version │ mode    │ pid      │ uptime │ ↺    │ status    │ cpu      │ mem      │ user     │ watching │
├────┼───────────────────┼─────────────┼─────────┼─────────┼──────────┼────────┼──────┼───────────┼──────────┼──────────┼──────────┼──────────┤
│ 2  │ sn34-data         │ default     │ N/A     │ fork    │ 4032914  │ 2s     │ 72   │ online    │ 100%     │ 529.0mb  │ user     │ disabled │
│ 1  │ sn34-generator    │ default     │ N/A     │ fork    │ 4032936  │ 2s     │ 72   │ online    │ 100%     │ 448.5mb  │ user     │ disabled │
│ 0  │ sn34-validator    │ default     │ N/A     │ fork    │ 4032918  │ 2s     │ 72   │ online    │ 100%     │ 504.0mb  │ user     │ disabled │
└────┴───────────────────┴─────────────┴─────────┴─────────┴──────────┴────────┴──────┴───────────┴──────────┴──────────┴──────────┴──────────┘
```
- **sn34-data**: Handles data downloads
- **sn34-generator**: Responsible for generating prompts, synthetic media, and validating miner-generated data
- **sn34-validator**: Core validator logic. Challenges, scoring, weight setting.


### Validator Operations

First, activate the virtual environment:
```bash
source .venv/bin/activate
```

Then run validator commands:
```bash
# Start validator services
gascli validator start
gascli v start                   # Using alias
gascli v stop
gascli v status
gascli v logs
gascli v --help
```


## Docker Deployment

As an alternative to the PM2-based setup above, you can run the validator stack in Docker. Three containers (validator, generator, data) run one process each and share bind-mounted cache and wallet; the validator container uses `network_mode: host` so its FastAPI callback is on the host port miners are told to hit.

### Prerequisites

- **Docker** (with Docker Compose v2)
- **NVIDIA Container Toolkit** for GPU passthrough. Install guide: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- **Bittensor wallet** files already created on the host (typically at `~/.bittensor/wallets/`)

### Quick Start

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd bitmind-subnet
   ```

2. **Create your `.env.validator`** and set the **required variables** (same as PM2):
   ```bash
   cp .env.validator.template .env.validator
   ```
   You **must** set: `WALLET_NAME`, `WALLET_HOTKEY`, `HUGGINGFACE_HUB_TOKEN`, `OPEN_ROUTER_API_KEY`. See [Required .env.validator variables](#required-envvalidator-variables-pm2-and-docker) and [.env.validator.template](../.env.validator.template) for the full list.

3. **Build and start** (use `--env-file .env.validator` so the same file drives both container env and Compose options like `WALLET_PATH` and `CALLBACK_PORT`):
   ```bash
   docker compose --env-file .env.validator up -d --build
   ```

4. **View logs** (per service):
   ```bash
   docker compose logs -f validator   # or generator, data
   ```


### Updating

**Automatic (recommended):** add a crontab entry so `docker/autoupdate.sh` runs periodically (it checks VERSION, then pulls, runs `down --remove-orphans`, rebuilds, and brings the stack up):

```bash
*/5 * * * * /path/to/bitmind-subnet/docker/autoupdate.sh >> /var/log/bitmind-docker-update.log 2>&1
```

**Manual:** rebuild and recreate all services:

```bash
git pull
docker compose --env-file .env.validator down --remove-orphans
docker compose --env-file .env.validator build
docker compose --env-file .env.validator up -d
```

### Configuration

| Variable | Docker behavior |
|---|---|
| `SN34_CACHE_DIR` | Bind-mounted into the container; same path as PM2 so cache is shared when you switch. Set in `.env.validator`; use `--env-file .env.validator`. |
| `HF_HOME` | Bind-mounted into the container; same path as PM2 so model cache is shared. Set in `.env.validator`. |
| `AUTO_UPDATE` | In-container autoupdate is disabled. Use manual rebuild or the cron-based `docker/autoupdate.sh` (see Updating). |
| `WALLET_PATH` | Host base path for wallets. Only the directory `WALLET_PATH/WALLET_NAME` is mounted (not the whole wallets folder). Set in `.env.validator`. |
| `WALLET_NAME` | Which wallet dir to mount; with `WALLET_PATH` only that wallet is visible to the container. |
| `BT_LOGGING_LOGGING_DIR` | Base path for validator state (scores, challenge tasks) and bittensor logs. State lives under `BT_LOGGING_LOGGING_DIR`/`WALLET_NAME`/`WALLET_HOTKEY`/…; bind-mounted so it persists. Default `~/.bittensor`. |
| `NETUID` | Auto-derived from `CHAIN_ENDPOINT`. Set explicitly if using a custom endpoint. |

### Wallet, cache, and validator state

- **Wallet:** Only the configured wallet directory (`WALLET_PATH`/`WALLET_NAME`) is bind-mounted, not the entire wallets folder.
- **Cache:** `SN34_CACHE_DIR` and `HF_HOME` are bind-mounted so PM2 and Docker share the same data (no re-download when switching).
- **Validator state:** Scores and challenge tasks are saved under bittensor’s logging path (`BT_LOGGING_LOGGING_DIR`/`WALLET_NAME`/`WALLET_HOTKEY`/…). That path is bind-mounted so state persists across container restarts.

> **Note**: The first startup will download 100+ GB of ML models into the HF cache directory. Subsequent restarts reuse the cached models.

### Common Operations

Use `--env-file .env.validator` so Compose reads `WALLET_PATH`, `WALLET_NAME`, `CALLBACK_PORT`, and cache paths from your config:

```bash
# Start all three services (validator, generator, data)
docker compose --env-file .env.validator up -d

# Stop all
docker compose down

# View logs (per service)
docker compose logs -f validator   # or generator, data

# Rebuild after code changes
docker compose --env-file .env.validator down --remove-orphans
docker compose --env-file .env.validator build && docker compose --env-file .env.validator up -d

# Restart one service
docker compose restart validator   # or generator, data

# Container status
docker compose ps

# Shell into a container
docker compose exec validator bash   # or generator, data
```
