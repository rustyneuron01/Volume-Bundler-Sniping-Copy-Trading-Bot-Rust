# Installation

## Prerequisites

1. **Python 3.10+** (required)
2. **uv** (fast Python package manager)
3. **Git** (for cloning the repository)

### System Dependencies

The installation script will automatically install the following system dependencies (unless `--no-system-deps` is specified):

- **Build tools**: pkg-config, cmake
- **Media processing**: ffmpeg
- **Browser automation**: Google Chrome, libnss3, libnspr4, xvfb
- **Process management**: Node.js, npm, PM2, dotenv

**Note**: The `--no-system-deps` option is primarily intended for **discriminative miners** who only need to submit models and don't require Chrome browser automation or Node.js process management tools. Validators should use the full installation to ensure all dependencies are available.

### Installing uv

Install uv using one of these methods:

```bash
# Method 1: Official installer (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Method 2: Using pip
pip install uv
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/BitMind-AI/bitmind-subnet.git
   cd bitmind-subnet
   ```

2. **Run the installation script**:
   ```bash
   ./install.sh
   ```

   **Options:**
   - `./install.sh --no-system-deps` - Skip system dependency installation (intended for discriminative miners)

The installation script will:
- Check for Python 3.10+ and uv
- Install system dependencies (unless `--no-system-deps` is specified):
  - pkg-config, cmake, ffmpeg
  - chrome web driver, libnss3, libnspr4, xvfb
  - Node.js, npm, PM2, dotenv
- Create a virtual environment using uv (fast dependency resolution)
- Install all dependencies from `pyproject.toml`
- Install the GAS package in development mode
- Install the `gascli` command-line tool
- Install additional git dependencies (Janus)

## Usage

### Activating the Virtual Environment

Before using `gascli`, you need to activate the virtual environment:

```bash
source .venv/bin/activate
```

### CLI Commands

Once the virtual environment is activated, you can use the GAS CLI:

```bash
gascli --help                    # Show main help
gascli validator --help          # Validator commands help
gascli discriminator --help      # Discriminative miner commands help
gascli generator --help          # Generative miner commands help
```

### Available Aliases

- `validator` → `vali`, `v`
- `discriminator` → `d`
- `generator` → `gen`, `g`

### Global Commands

```bash
# Show all services status
gascli status
```

### Alternative Usage (without activation)

You can also run commands directly without activating the environment:

```bash
.venv/bin/gascli --help
.venv/bin/gascli validator start
``` 
