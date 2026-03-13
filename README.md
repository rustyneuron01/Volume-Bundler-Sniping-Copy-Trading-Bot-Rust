<div align="center">
  <img src="docs/static/bm-logo-black.png" alt="BitMind Logo" width="120"/>
  
  <h1>GAS<br><small>Generative Adversarial Subnet</small></h1>
  <h3><code>Bittensor SN34</code></h3>
  <p>
    <a href="docs/Mining.md">⛏️ Mining</a> ·
    <a href="docs/Validating.md">🛡️ Validating</a> ·
    <a href="docs/Incentive.md">💰 Incentives</a> ·
    <a href="https://app.bitmind.ai/">🏆 Leaderboard</a>
  </p>
  <p>
    🤗 <a href="https://huggingface.co/gasstation">GAS-Station</a> ·
    <a href="https://www.bitmind.ai/apps">🌐 Apps</a>
  </p>
</div>

## About GAS
<div align="center">
<em>Fake content is evolving fast. Staying ahead demands relentless innovation.</em><br><br>
</div>

**GAS (Generative Adversarial Subnet)** is a Bittensor subnet inspired by Generative Adversarial Networks (GANs). Detectors and generators compete in a dynamic loop: detectors sharpen their ability to spot synthetic media, while generators push to create more convincing fakes. This adversarial process drives cutting-edge detection tools and continuously generates the training data needed to sustain progress.

Unlike static AI safety solutions, GAS thrives on open, incentivized competition, ensuring detectors evolve as fast as the threats they face.


## Competition Overview

GAS runs two parallel competition tracks on Bittensor Subnet 34:

| Track | What You Do | How You're Scored |
|-------|-------------|-------------------|
| **Discriminative Mining** | Submit AI-generated content detection models (image, video, audio) | `sn34_score` -- geometric mean of MCC and Brier score, measuring both accuracy and calibration |
| **Generative Mining** | Run a server that generates synthetic media on demand | Base reward for valid content × multiplier for fooling discriminators |

**Key facts:**
- **Three modalities**: Image, video, and audio detection are all scored independently
- **Cloud-evaluated**: Discriminator models are benchmarked on cloud infrastructure -- no GPU hosting required
- **Model format**: Safetensors only (ONNX is deprecated)
- **Datasets refresh weekly** with fresh GAS-Station data alongside static benchmarks
- **One model per modality per hotkey** for discriminative miners

See [Incentive Mechanism](docs/Incentive.md) for full scoring details.


## Quick Start

### Installation

```bash
git clone https://github.com/BitMind-AI/bitmind-subnet.git
cd bitmind-subnet
./install.sh
```

**Options:**
- `./install.sh --no-system-deps` - Skip system dependency installation (intended for discriminative miners)

### Using gascli
```bash
# Activate virtual environment to use gascli
source .venv/bin/activate

# Show available commands
gascli --help

# Validators: Start or restart validator services
gascli validator start

# Miners: Start or restart generative miner
gascli generator start

# Miners: Push discriminator models (all three modalities at once)
gascli d push \
  --image-model image_detector.zip \
  --video-model video_detector.zip \
  --audio-model audio_detector.zip \
  --wallet-name default --wallet-hotkey default

# Or push one model at a time
gascli d push --image-model image_detector.zip
gascli d push --video-model video_detector.zip
gascli d push --audio-model audio_detector.zip
```

**Available Aliases:**
- `validator` → `vali`, `v`
- `discriminator` → `d`
- `generator` → `gen`, `g`


### Not using gascli
```bash
# Validators: Start or restart validator services
# (Does not require virtualenv activation)
pm2 start validator.config.js  

# Miners: Start or restart generative miner
pm2 start gen_miner.config.js

# Miners: Push discriminator models
source .venv/bin/activate
python neurons/discriminator/push_model.py \
  --image-model image_detector.zip \
  --video-model video_detector.zip \
  --audio-model audio_detector.zip \
  --wallet-name default --wallet-hotkey default
```
For detailed installation and usage instructions, see [Installation Guide](docs/Installation.md).


## Core Components

> This documentation assumes basic familiarity with [Bittensor concepts](https://docs.bittensor.com/learn/bittensor-building-blocks). 

#### Discriminative Miners [[docs](docs/Discriminative-Mining.md)]
Discriminative miners submit detection models for evaluation against a wide variety of real and synthetic media across **image, video, and audio** modalities. Models are evaluated on cloud infrastructure and rewarded based on their accuracy and calibration. This significantly reduces the capital required to mine compared to previous versions that required GPU hosting, and allows the subnet to more reliably identify unique models and reward novel contributions proportionally to their accuracy.

#### Generative Miners [[docs](docs/Generative-Mining.md)]

Generative miners generate synthetic images and videos according to prompts from validators, and are rewarded based on their ability to pass validation checks and fool discriminative miners.

#### Validators [[docs](docs/Validating.md)]
Validators are responsible for challenging and scoring both miner types. Generative miners are sent prompts, and their returned synthetic media are validated to mitigate gaming and incentivize high quality results. Discriminative miners are continually evaluated against a mix of data from generative miners, real world data, and data generated locally on the validator.


## Subnet Architecture
![Subnet Architecture](docs/static/GAS-Architecture-Simple.png)

## Community

<p align="left">
  <a href="https://discord.gg/kKQR98CrUn">
    <img src="docs/static/Join-BitMind-Discord.png" alt="Join us on Discord" width="60%">
  </a>
</p> 

## Contributing

Contributions are welcome and can be made via a pull request to the `testnet` branch.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/BitMind-AI/bitmind-subnet)
