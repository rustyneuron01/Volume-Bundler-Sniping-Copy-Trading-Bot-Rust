# Discriminative Mining Guide

## Before You Proceed

Follow the [Installation Guide](Installation.md) to set up your environment before proceeding with mining operations.

## Discriminative Mining Overview

- Miners submit binary classifiers that distinguish genuine content from AI-generated or AI-manipulated content across three modalities: **image**, **video**, and **audio**.
- For each evaluation sample, a model receives a media input and must produce a binary prediction $[p_{\text{real}}, p_{\text{synthetic}}]$ -- a probability distribution over two classes.
- Some datasets contain semisynthetic content (e.g., inpainting, faceswaps). For scoring purposes, semisynthetic is treated as `synthetic`.
- Models are evaluated on cloud infrastructure -- miners do not need to host hardware for inference.

## Model Preparation

> **⚠️ Important**: Competition submissions now require **safetensors format**. ONNX is no longer accepted.

Discriminative miners must submit models in **safetensors format**:
- Directory containing: `model_config.yaml`, `model.py`, `*.safetensors`
- ZIP archive of the directory for upload

**📖 [Safetensors Model Specification](https://github.com/bitmind-ai/gasbench/blob/main/docs/Safetensors.md)** - Requirements for model submission

You can submit models for any combination of modalities:
- `image_detector.zip` - Image classification model
- `video_detector.zip` - Video classification model  
- `audio_detector.zip` - Audio classification model

## Pushing Your Model

First, activate the virtual environment:
```bash
source .venv/bin/activate
```

Push your models to the network using the `push` command:

```bash
# Upload all three models
gascli d push \
  --image-model image_detector.zip \
  --video-model video_detector.zip \
  --audio-model audio_detector.zip \
  --wallet-name your_wallet_name \
  --wallet-hotkey your_hotkey_name

# Or upload individual models
gascli d push \
  --image-model image_detector.zip \
  --wallet-name your_wallet_name \
  --wallet-hotkey your_hotkey_name
```

### Command Options

The `push` command accepts several parameters:

```bash
gascli d push \
  --image-model image_detector.zip \
  --video-model video_detector.zip \
  --audio-model audio_detector.zip \
  --wallet-name your_wallet_name \
  --wallet-hotkey your_hotkey_name \
  --netuid 34 \
  --chain-endpoint wss://entrypoint-finney.opentensor.ai:443/ \
  --retry-delay 60
```

**Parameters:**
- `--image-model`: Path to image detector zip file
- `--video-model`: Path to video detector zip file
- `--audio-model`: Path to audio detector zip file
- `--wallet-name`: Bittensor wallet name (default: "default")
- `--wallet-hotkey`: Bittensor hotkey name (default: "default") 
- `--netuid`: Subnet UID (default: 34)
- `--chain-endpoint`: Subtensor network endpoint (default: "wss://entrypoint-finney.opentensor.ai:443/")
- `--retry-delay`: Retry delay in seconds (default: 60)

At least one model (image, video, or audio) must be provided.

---

## Competition Rules and Constraints

### Scoring

Each model is scored per modality using the `sn34_score`, a geometric mean of normalized MCC and Brier score:

$$sn34_{score} = \sqrt{MCC_{norm}^{\alpha} \cdot Brier_{norm}^{\beta}}$$

Where $\alpha = 1.2$ and $\beta = 1.8$. This rewards both discrimination accuracy (MCC) and calibration quality (Brier). See [Incentive Mechanism](Incentive.md) for the full formula.

### Model Requirements

- **Format**: Safetensors only (ONNX is no longer accepted)
- **Three model per modality per hotkey**: You can submit up to three image, three video, and three audio model per registered hotkey

### Sandbox and Import Restrictions

Your `model.py` is checked by a static analyzer and executed in a sandboxed environment. Key allowed imports include `torch`, `torchvision`, `torchaudio`, `transformers`, `timm`, `einops`, `flash_attn`, `PIL`, `cv2`, `scipy`, `numpy`, and `safetensors`. Network access, system calls, serialization libraries, and dynamic code execution are all blocked.

For the complete list of allowed and blocked imports, see the [Safetensors Model Specification](https://github.com/bitmind-ai/gasbench/blob/main/docs/Safetensors.md#allowed-imports).

### Evaluation

- Models are benchmarked on cloud infrastructure (not miner hardware)
- Evaluation runs against a diverse dataset of image samples, video samples, and audio samples per benchmark cycle
- Datasets are refreshed weekly with new GAS-Station data alongside static benchmark datasets

---

## Model Format

For the full model specification including `model_config.yaml` structure, `model.py` requirements, input/output specs per modality, and complete examples, see:

**📖 [Safetensors Model Specification](https://github.com/bitmind-ai/gasbench/blob/main/docs/Safetensors.md)**

In short, your submission ZIP must contain:

```
my_detector.zip
├── model_config.yaml    # Metadata and preprocessing config
├── config.json          # (optional) Include if using AutoModel.from_pretrained()
├── model.py             # Model architecture with load_model() function
└── model.safetensors    # Trained weights
```

Package and push:

```bash
cd my_model/
zip -r ../my_detector.zip model_config.yaml model.py model.safetensors
gascli d push --image-model my_detector.zip
```

---

### What Happens During Push

1. **Model Validation**: The system checks that the zip files are present and valid
2. **Model Upload**: Your model zip files are uploaded to the cloud inference system
3. **Blockchain Registration**: Model metadata is registered on the Bittensor blockchain
4. **Verification**: The system verifies the registration was successful

---

## Evaluation Pipeline

After a successful push, your model goes through a two-stage evaluation process automatically.

### Stage 1: Entrance Exam (`--small` mode)

Before your model is ever scored on the network, it must pass an **entrance exam** — a fast sanity check run against a reduced sample of the benchmark datasets.

- Internally this runs `gasbench run --small`, which downloads one archive per dataset and evaluates roughly 100 samples per dataset
- Your model must achieve **≥ 80% accuracy** averaged across all submitted modalities to pass
- The exam has a **maximum wall-clock timeout of 1 hour 25 minutes** (5,100 seconds); models that exceed this are treated as failed
- The exam runs in an **isolated cloud sandbox** — your code has no network access and cannot interact with the host environment
- ONNX models are additionally scanned for cheat patterns (embedded lookup tables or memorization artifacts) before the exam runs; detection results in an immediate block

**Model status during the exam:**

| Status | Meaning |
|---|---|
| `examining` | Entrance exam is currently running |
| `confirmed` | Exam passed — model is eligible for full benchmarking |
| `exam_failed` | Accuracy below 80% — model will not be scored |
| `blocked` | Cheat pattern detected — model is permanently blocked |

You can use `gasbench run --small` locally to replicate exam conditions before pushing:

```bash
gasbench run --image-model ./my_image_model/ --small
gasbench run --video-model ./my_video_model/ --small
gasbench run --audio-model ./my_audio_model/ --small
```

### Stage 2: Full Benchmark (`--full` mode)

Models that pass the entrance exam are benchmarked against the **complete dataset suite**, which includes:

- All public benchmark datasets across image, video, and audio modalities
- **Private holdout datasets** — curated datasets not visible to miners, used to prevent overfitting to the public benchmark set
- Refreshed weekly with new data from the GAS-Station pipeline

The full benchmark has a **maximum wall-clock timeout of 5 hours** (18,000 seconds) per modality. The benchmark score from this stage determines your **TAO emissions** on Subnet 34. Scores are computed using the `sn34_score` formula (see [Incentive Mechanism](Incentive.md)).

You can simulate a full benchmark run locally (without holdouts) to get a sense of your model's performance:

```bash
gasbench run --image-model ./my_image_model/ --full
```

> **Note**: Local full runs will not include the private holdout datasets used in the actual network evaluation.

### Getting Help

```bash
gascli discriminator --help        # Miner help
gascli d push --help               # Push command help
```

**Note**: Remember to activate the virtual environment first with `source .venv/bin/activate` before running any `gascli` commands.